"""Plan-only, reviewed handoff fallback for the immutable checkpoint-9 GB copy.

This script never stops an existing process. Execution is one-shot, normal UID,
fixed source/target/checkpoint/deadline, at most four independent shard writers.
GNU timeout bounds only children created by this invocation; failure retains
partial files, locks, logs, and an audit. No automatic retry or deadline extension.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import datetime as dt
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

try:
    from lab.rubin_two_node.retain_profile_checkpoint import (
        UID, ROOT, DEST, GB_RAID, GB_DEST, GB_NODE, DEADLINE,
        checked, inventory, file_state, checkpoint_metadata,
    )
except ModuleNotFoundError:
    from retain_profile_checkpoint import (
        UID, ROOT, DEST, GB_RAID, GB_DEST, GB_NODE, DEADLINE,
        checked, inventory, file_state, checkpoint_metadata,
    )

ITERATION = 9
EXPECTED_ID = 'b7cb2def84d3abc6614953c215c8b97ab6e26bac5b741a2d3d36fa17bd2b2e80'
GB_PARTIAL = GB_DEST + '.partial'
LOCAL_HOST = 'dlcluster-login-03'
STATE_NAME = 'checkpoint-retention-profile-9-state.json'
STAGE_NAME = 'checkpoint9-gb-stage.json'
AUDIT_NAME = 'checkpoint9-parallel-fallback'
AUTHORIZATION = 'parallel_checkpoint9_copy_after_original_writers_terminal'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Budget:
    """Absolute wall-clock cutoff, additionally protected against clock rollback."""
    def __init__(self, deadline=DEADLINE):
        self.deadline = deadline
        self.monotonic_end = time.monotonic() + deadline - time.time()

    def remaining(self, maximum=None):
        value = min(self.deadline - time.time(), self.monotonic_end - time.monotonic())
        if value <= 0:
            raise TimeoutError('Absolute 05:40 UTC copy/verification/publication budget expired')
        return min(value, maximum) if maximum is not None else value

    @contextmanager
    def enforce(self):
        seconds = self.remaining()
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError('Refuse to replace an existing deadline alarm')
        old = signal.getsignal(signal.SIGALRM)
        def expired(*_):
            raise TimeoutError('Absolute checkpoint fallback deadline expired')
        signal.signal(signal.SIGALRM, expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def write_json(path, value, budget, exclusive=False):
    """A partial audit remains if the budget expires before atomic publication."""
    budget.remaining()
    path = Path(path)
    if path.exists() or path.is_symlink():
        checked(path)
        if exclusive:
            raise FileExistsError(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    budget.remaining()
    temporary.replace(path)


def proc_snapshot():
    """Inspect only ordinary-user processes; unreadable live entries fail closed."""
    found = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != UID:
                continue
            raw_stat = (path / 'stat').read_text().rpartition(') ')[2].split()
            cmdline = (path / 'cmdline').read_bytes()
            writers = []
            for fd in (path / 'fd').iterdir():
                try:
                    destination = os.readlink(fd).removesuffix(' (deleted)')
                    if not any(destination == p or destination.startswith(p + '/') for p in (GB_PARTIAL, GB_DEST)):
                        continue
                    flags = next(line.split()[1] for line in (path / 'fdinfo' / fd.name).read_text().split('\n') if line.startswith('flags:'))
                    if int(flags, 8) & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                        writers.append(destination)
                except FileNotFoundError:
                    continue
            found[int(path.name)] = {
                'pid': int(path.name), 'start_ticks': int(raw_stat[19]), 'state': raw_stat[0],
                'uid': UID, 'cmdline_sha256': hashlib.sha256(cmdline).hexdigest(),
                'argv': [p.decode(errors='replace') for p in cmdline.split(b'\0') if p],
                'writers': writers,
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def check_terminal(identity, processes):
    current = processes.get(identity['pid'])
    if current and current['start_ticks'] == identity['start_ticks'] and current['state'] != 'Z':
        raise ValueError('Original process is still live: ' + str(identity['pid']))
    return {'pid': identity['pid'], 'start_ticks': identity['start_ticks'],
            'terminal': True, 'observation': 'absent' if not current else 'zombie_or_pid_reused'}


def no_writers(processes):
    for process in processes.values():
        if process['state'] == 'Z':
            continue
        argv = ' '.join(process['argv'])
        original_worker = ('retain_profile_checkpoint.py' in argv and '--execute-retention' in process['argv'])
        if original_worker or process['writers'] or any(p in argv for p in (GB_PARTIAL, GB_DEST)):
            raise ValueError('Existing checkpoint transfer/writer: ' + str(process['pid']))


def validate_identity(value, require_worker=False):
    if not isinstance(value, dict) or value.get('uid') != UID:
        raise ValueError('Expected observed ordinary-user process identity')
    for key in ('pid', 'start_ticks'):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError('Missing exact process ' + key)
    argv = value.get('argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise ValueError('Missing observed process argv')
    checksum = hashlib.sha256(b'\0'.join(a.encode() for a in argv) + b'\0').hexdigest()
    if value.get('cmdline_sha256') != checksum:
        raise ValueError('Observed process argv/hash mismatch')
    if require_worker and (not any(Path(a).name == 'retain_profile_checkpoint.py' for a in argv)
                           or '--execute-retention' not in argv):
        raise ValueError('Handoff does not identify the original retention worker')
    if not require_worker and not any(Path(a).name in ('ssh', 'rsync', 'timeout') for a in argv):
        raise ValueError('Handoff does not identify a copy process')


def load_handoff(path, expected_sha, budget):
    budget.remaining()
    checked(path)
    if Path(path).stat().st_size > 1024**2:
        raise ValueError('Oversized handoff record')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Reviewed handoff SHA256 differs')
    handoff = json.loads(raw)
    expected = {'schema_version': 1, 'authorization': AUTHORIZATION, 'reviewed': True,
                'source_root': DEST, 'destination_root': GB_PARTIAL,
                'checkpoint_id': EXPECTED_ID, 'deadline_utc': '2026-09-16T05:40:00+00:00',
                'original_worker_host': LOCAL_HOST, 'gb_host': GB_NODE,
                'original_worker_terminal': True, 'original_gb_copy_terminal': True}
    if any(handoff.get(k) != v for k, v in expected.items()):
        raise ValueError('Handoff scope, terminal proof, review, or deadline differs')
    reviewed = dt.datetime.fromisoformat(handoff['reviewed_at']).timestamp()
    if not 0 <= time.time() - reviewed <= 600 or not handoff.get('reviewer'):
        raise ValueError('Review must identify reviewer and be at most ten minutes old')
    validate_identity(handoff['original_worker'], require_worker=True)
    for key in ('original_local_copy_processes', 'original_gb_copy_processes'):
        if not isinstance(handoff.get(key), list) or not handoff[key]:
            raise ValueError('Require exact observed original copy identities on both hosts')
        for process in handoff[key]:
            validate_identity(process)
    state_path = Path(ROOT, STATE_NAME)
    checked(state_path)
    state_raw = state_path.read_bytes()
    if hashlib.sha256(state_raw).hexdigest() != handoff.get('original_state_sha256'):
        raise ValueError('Original worker state differs from reviewed handoff')
    state = json.loads(state_raw)
    if state.get('pid') != handoff['original_worker']['pid'] or state.get('phase') not in ('staging_durable_checkpoint_to_gb_raid', 'failed'):
        raise ValueError('Original worker did not reach GB copy, or has already completed')
    return handoff


def guarded_processes(identities, budget):
    receipts = []
    for index in range(2):
        budget.remaining()
        snapshot = proc_snapshot()
        terminal = [check_terminal(identity, snapshot) for identity in identities]
        no_writers(snapshot)
        receipts.append({'observed_at': utc(), 'terminal': terminal, 'no_user_writers': True})
        if index == 0:
            time.sleep(min(1, budget.remaining()))
    return receipts


def verify_source(budget):
    budget.remaining()
    manifest = inventory(DEST)
    if manifest['id'] != EXPECTED_ID or len(manifest['shards']) != 4:
        raise ValueError('Require the reviewed checkpoint identity and exactly four shards')
    record_path = Path(ROOT, 'checkpoint-retention-profile-9.json')
    checked(record_path)
    record = json.loads(record_path.read_text())
    if (record.get('destination_root') != DEST or record.get('destination_checkpoint_id') != EXPECTED_ID
            or record.get('source_checkpoint_id') != EXPECTED_ID or record.get('files') != manifest['files']
            or record.get('uid') != UID or record.get('rsync_exit_code') != 0):
        raise ValueError('Durable retention record has not verified this checkpoint')
    return manifest, file_state(DEST, manifest), hashlib.sha256(record_path.read_bytes()).hexdigest()


def inspect_partial(manifest):
    checked(Path(GB_RAID))
    if Path(GB_DEST).exists() or Path(GB_DEST).is_symlink():
        raise FileExistsError('Completed checkpoint target already exists')
    partial = Path(GB_PARTIAL)
    checked(partial)
    expected = manifest['files']
    directories = {str(p) for name in expected for p in Path(name).parents if str(p) != '.'}
    for path in partial.rglob('*'):
        info = checked(path)
        name = str(path.relative_to(partial))
        if path.is_dir():
            if name not in directories:
                raise ValueError('Unexpected partial directory: ' + name)
        elif name not in expected or info.st_size > expected[name]['bytes']:
            raise ValueError('Unexpected or oversized partial file: ' + name)
    return {'partial_files': sum(1 for p in partial.rglob('*') if p.is_file()),
            'partial_bytes': sum(p.stat().st_size for p in partial.rglob('*') if p.is_file())}


def remote_source(operation, handoff, manifest):
    """Standalone CPU-only receiver guard; no deployment or existing-process signals."""
    functions = (utc, checked, checkpoint_metadata, inventory, proc_snapshot, check_terminal,
                 no_writers, guarded_processes, inspect_partial)
    source = 'import datetime as dt, hashlib, json, os, signal, socket, stat, time\nfrom pathlib import Path\n'
    for key, value in {'UID': UID, 'ITERATION': 9, 'GB_RAID': GB_RAID, 'GB_DEST': GB_DEST,
                       'GB_PARTIAL': GB_PARTIAL, 'DEADLINE': DEADLINE}.items():
        source += f'{key} = {value!r}\n'
    source += inspect.getsource(Budget)  # contextmanager is needed even when only remaining() is used.
    source = 'from contextlib import contextmanager\n' + source
    source += '\n'.join(inspect.getsource(fn) for fn in functions)
    source += f'\nmanifest = {manifest!r}\nhandoff = {handoff!r}\noperation = {operation!r}\n'
    source += '''
if os.getuid() != UID or os.getgid() != 30 or socket.gethostname().split('.')[0] != handoff['gb_host']:
    raise ValueError('Unexpected GB host/user')
budget = Budget()
with budget.enforce():
    receipts = guarded_processes(handoff['original_gb_copy_processes'], budget)
    summary = inspect_partial(manifest)
    lock = Path(GB_RAID, '.checkpoint9-parallel-fallback.lock')
    if operation == 'prepare':
        budget.remaining()
        with lock.open('x') as stream:
            json.dump({'handoff_sha256': handoff['_sha256'], 'deadline': DEADLINE, 'created_at': utc()}, stream)
    elif operation == 'publish':
        checked(lock)
        if json.loads(lock.read_text())['handoff_sha256'] != handoff['_sha256']:
            raise ValueError('Different fallback owns target lock')
        actual = inventory(GB_PARTIAL)
        if actual != manifest:
            raise ValueError('Complete destination inventory differs')
        if budget.remaining() < 10:
            raise TimeoutError('Insufficient shared budget for rename and completion record')
        Path(GB_PARTIAL).rename(GB_DEST)
        summary.update(checkpoint_id=actual['id'], files=actual['files'], verified_at=utc())
    elif operation != 'inspect':
        raise ValueError('Unknown operation')
    print(json.dumps({'observations': receipts, **summary}))
'''
    return source


def remote(operation, handoff, manifest, budget):
    seconds = budget.remaining(120)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', GB_NODE, 'python3 -'],
                            input=remote_source(operation, handoff, manifest), text=True,
                            capture_output=True, timeout=seconds)
    if result.returncode:
        raise RuntimeError('GB ' + operation + ' failed: ' + result.stderr[-4000:])
    budget.remaining()
    return json.loads(result.stdout)


def receiver_command():
    # Compute the remaining absolute budget on the receiver after SSH setup.
    code = ("import os,sys,time; "
            f"assert os.getuid()=={UID} and os.getgid()==30; "
            f"assert '--server' in sys.argv[1:] and '--sender' not in sys.argv[1:] and sys.argv[-1].rstrip('/')=={GB_PARTIAL!r}; "
            f"seconds=int({DEADLINE!r}-time.time()-5); "
            "assert seconds>0, 'absolute receiver deadline expired'; "
            "os.execvp('timeout',['timeout','--foreground','--kill-after=2s',str(seconds),'rsync',*sys.argv[1:]])")
    return shlex.join(['python3', '-c', code])


def transfer_argv(names, budget):
    if not names or len(set(names)) != len(names):
        raise ValueError('Require distinct transfer paths')
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or name.startswith('-'):
            raise ValueError('Unsafe transfer path')
    # Outer and receiver watchdogs govern only these newly spawned rsync children.
    seconds = math.floor(budget.remaining() - 5)
    if seconds < 1:
        raise TimeoutError('No copy budget remains before absolute cutoff')
    return ['timeout', '--foreground', '--kill-after=2s', str(seconds),
            'rsync', '-rt', '--relative', '--partial', '--no-whole-file',
            '--no-owner', '--no-group', '--chmod=Du+rwx,Fu+rw', '--stats',
            '--rsync-path=' + receiver_command(),
            '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
            *[DEST + '/./' + name for name in names], GB_NODE + ':' + GB_PARTIAL + '/']


def transfer(names, log_path, budget):
    argv = transfer_argv(names, budget)
    started = utc()
    with Path(log_path).open('xb') as stream:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=budget.remaining())
    budget.remaining()
    receipt = {'paths': names, 'started_at': started, 'ended_at': utc(),
               'exit_code': result.returncode, 'log': str(log_path)}
    write_json(Path(str(log_path) + '.json'), receipt, budget, exclusive=True)
    if result.returncode:
        raise RuntimeError('Transfer failed; partials/log retained: ' + json.dumps(receipt))
    return receipt


def copy_shards(manifest, audit_dir, budget):
    names = [name for name, _ in manifest['shards']]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError('Exactly four unique immutable shard writers required')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(transfer, [name], Path(audit_dir, f'shard-{i}.log'), budget)
                   for i, name in enumerate(names)]
        # No retries; finish/join only these already bounded children on failure.
        results = [future.result() for future in futures]
    return results


def execute(handoff_path, handoff_sha):
    budget = Budget()
    audit = {'schema_version': 1, 'status': 'starting', 'started_at': utc(),
             'deadline_utc': dt.datetime.fromtimestamp(DEADLINE, dt.timezone.utc).isoformat(),
             'checkpoint_id': EXPECTED_ID, 'handoff_sha256': handoff_sha, 'stages': []}
    audit_dir = Path(ROOT, AUDIT_NAME)
    audit_path = audit_dir / 'state.json'
    with budget.enforce():
        if os.getuid() != UID or os.getgid() != 30 or socket.gethostname().split('.')[0] != LOCAL_HOST:
            raise ValueError('Execute only on dl3 as ordinary UID28644:GID30')
        checked(Path(ROOT))
        handoff = load_handoff(Path(handoff_path), handoff_sha, budget)
        handoff['_sha256'] = handoff_sha
        identities = [handoff['original_worker'], *handoff['original_local_copy_processes']]
        audit['local_observations'] = guarded_processes(identities, budget)
        manifest, before, record_sha = verify_source(budget)
        audit.update(total_bytes=sum(x['bytes'] for x in manifest['files'].values()), durable_record_sha256=record_sha)
        if Path(ROOT, STAGE_NAME).exists() or Path(ROOT, STAGE_NAME).is_symlink():
            raise FileExistsError('Existing completed GB stage record')
        # All review/source/process checks precede the first fallback output write.
        audit['remote_preflight'] = remote('inspect', handoff, manifest, budget)
        budget.remaining()
        audit_dir.mkdir(mode=0o700)
        try:
            write_json(audit_path, audit, budget)
            audit['remote_prepare'] = remote('prepare', handoff, manifest, budget)
            audit['status'] = 'copying'
            write_json(audit_path, audit, budget)
            metadata = [name for name, meta in manifest['files'].items() if 'sha256' in meta]
            audit['stages'].append(transfer(metadata, audit_dir / 'metadata.log', budget))
            audit['stages'].extend(copy_shards(manifest, audit_dir, budget))
            budget.remaining()
            if inventory(DEST) != manifest or file_state(DEST, manifest) != before:
                raise ValueError('Durable source changed during fallback')
            audit['local_final_observations'] = guarded_processes(identities, budget)
            gb = remote('publish', handoff, manifest, budget)
            record = {k: gb[k] for k in ('checkpoint_id', 'files', 'verified_at')}
            record.update(source_root=DEST, destination_root=GB_DEST, uid=UID, rsync_exit_code=0,
                          method='reviewed_four_shard_parallel_fallback', handoff_sha256=handoff_sha,
                          verification='rsync_transfer_plus_sizes_and_metadata_sha256')
            write_json(Path(ROOT, STAGE_NAME), record, budget, exclusive=True)
            audit.update(status='complete', ended_at=utc(), gb=record)
            write_json(audit_path, audit, budget)
        except BaseException as error:
            audit.update(status='failed', failed_at=utc(), error=repr(error), partials_preserved=True)
            try:
                write_json(audit_path, audit, budget)
            except BaseException:
                pass  # Never extend the deadline to write a failure audit.
            print(json.dumps(audit), file=sys.stderr)
            raise
    return audit


def plan():
    return {'mode': 'plan_only', 'source': DEST, 'destination': GB_NODE + ':' + GB_PARTIAL,
            'completed_destination': GB_DEST, 'checkpoint_id': EXPECTED_ID, 'iteration': 9,
            'deadline_utc': '2026-09-16T05:40:00+00:00', 'max_parallel_shards': 4,
            'metadata': 'separate transfer', 'retry': False, 'existing_process_actions': [],
            'requires': ['reviewed handoff JSON + explicit SHA256', 'original worker and GB copy terminal',
                         'two fresh no-writer scans on each host', 'verified durable source',
                         'owned existing partial; no completed target'],
            'completion_record': str(Path(ROOT, STAGE_NAME))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-handoff', action='store_true')
    parser.add_argument('--handoff-record')
    parser.add_argument('--handoff-sha256')
    args = parser.parse_args(argv)
    if not args.execute_reviewed_handoff:
        print(json.dumps(plan(), indent=2))
        return
    if not args.handoff_record or not args.handoff_sha256:
        parser.error('Execution requires both reviewed handoff record and SHA256')
    print(json.dumps(execute(args.handoff_record, args.handoff_sha256), indent=2))


if __name__ == '__main__':
    main()
