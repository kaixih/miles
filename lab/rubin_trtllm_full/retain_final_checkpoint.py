#!/usr/bin/env python3
"""Pull completed checkpoint49 from the allocated node to direct dl3 NFS.

Default prints a plan. --execute requires completed main exits and a stable
model-only final checkpoint. No training, deletion, node allocation, overwrite
or automatic retry is performed. Existing partial destinations are preserved.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rubin_two_node'))
import retain_profile_checkpoint as helpers
from checkpoint_metadata import checkpoint_metadata

UID, GID, ITERATION = 28644, 30, 49
JOBS = {'rubin': '2212643', 'gb300': '2212644'}
BASE = Path('/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full')
MAX_SECONDS = 1800
FINAL_RESERVE_SECONDS = 90
INCOMPLETE_MARKER = '_INCOMPLETE.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value=None):
    return dt.datetime.fromtimestamp(time.time() if value is None else value, dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Explicit lease timezone required')
    return parsed.timestamp()


def validate_config(raw, job_id=None):
    platform = raw.get('platform')
    require(platform in JOBS, 'Unknown platform')
    job = JOBS[platform] if job_id is None else str(job_id)
    require(re.fullmatch(r'[1-9][0-9]*', job), 'Explicit job ID must be a positive decimal allocation ID')
    expected_run = f'20260917-{platform}-j{job}-trtllm'
    require(str(raw.get('job_id')) == job and raw.get('run_id') == expected_run, 'Wrong selected campaign/job identity')
    require(raw.get('run_dir') == str(BASE / expected_run), 'Wrong durable campaign root')
    require(re.fullmatch(r'[A-Za-z0-9.-]+', raw.get('node', '')), 'Unsafe node name')
    p = Path(raw['node_run_dir'])
    require(p.is_absolute() and '..' not in p.parts and str(p) == raw['node_run_dir']
            and str(p).startswith(('/tmp/', '/raid/')) and p.name == 'run'
            and re.search(r'(?:^|[-/])j' + re.escape(job) + r'(?=$|[-/])', str(p)),
            'Unbound node-local checkpoint path')
    require(raw.get('uid', UID) == UID and raw.get('gid', GID) == GID, 'Normal writer identity mismatch')
    require(raw.get('sglang_moe_runner_backend') == 'flashinfer_trtllm'
            and raw.get('save_optimizer') is False, 'Require the model-only TRTLLM main run')
    timestamp(raw['lease_deadline'])
    return dict(raw)


def load_config(path, job_id=None):
    return validate_config(json.loads(Path(path).read_text()), job_id=job_id)


def allocation_guard(c, raw, now):
    fields = dict(re.findall(r'(\w+)=([^\s]+)', raw))
    for name, expected in {'JobId': str(c['job_id']), 'JobState': 'RUNNING',
                           'NodeList': c['node'], 'NumNodes': '1'}.items():
        require(fields.get(name) == expected, 'Allocation mismatch: ' + name)
    require(re.fullmatch(r'[^()]+\(28644\)', fields.get('UserId', '')), 'Wrong allocation owner')
    require(re.fullmatch(r'[^()]+\(30\)', fields.get('GroupId', '')), 'Wrong allocation group')
    end = timestamp(fields['EndTime'] + 'Z')
    require(end == timestamp(c['lease_deadline']), 'Original allocation end changed')
    require(end - now > 150, 'Too little original lease remains')
    return {'checked_at': utc(now), 'lease_deadline': c['lease_deadline'], 'slurm': fields}


def remaining(deadline, now=None, cap=60):
    seconds = int(deadline - (time.time() if now is None else now))
    if seconds <= 0:
        raise TimeoutError('Original retention deadline has expired')
    return min(cap, seconds)


def operation_deadline(c, now):
    deadline = min(now + MAX_SECONDS, timestamp(c['lease_deadline']) - 120)
    require(deadline - now > FINAL_RESERVE_SECONDS + 30, 'Insufficient bounded retention budget')
    return deadline


def read_owned_json(path):
    path = Path(path); helpers.checked(path)
    require(path.is_file() and path.stat().st_size < 8 * 1024**2, 'Oversized or nonregular receipt')
    return json.loads(path.read_text())


def main_completion(c):
    root = Path(c['run_dir'])
    driver = read_owned_json(root / 'train-driver-exit.json')
    train = read_owned_json(root / 'train_exit.json')
    plan = read_owned_json(root / 'cudagraph-main-plan.json')
    require(driver.get('exit_code') == 0 and driver.get('run_id') == c['run_id']
            and train.get('exit_code') == 0, 'Both exact main exits must be zero')
    require(plan.get('run_id') == c['run_id'] and plan.get('recipe', {}).get('save_optimizer') is False
            and plan['recipe'].get('rollouts') == 50 and plan['recipe'].get('optimizer_updates') == 200,
            'Main provenance does not prove the model-only final-save recipe')
    return {'train_exit': train, 'driver_exit': driver,
            'plan_sha256': hashlib.sha256((root / 'cudagraph-main-plan.json').read_bytes()).hexdigest()}


def source_snapshot(c):
    """Self-contained entrypoint appended to canonical inventory helper source."""
    import os, socket
    from pathlib import Path
    assert (os.getuid(), os.getgid()) == (28644, 30)
    assert socket.gethostname().split('.')[0] == c['node'].split('.')[0]
    root = Path(c['node_run_dir']) / 'checkpoints'
    result = inventory(root, iteration=49)
    states = file_state(root, result)
    # Repeat small metadata/inventory reads to reject a save changing mid-probe.
    assert inventory(root, iteration=49) == result and file_state(root, result) == states
    return {'root': str(root), 'inventory': result, 'file_state': states}


def source_program(c):
    imports = 'import hashlib,json,os,stat\nfrom pathlib import Path\nUID=28644\nITERATION=49\n'
    functions = (checkpoint_metadata, helpers.checked, helpers.inventory, helpers.file_state, source_snapshot)
    return imports + '\n\n'.join(inspect.getsource(f) for f in functions) + '\nprint(json.dumps(source_snapshot(' + repr(c) + ')))'


def validate_snapshot(snapshot):
    manifest = snapshot['inventory']
    require(manifest['iteration'] == ITERATION, 'Unexpected retained iteration')
    names = set(manifest['files'])
    require('latest_checkpointed_iteration.txt' in names
            and 'rollout/global_dataset_state_dict_49.pt' in names, 'Final tracker or rollout state missing')
    require(set(snapshot['file_state']) == names, 'Incomplete source stability record')
    for name, item in manifest['files'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts and '\n' not in name and '\r' not in name,
                'Unsafe checkpoint inventory path')
        require(name == 'latest_checkpointed_iteration.txt'
                or name == 'rollout/global_dataset_state_dict_49.pt'
                or p.parts[0] == 'iter_0000049', 'Inventory escapes checkpoint49')
        require(type(item['bytes']) is int and item['bytes'] > 0, 'Invalid checkpoint member size')
        state = snapshot['file_state'][name]
        require(len(state) == 3 and state[1] == item['bytes'], 'File state does not match inventory')
    total = sum(item['bytes'] for item in manifest['files'].values())
    require(0 < total <= 128 * 1024**3 and len(names) <= 1024, 'Checkpoint exceeds the model-only retention bound')
    return total


def stable_source(before, after):
    require(before == after, 'Source checkpoint changed during retention; partial destination preserved')


def verify_destination(root, expected, marker=None):
    # SSH JSON transports tuple-valued shard entries as lists.
    actual = json.loads(json.dumps(helpers.inventory(root, iteration=49)))
    require(actual == expected, 'Destination sizes/metadata SHA differ from source')
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if marker is not None:
        require(read_owned_json(root / INCOMPLETE_MARKER) == marker, 'Destination reservation marker changed')
        actual_files.remove(INCOMPLETE_MARKER)
    require(actual_files == set(expected['files']), 'Unexpected destination files')
    return actual


def reserve_destination(destination, run_id):
    """NFS-compatible exclusive mkdir; only the later PASS receipt publishes it."""
    destination.mkdir(mode=0o700)  # Existing files/directories always fail, even if empty.
    marker = {'status': 'INCOMPLETE', 'run_id': run_id, 'invocation_id': uuid.uuid4().hex,
              'created_at': utc(), 'meaning': 'Directory existence is not completion; require the PASS retention receipt.'}
    with (destination / INCOMPLETE_MARKER).open('x') as stream:
        json.dump(marker, stream, indent=2); stream.write('\n')
    return marker


def finish_destination(destination, marker):
    """Remove only this invocation's marker after data verification; keep all data."""
    path = destination / INCOMPLETE_MARKER
    require(read_owned_json(path) == marker, 'Refuse to remove a changed destination marker')
    path.unlink()


def persist(path, value):
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    os.replace(temp, path)


def rsync_command(c, partial, file_list, deadline, now=None):
    budget = remaining(deadline, now=now, cap=MAX_SECONDS)
    return ['rsync', '-rt', '--no-owner', '--no-group', '--no-perms', '--omit-dir-times',
            '--chmod=Du+rwx,Fu+rw', '--partial', '--stats', '--protect-args',
            '--files-from=' + str(file_list), '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
            '--rsync-path=' + shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', str(budget) + 's', 'rsync']),
            c['node'] + ':' + c['node_run_dir'] + '/checkpoints/', str(partial) + '/']


def execute(c):
    require((os.getuid(), os.getgid()) == (UID, GID), 'Run on dl3 as normal UID28644:GID30')
    root = Path(c['run_dir']); helpers.checked(root)
    final, partial = root / 'final-checkpoint', root / 'final-checkpoint.partial'
    receipt_path = root / 'final-checkpoint-retention.json'
    state_path = root / 'final-checkpoint-retention-state.json'
    lock_path = root / 'final-checkpoint-retention.lock'
    require(not lock_path.is_symlink(), 'Symlinked retention lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any(p.exists() or p.is_symlink() for p in (final, partial, receipt_path, state_path)),
                'Existing published/partial retention evidence: inspect; never overwrite or restart')
        completion = main_completion(c)
        deadline = operation_deadline(c, time.time())
        state = {'phase': 'preflight', 'started_at': utc(), 'run_id': c['run_id'],
                 'deadline_utc': utc(deadline), 'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'dependency_sha256': {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                       for f in (helpers.__file__, inspect.getfile(checkpoint_metadata))}}
        persist(state_path, state)
        def allocation():
            raw = subprocess.check_output(['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o', str(c['job_id'])],
                                          text=True, timeout=remaining(deadline, cap=15))
            return allocation_guard(c, raw, time.time())
        def snapshot():
            allocation()
            budget = remaining(deadline, cap=45)
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', c['node'],
                       shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', str(max(1, budget - 5)) + 's', 'python3', '-B', '-'])]
            result = subprocess.run(command, input=source_program(c), text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=budget)
            require(result.returncode == 0, 'Checkpoint inventory failed: ' + result.stderr[-2500:])
            value = json.loads(result.stdout); validate_snapshot(value)
            return value
        try:
            initial_allocation = allocation()
            filesystem = subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE,SOURCE,TARGET'],
                                                 text=True, timeout=remaining(deadline, cap=15)).strip()
            require(filesystem.split()[0].startswith('nfs'), 'Destination must use dl3 direct NFS, not CIFS')
            first = snapshot()
            time.sleep(min(2, remaining(deadline, cap=2)))
            before = snapshot(); stable_source(first, before)
            total = validate_snapshot(before)
            require(shutil.disk_usage(root).free >= int(total * 1.1) + 10 * 1024**3, 'Insufficient durable NFS capacity')
            persist(root / 'final-checkpoint-source-before.json', before)
            file_list = root / 'final-checkpoint-files.txt'
            require(not file_list.exists(), 'Existing checkpoint transfer list')
            with file_list.open('x') as stream:
                stream.write('\n'.join(sorted(before['inventory']['files'])) + '\n')
            marker = reserve_destination(final, c['run_id'])
            copy_deadline = deadline - FINAL_RESERVE_SECONDS
            command = rsync_command(c, final, file_list, copy_deadline)
            state.update(phase='copying_node_to_direct_nfs', copy_started_at=utc(), checkpoint_bytes=total,
                         allocation=initial_allocation, nfs_filesystem=filesystem, argv=command,
                         incomplete_destination=str(final), reservation=marker)
            persist(state_path, state)
            allocation()
            code = helpers.copy_process(command, copy_deadline, root / 'final-checkpoint-rsync.log')
            state.update(phase='verifying', copied_at=utc(), rsync_exit_code=code); persist(state_path, state)
            after = snapshot(); stable_source(before, after)
            persist(root / 'final-checkpoint-source-after.json', after)
            destination = verify_destination(final, before['inventory'], marker)
            allocation(); remaining(deadline)
            finish_destination(final, marker)
            record = {'status': 'PASS', 'run_id': c['run_id'], 'iteration': 49,
                      'source_node': c['node'], 'source_root': before['root'], 'destination_root': str(final),
                      'source_checkpoint_id': before['inventory']['id'], 'destination_checkpoint_id': destination['id'],
                      'files': destination['files'], 'bytes': total, 'source_stable_before_after': True,
                      'publication': 'exclusive_directory_then_verified_PASS_receipt', 'incomplete_marker_removed': True,
                      'rsync_exit_code': code, 'verification': 'rsync_transfer_plus_sizes_and_metadata_sha256',
                      'tensor_verification': 'rsync transfer checksum and sizes; no extra full tensor SHA256 reread',
                      'started_at': state['started_at'], 'completed_at': utc(), 'deadline_utc': utc(deadline),
                      'main_completion': completion, 'uid': UID, 'gid': GID,
                      'helper_sha256': state['helper_sha256'], 'dependency_sha256': state['dependency_sha256']}
            persist(receipt_path, record)
            state.update(phase='complete', completed_at=utc(), status='PASS', destination_root=str(final))
            persist(state_path, state)
            print(json.dumps(record, indent=2))
            return 0
        except BaseException as exc:
            state.update(phase='failed', failed_at=utc(), status='FAILED', error=repr(exc),
                         partials_preserved=True, source_files_unchanged_by_operator=True)
            persist(state_path, state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--job-id', help='Explicit replacement allocation; config/run/paths must match exactly')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(); c = load_config(args.config, job_id=args.job_id)
    if args.execute:
        return execute(c)
    print(json.dumps({'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'run_id': c['run_id'], 'iteration': 49,
                      'source': c['node'] + ':' + c['node_run_dir'] + '/checkpoints',
                      'destination': c['run_dir'] + '/final-checkpoint', 'maximum_seconds': MAX_SECONDS,
                      'lease_safety_margin_seconds': 120, 'final_verification_reserve_seconds': FINAL_RESERVE_SECONDS,
                      'preserve_partial': True, 'overwrite': False, 'delete_source': False,
                      'completion_signal': 'PASS retention receipt and absent _INCOMPLETE.json marker',
                      'route': 'Node-local -> SSH rsync initiated on dl3 -> direct NFS'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
