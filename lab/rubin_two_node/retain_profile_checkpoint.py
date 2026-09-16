"""Freeze Rubin A2 checkpoint 9, retain it via dl3 NFS, then stage GB RAID.

Run on dl3 as UID28644. Default prints the exact plan. --execute-retention
does not alter either training process or delete any original checkpoint.
The temporary hardlinks protect a completed save from native retention while
copying to RAID. Only links created by this invocation are removed afterward.
No existing destination is reused or overwritten. Tensor verification is the
rsync transfer checksum plus sizes, not an extra full tensor SHA256 read.
"""
try:
    from lab.rubin_two_node.checkpoint_metadata import checkpoint_metadata
except ModuleNotFoundError:
    from checkpoint_metadata import checkpoint_metadata

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import time

UID = 28644
ITERATION = 9
RUN_ID = '20260916-j2198331-qwen3-a2-mb4096'
NODE = 'vr-nvl72-ts2-l11-038-c15'
GB_NODE = 'gb300-nvl-012-compute04'
CONTAINER = 'miles-rubin-qwen3-j2198331-a2-0'
IMAGE = 'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c'
RAY_ID = 'raysubmit_qCmmzUzbwHrBKAiU'
ENTRYPOINT_SHA256 = 'a9bb4cf3c8dcba293752503120d58441b299e86f0db0ce4ab185e98f0322e151'
SOURCE = '/tmp/miles-rubin-j2198331/run-a2-mb4096/checkpoints'
RAID = '/raid/dldata/miles-kaixih-j2198331-profile'
GB_RAID = '/raid/tmp/miles-kaixih-j2198810-profile'
ROOT = '/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs'
FROZEN = RAID + '/profile-checkpoint-9'
DEST = ROOT + '/profile-checkpoint-9'
GB_DEST = GB_RAID + '/profile-checkpoint-9'
DEADLINE = dt.datetime(2026, 9, 16, 5, 40, tzinfo=dt.timezone.utc).timestamp()
WAIT_DEADLINE = dt.datetime(2026, 9, 16, 3, 30, tzinfo=dt.timezone.utc).timestamp()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def persist(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def checked(path):
    path = Path(path)
    info = path.lstat()
    if path.resolve() != path or info.st_uid != UID or stat.S_ISLNK(info.st_mode):
        raise ValueError('Foreign-owned or symlinked checkpoint path: ' + str(path))
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError('Nonregular checkpoint path: ' + str(path))
    return info


def inventory(root, iteration=ITERATION):
    root = Path(root)
    directory = root / f'iter_{iteration:07d}'
    checked(root)
    checked(directory)
    for path in directory.rglob('*'):
        checked(path)
    tracker = root / 'latest_checkpointed_iteration.txt'
    indexes = [p for p in (directory / '.metadata', directory / 'metadata.json') if p.is_file()]
    if not indexes:
        raise ValueError('Missing distributed checkpoint metadata')
    metadata = checkpoint_metadata(root, iteration)
    files = {}
    for path in [tracker, *metadata]:
        info = checked(path)
        if not path.is_file() or not 0 < info.st_size <= 32 * 1024**2:
            raise ValueError('Missing/empty/oversized checkpoint metadata: ' + str(path))
        files[str(path.relative_to(root))] = {'bytes': info.st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if tracker.read_text().strip() != str(iteration):
        raise ValueError('Tracker is not the selected checkpoint')
    shards = sorted((str(p.relative_to(root)), checked(p).st_size) for p in directory.rglob('*.distcp'))
    if not shards or any(size <= 0 for _, size in shards):
        raise ValueError('Missing or empty tensor shards')
    for name, size in shards:
        if not (root / name).is_file():
            raise ValueError('Shard is not a regular file')
        files[name] = {'bytes': size}
    identity = {'iteration': iteration,
                'metadata': [{'path': str(p.relative_to(root)), 'sha256': files[str(p.relative_to(root))]['sha256']} for p in metadata],
                'shards': shards}
    return {'id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), 'files': files, **identity}


def file_state(root, manifest):
    return {name: (checked(Path(root, name)).st_ino, checked(Path(root, name)).st_size,
                   checked(Path(root, name)).st_mtime_ns) for name in manifest['files']}


def copy_process(argv, deadline, log_path):
    """Timeout only the new copy process group; never signal a training process."""
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Retention deadline passed before starting copy')
    with open(log_path, 'ab', buffering=0) as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError('Bounded checkpoint copy timed out; partial destination retained')
        if code:
            raise RuntimeError('Checkpoint copy failed, exit ' + str(code))
    return code


def pin_files(source, pin, manifest):
    """Create new links, with an independent tracker; reject an existing pin."""
    source, pin = Path(source), Path(pin)
    pin.mkdir(mode=0o700)
    created_files, created_dirs = [], [pin]
    try:
        for name in manifest['files']:
            target = pin / name
            missing = []
            parent = target.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for parent in reversed(missing):
                parent.mkdir(mode=0o700)
                created_dirs.append(parent)
            if name == 'latest_checkpointed_iteration.txt':
                target.write_bytes((source / name).read_bytes())
            else:
                os.link(source / name, target)
            created_files.append(target)
        if inventory(pin)['id'] != manifest['id']:
            raise ValueError('Pinned checkpoint differs from the source')
        return created_files, created_dirs
    except BaseException:
        unpin(created_files, created_dirs)
        raise


def unpin(files, dirs):
    for path in reversed(files):
        path.unlink()
    for path in reversed(dirs):
        path.rmdir()


def identity_guard():
    import urllib.request
    if os.getuid() != UID or os.getgid() != 30:
        raise ValueError('Expected ordinary UID28644:GID30')
    container = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True, timeout=20))[0]
    if not container['State']['Running'] or container['Config']['Image'] != IMAGE or container['Config']['User'] != '28644:30':
        raise ValueError('Unexpected Rubin container identity/state')
    mounts = {m['Destination']: m['Source'] for m in container['Mounts']}
    if mounts.get('/run-output') != str(Path(SOURCE).parent):
        raise ValueError('Unexpected actual training output mount')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    jobs = json.load(opener.open('http://10.102.74.82:28265/api/jobs/', timeout=15))
    found = [j for j in jobs if j.get('submission_id') == RAY_ID]
    if len(found) != 1 or found[0]['status'] not in ('RUNNING', 'SUCCEEDED'):
        raise ValueError('Original Rubin Ray job absent, failed, or stopped')
    job = found[0]
    if job.get('runtime_env', {}).get('env_vars', {}).get('RUBIN_RUN_ID') != RUN_ID:
        raise ValueError('Unexpected Ray runtime identity')
    if hashlib.sha256(job['entrypoint'].encode()).hexdigest() != ENTRYPOINT_SHA256:
        raise ValueError('Original Ray entrypoint changed')
    args = shlex.split(job['entrypoint'])
    if args.count('--save') != 1 or args[args.index('--save') + 1] != '/run-output/checkpoints':
        raise ValueError('Unexpected checkpoint save path')
    return {'submission_id': RAY_ID, 'entrypoint_sha256': hashlib.sha256(job['entrypoint'].encode()).hexdigest(),
            'container_id': container['Id'], 'image': IMAGE, 'checked_at': utc()}


def node_capture():
    checked(Path(RAID))
    frozen, partial = Path(FROZEN), Path(FROZEN + '.partial')
    pin = Path(SOURCE).parent / '.profile-checkpoint-9-pin'
    if any(p.exists() for p in (frozen, partial, pin)):
        raise ValueError('Capture destination/pin already exists; inspect rather than overwrite')
    guard = identity_guard()
    process_stat = Path('/proc/self/stat').read_text().rpartition(') ')[2].split()
    persist(Path(RAID, 'manifests', f'checkpoint9-observer-{os.getpid()}.json'),
            {'pid': os.getpid(), 'parent_pid': os.getppid(), 'start_ticks': int(process_stat[19]),
             'uid': os.getuid(), 'run_id': RUN_ID, 'ssh_connection': os.environ.get('SSH_CONNECTION'),
             'started_at': utc(), 'identity': guard, 'source_root': SOURCE, 'destination_root': FROZEN})
    last_identity_check = time.monotonic()
    while time.time() < WAIT_DEADLINE:
        if time.monotonic() - last_identity_check >= 60:
            guard = identity_guard()
            last_identity_check = time.monotonic()
        tracker = Path(SOURCE, 'latest_checkpointed_iteration.txt')
        if tracker.exists() and tracker.read_text().strip().isdigit() and int(tracker.read_text().strip()) > ITERATION:
            raise ValueError('Selected save has been superseded before capture')
        try:
            first = inventory(SOURCE)
            before = file_state(SOURCE, first)
        except (FileNotFoundError, ValueError):
            time.sleep(10)
            continue
        time.sleep(10)
        try:
            second = inventory(SOURCE)
            if first != second or before != file_state(SOURCE, second):
                continue
        except (FileNotFoundError, ValueError):
            continue
        break
    else:
        raise TimeoutError('Checkpoint9 did not become stable by the bounded wait deadline')
    guard = identity_guard()
    total = sum(meta['bytes'] for meta in second['files'].values())
    if shutil.disk_usage(RAID).free < total * 1.1 + 10 * 1024**3:
        raise ValueError('Insufficient RAID capacity for the actual checkpoint')
    created_files, created_dirs = pin_files(SOURCE, pin, second)
    started = utc()
    try:
        if inventory(SOURCE) != second or file_state(SOURCE, second) != before:
            raise ValueError('Source changed while pinning checkpoint9')
        partial.mkdir(mode=0o700)
        code = copy_process(['rsync', '-rt', '--no-owner', '--no-group', '--chmod=Du+rwx,Fu+rw',
                             '--stats', str(pin) + '/', str(partial) + '/'],
                            min(DEADLINE, time.time() + 1800), Path(RAID, 'checkpoint9-local-rsync.log'))
        destination = inventory(partial)
        if destination != second:
            raise ValueError('Frozen RAID checkpoint inventory differs after transfer')
        partial.rename(frozen)
        record = {'exit_code': 0, 'rsync_exit_code': code, 'uid': os.getuid(), 'run_id': RUN_ID,
                  'source_node': NODE, 'source_root': SOURCE, 'destination_root': FROZEN,
                  'iteration': ITERATION, 'source_checkpoint_id': second['id'],
                  'destination_checkpoint_id': destination['id'], 'files': second['files'],
                  'verification': 'rsync_transfer_plus_sizes_and_metadata_sha256',
                  'started_at': started, 'completed_at': utc(), 'bytes': total, 'identity': guard}
        persist(Path(RAID, 'manifests/checkpoint9-capture.json'), record)
        return record
    finally:
        # Only the fresh hardlinks and directories created above; no source removal.
        unpin(created_files, created_dirs)


def remote_code(node, suffix, timeout):
    helper_import = ('try:\n    from lab.rubin_two_node.checkpoint_metadata import checkpoint_metadata\n'
                     'except ModuleNotFoundError:\n    from checkpoint_metadata import checkpoint_metadata\n\n')
    source = Path(__file__).read_text().replace(helper_import, inspect.getsource(checkpoint_metadata) + '\n', 1)
    # Match the real final guard, not this function's own string literal.
    code = source.rsplit("\nif __name__ == '__main__':", 1)[0] + '\n' + suffix
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, 'python3 -'], input=code,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Remote retention failed: ' + result.stderr[-5000:])
    return json.loads(result.stdout)


def execute():
    if os.getuid() != UID or os.getgid() != 30:
        raise ValueError('Run on dl3 as UID28644:GID30')
    root, destination = Path(ROOT), Path(DEST)
    checked(root)
    record_path = root / 'checkpoint-retention-profile-9.json'
    state_path = root / 'checkpoint-retention-profile-9-state.json'
    partial = Path(DEST + '.partial')
    if any(p.exists() for p in (destination, partial, record_path, state_path)):
        raise ValueError('Existing retention artifacts: inspect; this worker never overwrites/restarts')
    state = {'started_at': utc(), 'phase': 'waiting_for_completed_checkpoint9', 'run_id': RUN_ID,
             'deadline_utc': dt.datetime.fromtimestamp(DEADLINE, dt.timezone.utc).isoformat(), 'pid': os.getpid()}
    persist(state_path, state)
    try:
        capture = remote_code(NODE, 'print(json.dumps(node_capture()))', max(1, WAIT_DEADLINE - time.time() + 1900))
        persist(root / 'checkpoint9-local-capture.json', capture)
        total = capture['bytes']
        if shutil.disk_usage(root).free < total * 1.1 + 10 * 1024**3:
            raise ValueError('Insufficient durable NFS capacity')
        partial.mkdir(mode=0o700)
        state.update(phase='copying_frozen_raid_to_durable_nfs', copy_started_at=utc(), checkpoint_bytes=total)
        persist(state_path, state)
        code = copy_process(['rsync', '-rt', '--no-owner', '--no-group', '--chmod=Du+rwx,Fu+rw',
                             '--stats', '-e', 'ssh -o BatchMode=yes', NODE + ':' + FROZEN + '/', str(partial) + '/'],
                            DEADLINE - 1200, root / 'checkpoint9-nfs-rsync.log')
        verified = inventory(partial)
        if verified['id'] != capture['source_checkpoint_id'] or verified['files'] != capture['files']:
            raise ValueError('Durable checkpoint differs from frozen source')
        partial.rename(destination)
        record = dict(capture, destination_root=DEST, intermediate_root=FROZEN, rsync_exit_code=code,
                      retained_at=utc(), copy_started_at=state['copy_started_at'],
                      destination_checkpoint_id=verified['id'])
        persist(record_path, record)
        state.update(phase='staging_durable_checkpoint_to_gb_raid', durable_verified_at=utc())
        persist(state_path, state)
        remote_code(GB_NODE, "checked(Path(GB_RAID)); p=Path(GB_DEST+'.partial'); "
                    "assert not Path(GB_DEST).exists() and not p.exists(); "
                    f"assert shutil.disk_usage(GB_RAID).free > {int(total * 1.1 + 10 * 1024**3)}; "
                    "p.mkdir(mode=0o700); print(json.dumps({'prepared':str(p),'uid':os.getuid()}))", 30)
        copy_process(['rsync', '-rt', '--no-owner', '--no-group', '--chmod=Du+rwx,Fu+rw', '--stats',
                      '-e', 'ssh -o BatchMode=yes', str(destination) + '/', GB_NODE + ':' + GB_DEST + '.partial/'],
                     DEADLINE, root / 'checkpoint9-gb-raid-rsync.log')
        gb = remote_code(GB_NODE, 'p=Path(GB_DEST+".partial"); r=inventory(p); '
                         f'assert r["id"] == {verified["id"]!r}; '
                         'assert not Path(GB_DEST).exists(); p.rename(GB_DEST); '
                         'print(json.dumps({"checkpoint_id":r["id"],"files":r["files"],"verified_at":utc()}))', 120)
        persist(root / 'checkpoint9-gb-stage.json', dict(gb, source_root=DEST, destination_root=GB_DEST,
                                                       uid=UID, rsync_exit_code=0))
        state.update(phase='complete', completed_at=utc(), checkpoint_id=verified['id'], exit_code=0)
        persist(state_path, state)
    except BaseException as error:
        state.update(phase='failed', failed_at=utc(), error=repr(error), exit_code=1)
        persist(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-retention', action='store_true')
    args = parser.parse_args()
    if args.execute_retention:
        execute()
    else:
        print(json.dumps({'source': NODE + ':' + SOURCE, 'iteration': ITERATION, 'frozen': FROZEN,
                          'durable': DEST, 'gb_stage': GB_NODE + ':' + GB_DEST,
                          'wait_deadline': WAIT_DEADLINE, 'deadline': DEADLINE,
                          'training_process_actions': [], 'default': 'plan_only'}, indent=2))


if __name__ == '__main__':
    main()
