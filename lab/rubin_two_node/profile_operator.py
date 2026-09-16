#!/usr/bin/env python3
"""Experiment-specific profiling operator; PLAN ONLY unless --execute is explicit.

Copy this file to dl3 and run there as UID28644:GID30. It preserves both images
and sets the paired replay token budgets to 4096. Actions: prepare -> inspect printed replay plan ->
submit --expected-checkpoint-id ID -> retain -> stop. Each action plans by default.
Prepare requires completed learning evidence and explicit permission to stop the
exact completed main container. Never use this to interrupt a main run.

A replay is 2 rollouts/8 updates, not 2 optimizer steps. Both platforms read the
same frozen Rubin iteration-9 checkpoint from verified local RAID storage, only
after their main runs naturally finish all 50 rollouts, or explicitly verified
original-watchdog saved STOPPED outcomes. --initial-policy instead profiles shared
HF/release inputs with a fresh optimizer; no checkpoint9 copy is needed. Cross-version
checkpoint resume remains an execution check. SGLang profiling is separately triggered on an actual replay
engine URL using the bounded payload printed by profile_qwen3_replay.py.
"""
try:
    from lab.rubin_two_node.checkpoint_metadata import checkpoint_metadata
except ModuleNotFoundError:
    from checkpoint_metadata import checkpoint_metadata

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import time

BASE = '/home/scratch.kaixih_ent'
REPO = BASE + '/repo/miles-rubin-cu134'
RUBIN_ROOT = BASE + '/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs'
PROFILE_ITERATION = 9
MAIN_FINAL_ITERATION = 49
CHECKPOINT = RUBIN_ROOT + '/profile-checkpoint-9'
CHECKPOINT_RECORD = 'checkpoint-retention-profile-9.json'
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED'}
RUNS = {
    'rubin': dict(node='vr-nvl72-ts2-l11-038-c15', ip='10.102.74.82', nic='mp0',
        job='2198331', main='miles-rubin-qwen3-j2198331-a2-0', root=RUBIN_ROOT,
        runtime_output='/tmp/miles-rubin-j2198331/run-a2-mb4096',
        raid_root='/raid/dldata/miles-kaixih-j2198331-profile',
        main_id='raysubmit_qCmmzUzbwHrBKAiU', main_run='20260916-j2198331-qwen3-a2-mb4096',
        lease='2026-09-16T06:09:18+00:00', megatron='/opt/Megatron-LM',
        image='gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c'),
    'gb300': dict(node='gb300-nvl-012-compute04', ip='10.85.212.9', nic='enP5p9s0',
        job='2198810', main='miles-gb300-qwen3-j2198810-0',
        root=BASE + '/repro/miles-gb300-qwen3-gsm8k/20260915-j2198810-nfs',
        runtime_output='/tmp/miles-gb300-j2198810/run',
        raid_root='/raid/tmp/miles-kaixih-j2198810-profile',
        main_id='raysubmit_W7k1mh2Y9Uh9wmmk', main_run='20260915-j2198810-qwen3-a1',
        lease='2026-09-16T07:09:02+00:00', megatron='/root/Megatron-LM',
        image='radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986'),
}
# Captured from the original launch records before either watchdog's deadline.
# These are immutable provenance pins, not configurable deadline extensions.
SAVED_STOP_PROVENANCE = {
    'rubin': dict(soft='2026-09-16T05:20:00+00:00', hard='2026-09-16T05:50:00+00:00',
        pid=4565, armed_at='2026-09-16T01:39:43.839235+00:00',
        watchdog_launch_sha256='6b2cda01762f412fb8f37c6f02e78e65abc3e781f929eb183c1d099d360b93aa',
        train_launch_sha256='ccf2d67651ad718547dd9846e8773dd49362368a0cb674fa7138e298d6dffc3b',
        entrypoint_sha256='a9bb4cf3c8dcba293752503120d58441b299e86f0db0ce4ab185e98f0322e151',
        driver_file='run_stage.py', driver_sha256='fa433d07599897eec0d93c9b74367b05b9d1a5c6ce11e339ecda4d441a59d741'),
    'gb300': dict(soft='2026-09-16T06:20:00+00:00', hard='2026-09-16T06:50:00+00:00',
        pid=4961, armed_at='2026-09-15T23:35:26.871770+00:00',
        watchdog_launch_sha256='d675073750b2bfec729ce63ff4a67c6335bbd20a213bc17f07a7d391d96f070e',
        train_launch_sha256='c742eee6f80732f3d7329fa4ffc2e4c7c2f76f60c9f30bbb6440a2e5dff7ef5c',
        entrypoint_sha256='da50906e2c1cd34d0770f5f0d989fd13e6b3b2f0c24bc70390a1b392a08184e2',
        driver_file='gb300_run_stage.py', driver_sha256='fe85d9be68708639bc9861cc7c90da0e11491e38a805461b12b180c14d050eaa'),
}
WATCHDOG_SOURCE_SHA256 = '5a2436521b2d19139f836a8bcadb3193907d54c680537023d3f71cbf91ed61a5'


INPUT_HASHES = {
    'train.jsonl': 'f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8',
    'test-fixed-256.jsonl': '93ed3ccda6ecd09ce0665d0423bf8720b7db97823b0ce2a1c4d20483f410ce99',
}
PORTS = [27379, 27380, 27381, 27382, 27383, 27384, 27385, 29265]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(argv, timeout=60):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def remote(c, argv, timeout=60):
    return run(['ssh', '-o', 'BatchMode=yes', c['node'], shlex.join(argv)], timeout)


def node_python(c, code, timeout=60):
    return json.loads(remote(c, ['python3', '-c', code], timeout))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def config(args):
    c = dict(RUNS[args.platform])
    c.update(platform=args.platform, run_id=args.run_id, initial_policy=getattr(args, 'initial_policy', False))
    c['name'] = 'miles-profile-' + args.run_id
    c['local'] = c['raid_root'] + '/' + c['name']
    c['models'] = c['raid_root'] + '/models'
    c['profile_checkpoint'] = c['raid_root'] + '/profile-checkpoint-9'
    c['model_manifest'] = c['raid_root'] + '/manifests/input-staging.json'
    c['durable'] = c['root'] + '/profiles/' + args.run_id
    c['input'] = c['runtime_output'] + '/inputs'
    c['source_plan'] = c['root'] + '/original-ray-job.json'
    c['submission_id'] = 'qwen3-profile-' + args.run_id
    c['dashboard'] = 'http://' + c['ip'] + ':29265'
    return c


def host_input(c, path):
    return '/mnt/cifs' + path if c['platform'] == 'gb300' and path.startswith(BASE + '/') else path


def budget(c, requested, margin):
    end = dt.datetime.fromisoformat(c['lease']).timestamp()
    remaining = int(end - time.time() - margin)
    if remaining < 900:
        raise RuntimeError('Less than 15 minutes before the reserved retention margin; do not launch')
    return min(requested, remaining)


def docker_command(c):
    argv = ['docker', 'run', '--detach', '--name', c['name'], '--label', 'miles.profile_run=' + c['run_id'],
            '--gpus', 'all', '--network', 'host', '--ipc', 'host', '--privileged',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864', '--ulimit', 'nofile=65535:65535',
            '--shm-size', '16g', '--user', '28644:30', '--workdir', '/opt/miles']
    mounts = [(host_input(c, REPO), '/opt/miles', True),
              (c['models'], BASE + '/models', True),
              *(([(c['model_manifest'], '/profile-input-manifest.json', True)]) if c.get('initial_policy')
                else [(c['profile_checkpoint'], '/profile-checkpoint', True)]),
              (host_input(c, c['source_plan']), '/profile-source/original-ray-job.json', True),
              (c['local'] + '/run', '/run-output', False),
              (c['input'], '/run-output/inputs', True),
              (c['local'] + '/cache', '/cache', False),
              ('/dev/nvidia-caps', '/dev/nvidia-caps', False),
              ('/dev/nvidia-caps-imex-channels', '/dev/nvidia-caps-imex-channels', False)]
    for source, dest, readonly in mounts:
        argv += ['--mount', f'type=bind,src={source},dst={dest}' + (',readonly' if readonly else '')]
    env = {'HOME': '/cache/home', 'PYTHONPATH': '/opt/miles:' + c['megatron'], 'PYTHONUNBUFFERED': '1',
           'MILES_SCRIPT_EXTERNAL_RAY': '1', 'RAY_ADDRESS': c['ip'] + ':27379',
           'RAY_API_SERVER_ADDRESS': c['dashboard'], 'RAY_USAGE_STATS_ENABLED': '0',
           'RAY_DEDUP_LOGS': '0', 'MASTER_ADDR': c['ip'], 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'NCCL_CUMEM_ENABLE': '1', 'NCCL_NVLS_ENABLE': '0',
           'NCCL_SOCKET_IFNAME': c['nic'], 'GLOO_SOCKET_IFNAME': c['nic'],
           'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'MAX_JOBS': '8',
           'no_proxy': 'localhost,127.0.0.1,' + c['ip'], 'NO_PROXY': 'localhost,127.0.0.1,' + c['ip']}
    for key, suffix in {'TMPDIR': 'tmp', 'RAY_TMPDIR': 'ray', 'TRITON_CACHE_DIR': 'triton',
                        'FLASHINFER_WORKSPACE_BASE': 'flashinfer', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
                        'CUDA_CACHE_PATH': 'cuda', 'HF_HOME': 'huggingface',
                        'PYTHONPYCACHEPREFIX': 'pycache', 'XDG_CACHE_HOME': 'xdg'}.items():
        env[key] = '/cache/' + suffix
    for key, value in env.items():
        argv += ['--env', key + '=' + value]
    return argv + [c['image'], 'sleep', 'infinity']


def ray_command(c):
    return ['docker', 'exec', c['name'], 'ray', 'start', '--head', '--node-ip-address', c['ip'],
            '--num-gpus', '4', '--num-cpus', '32', '--disable-usage-stats', '--port', '27379',
            '--node-manager-port', '27380', '--object-manager-port', '27381',
            '--runtime-env-agent-port', '27382', '--dashboard-agent-grpc-port', '27383',
            '--dashboard-agent-listen-port', '27384', '--metrics-export-port', '27385',
            '--min-worker-port', '27400', '--max-worker-port', '27999',
            '--dashboard-host', c['ip'], '--dashboard-port', '29265', '--temp-dir', '/cache/ray']


def replay_command(c, seconds, execute=False, checkpoint_id='', initial_model_id=''):
    cmd = ['docker', 'exec', c['name'], 'python3', '/opt/miles/lab/rubin_two_node/profile_qwen3_replay.py',
           '--source-plan', '/profile-source/original-ray-job.json',
           '--output-dir', '/run-output/replay', '--run-id', c['run_id'],
           '--ray-address', c['dashboard'], '--megatron-path', c['megatron'],
           '--max-runtime-seconds', str(seconds), '--max-trace-gib', '10',
           '--profile-max-tokens-per-gpu', '4096']
    if c.get('initial_policy'):
        cmd += ['--initial-policy', '--initial-model-manifest', '/profile-input-manifest.json']
        if execute:
            cmd += ['--expected-initial-model-id', initial_model_id]
    else:
        cmd += ['--checkpoint-root', '/profile-checkpoint', '--checkpoint-iteration', str(PROFILE_ITERATION)]
        if execute:
            cmd += ['--expected-checkpoint-id', checkpoint_id]
    return cmd + (['--execute-run'] if execute else ['--print-only'])


class OperationDeadline:
    """One nonrenewable wall/monotonic budget, always ending before the lease."""
    def __init__(self, c, requested_seconds, action, *, lease_reserve_seconds=120):
        self.started = time.time()
        self.started_monotonic = time.monotonic()
        self.deadline = min(self.started + requested_seconds,
                            dt.datetime.fromisoformat(c['lease']).timestamp() - lease_reserve_seconds)
        self.lease_reserve_seconds = lease_reserve_seconds
        self.monotonic_deadline = self.started_monotonic + self.deadline - self.started
        self.action = action
        self.stages = []
        self.remaining()

    def remaining(self, maximum=None):
        seconds = min(self.deadline - time.time(), self.monotonic_deadline - time.monotonic())
        if seconds <= 0:
            raise TimeoutError(self.action + ' overall deadline expired')
        return min(seconds, maximum) if maximum is not None else seconds

    def describe(self):
        return {'started_at': dt.datetime.fromtimestamp(self.started, dt.timezone.utc).isoformat(),
                'deadline_at': dt.datetime.fromtimestamp(self.deadline, dt.timezone.utc).isoformat(),
                'deadline_timestamp': self.deadline, 'lease_shutdown_reserve_seconds': self.lease_reserve_seconds,
                'elapsed_seconds': time.monotonic() - self.started_monotonic,
                'stages': [dict(stage) for stage in self.stages]}

    @contextlib.contextmanager
    def enforce(self):
        # The CLI runs on the main thread of a POSIX host. This also bounds local
        # NFS reads/writes; per-subprocess timeouts alone would not bound hashing.
        def expired(_signum, _frame):
            raise TimeoutError(self.action + ' overall deadline expired')
        if signal.getitimer(signal.ITIMER_REAL)[0]:
            raise RuntimeError('Refuse to replace an existing process alarm')
        seconds = self.remaining()
        previous = signal.signal(signal.SIGALRM, expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            yield
            self.remaining()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def stage(self, name, function):
        self.remaining()
        start = time.monotonic()
        record = {'name': name, 'started_at': utc(), 'status': 'running'}
        self.stages.append(record)
        try:
            result = function()
            self.remaining()
            record['status'] = 'complete'
            return result
        except BaseException as exc:
            record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            record.update(elapsed_seconds=time.monotonic() - start, ended_at=utc())


def bounded(deadline, maximum):
    return deadline.remaining(maximum) if deadline is not None else maximum


def _remote_deadline_preamble(deadline):
    # Embedded in read/capture workers so losing SSH cannot renew remote work.
    return """import signal,time
_deadline = DEADLINE
_seconds = _deadline - time.time()
if _seconds <= 0: raise TimeoutError('Remote operation deadline expired')
def _expired(signum, frame): raise TimeoutError('Remote operation deadline expired')
signal.signal(signal.SIGALRM, _expired)
signal.setitimer(signal.ITIMER_REAL, _seconds)
def _check_deadline():
 if time.time() >= _deadline: raise TimeoutError('Remote operation deadline expired')
""".replace('DEADLINE', repr(deadline.deadline))


def _write_before_deadline(path, obj, deadline):
    deadline.remaining()
    path = Path(path)
    temporary = path.with_name(path.name + '.partial-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    deadline.remaining()
    os.replace(temporary, path)
    deadline.remaining()


def jobs(c, dashboard, deadline=None):
    timeout = bounded(deadline, 15)
    return node_python(c, "import json,urllib.request; print(json.dumps(json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(" +
                       repr(dashboard + '/api/jobs/') + ",timeout=" + repr(timeout) + "))))", bounded(deadline, 25))


def allocation(c):
    output = run(['scontrol', 'show', 'job', '-o', c['job']])
    fields = dict(x.split('=', 1) for x in output.split() if '=' in x)
    if fields.get('JobState') != 'RUNNING':
        raise RuntimeError('Allocation is not RUNNING')
    nodes = run(['scontrol', 'show', 'hostnames', fields['NodeList']]).split()
    if c['node'] not in nodes:
        raise RuntimeError('Expected node no longer belongs to allocation')
    # Scheduler timestamps can be timezone-dependent. Never extend the recorded lease.
    if time.time() >= dt.datetime.fromisoformat(c['lease']).timestamp():
        raise RuntimeError('Recorded allocation lease expired')


def _checkpoint_inventory(root, iteration):
    """Read metadata and shard sizes only; same fingerprint ordering as replay."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(root)
    directory = root / f'iter_{iteration:07d}'

    def owned(path):
        if path.resolve() != path or path.stat().st_uid != 28644:
            raise RuntimeError('Checkpoint path is symlinked or foreign-owned: ' + str(path))

    owned(root)
    owned(directory)
    for path in directory.rglob('*'):
        owned(path)
    tracker = root / 'latest_checkpointed_iteration.txt'
    indexes = [p for p in (directory / '.metadata', directory / 'metadata.json') if p.is_file()]
    metadata = checkpoint_metadata(root, iteration)
    if not indexes:
        raise RuntimeError('Checkpoint has no distributed metadata index')
    files = {}
    for path in [tracker, *metadata]:
        owned(path)
        size = path.stat().st_size
        if not path.is_file() or not 0 < size <= 32 * 1024**2:
            raise RuntimeError('Checkpoint metadata is missing, empty, or oversized: ' + str(path))
        files[str(path.relative_to(root))] = {'bytes': size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if tracker.read_text().strip() != str(iteration):
        raise RuntimeError('Checkpoint tracker does not equal requested iteration')
    shards = sorted((str(p.relative_to(root)), p.stat().st_size) for p in directory.rglob('*.distcp'))
    if not shards or any(size <= 0 for _, size in shards):
        raise RuntimeError('Checkpoint shards are missing or empty')
    for name, size in shards:
        if not (root / name).is_file():
            raise RuntimeError('Checkpoint shard is not a regular file: ' + name)
        files[name] = {'bytes': size}
    identity = {'iteration': iteration,
                'metadata': [{'path': str(p.relative_to(root)), 'sha256': files[str(p.relative_to(root))]['sha256']}
                             for p in metadata], 'shards': shards}
    return {'id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            'files': files, **identity}


def check_checkpoint_retention():
    """Require the separately verified frozen iteration-9 copy; never copy it."""
    checkpoint = Path(CHECKPOINT)
    record_path = Path(RUBIN_ROOT, CHECKPOINT_RECORD)
    record = json.loads(record_path.read_text())
    expected = {'exit_code': 0, 'uid': 28644, 'run_id': RUNS['rubin']['main_run'],
                'source_node': RUNS['rubin']['node'],
                'source_root': RUNS['rubin']['runtime_output'] + '/checkpoints',
                'destination_root': CHECKPOINT, 'iteration': PROFILE_ITERATION,
                'rsync_exit_code': 0,
                'verification': 'rsync_transfer_plus_sizes_and_metadata_sha256'}
    if any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Shared Rubin checkpoint lacks a matching verified retention record')
    if record_path.stat().st_uid != 28644 or checkpoint.stat().st_uid != 28644:
        raise RuntimeError('Checkpoint retention writer UID mismatch')
    if checkpoint.resolve() != checkpoint or record_path.resolve() != record_path:
        raise RuntimeError('Checkpoint retention paths must not contain symlinks')
    directory = f'iter_{PROFILE_ITERATION:07d}'
    cursor = f'rollout/global_dataset_state_dict_{PROFILE_ITERATION}.pt'
    indexes = [name for name in (directory + '/.metadata', directory + '/metadata.json')
               if (checkpoint / name).is_file()]
    critical = {'latest_checkpointed_iteration.txt', *(str(p.relative_to(checkpoint)) for p in checkpoint_metadata(checkpoint, PROFILE_ITERATION))}
    files = record.get('files', {})
    if not indexes or not critical.issubset(files) or not any(n.startswith(directory + '/') and n.endswith('.distcp') for n in files):
        raise RuntimeError('Checkpoint retention manifest is incomplete')
    for name, meta in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
            raise RuntimeError('Invalid checkpoint manifest path')
        path = checkpoint / relative
        if path.resolve() != path or not path.is_file() or path.stat().st_uid != 28644:
            raise RuntimeError('Checkpoint file missing, symlinked, or foreign-owned: ' + name)
        if (type(meta.get('bytes')) is not int or meta['bytes'] <= 0
                or path.stat().st_size != meta['bytes']):
            raise RuntimeError('Checkpoint file size/hash metadata mismatch: ' + name)
        # rsync verifies transferred data; all shard sizes and small metadata
        # hashes are compared separately. This is not a full tensor-content hash.
        if name in critical:
            if (meta['bytes'] > 32 * 1024**2
                    or not re.fullmatch('[0-9a-f]{64}', meta.get('sha256', ''))
                    or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']):
                raise RuntimeError('Checkpoint metadata changed after verified retention: ' + name)
    snapshot = _checkpoint_inventory(CHECKPOINT, PROFILE_ITERATION)
    if snapshot['shards'] != sorted((n, m['bytes']) for n, m in files.items() if n.endswith('.distcp')):
        raise RuntimeError('Checkpoint shard inventory differs from retained manifest')
    fingerprint = snapshot['id']
    if record.get('source_checkpoint_id') != fingerprint or record.get('destination_checkpoint_id') != fingerprint:
        raise RuntimeError('Retained checkpoint fingerprint differs from its verified source')
    return {'record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
            'checkpoint_id': fingerprint, 'iteration': PROFILE_ITERATION}


def _input_staging_summary(raid_root, node, source_roots):
    """Verify the storage agent's completed local input inventory, without GPU IO."""
    import hashlib
    import json
    from pathlib import Path

    raid = Path(raid_root)
    record_path = raid / 'manifests/input-staging.json'
    record = json.loads(record_path.read_text())
    required = {'schema_version': 1, 'status': 'complete', 'uid': 28644, 'gid': 30,
                'node': node, 'raid_root': raid_root, 'verification': 'shard_inventory_sizes_and_metadata_sha256'}
    if any(record.get(key) != value for key, value in required.items()):
        raise RuntimeError('Missing or mismatched completed input-staging record')
    for path in (raid, record_path):
        if path.resolve() != path or path.stat().st_uid != 28644:
            raise RuntimeError('Input staging root/record is symlinked or foreign-owned')
    if set(record.get('models', {})) != set(source_roots):
        raise RuntimeError('Input staging must contain the exact HF and reference model pair')
    summary = {}
    for name, expected_source in source_roots.items():
        model = record['models'][name]
        destination = raid / 'models' / name
        if model.get('source_root') != expected_source or model.get('destination_root') != str(destination):
            raise RuntimeError('Input staging source/destination mismatch: ' + name)
        if destination.resolve() != destination or destination.stat().st_uid != 28644:
            raise RuntimeError('Model staging root is symlinked or foreign-owned: ' + name)
        files = {}
        for path in destination.rglob('*'):
            if path.resolve() != path or path.stat().st_uid != 28644:
                raise RuntimeError('Staged model path is symlinked or foreign-owned: ' + str(path))
            if path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError('Staged model path is not a regular file: ' + str(path))
            entry = {'bytes': path.stat().st_size}
            if path.suffix not in ('.safetensors', '.distcp'):
                entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(destination))] = entry
        fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if (not files or files != model.get('files') or fingerprint != model.get('fingerprint')
                or len(files) != model.get('file_count')
                or sum(entry['bytes'] for entry in files.values()) != model.get('total_bytes')):
            raise RuntimeError('Staged input inventory or metadata changed: ' + name)
        summary[name] = {'fingerprint': fingerprint, 'file_count': len(files), 'total_bytes': model['total_bytes']}
    return {'record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(), 'models': summary}


def check_node_prerequisites(c, retained_checkpoint, main_iteration=MAIN_FINAL_ITERATION, saved_stop=None):
    sources = {name: host_input(c, BASE + '/models/' + name)
               for name in ('Qwen3-30B-A3B', 'Qwen3-30B-A3B_torch_dist')}
    code = ('import json\n' + inspect.getsource(checkpoint_metadata) + '\n' + inspect.getsource(_checkpoint_inventory) + '\n'
            + inspect.getsource(_input_staging_summary) + '\n'
            + 'result = {"inputs": _input_staging_summary(' + repr(c['raid_root']) + ',' + repr(c['node']) + ',' + repr(sources) + '),'
            + '"main_final_checkpoint": _checkpoint_inventory(' + repr(c['runtime_output'] + '/checkpoints') + ',' + str(main_iteration) + ')}\n')
    if not c.get('initial_policy'):
        code += 'result["selected_checkpoint"]=_checkpoint_inventory(' + repr(c['profile_checkpoint']) + ',' + str(PROFILE_ITERATION) + ')\n'
    if saved_stop is not None:
        code += ('from pathlib import Path\nimport hashlib\n'
            + 'watch=Path(' + repr(c['runtime_output'] + '/watchdog.json') + ')\n'
            + 'sentinel=Path(' + repr(c['runtime_output'] + '/checkpoint-now') + ')\n'
            + 'assert watch.resolve()==watch and watch.stat().st_uid==28644\n'
            + 'assert hashlib.sha256(watch.read_bytes()).hexdigest()==' + repr(saved_stop['watchdog_sha256']) + '\n'
            + 'assert not sentinel.exists() and not sentinel.is_symlink(), "Save sentinel still present"\n'
            + 'result["saved_stop_sentinel_absent"]=True\n')
    result = node_python(c, code + 'print(json.dumps(result))', 120)
    if not c.get('initial_policy') and result['selected_checkpoint']['id'] != retained_checkpoint['checkpoint_id']:
        raise RuntimeError('Local selected checkpoint differs from verified durable checkpoint')
    initial_id = hashlib.sha256(json.dumps({name: model['fingerprint']
        for name, model in result['inputs']['models'].items()}, sort_keys=True).encode()).hexdigest()
    return {'input_staging': result['inputs'], 'initial_model_id': initial_id,
            'profile_policy': 'initial_policy_fresh_optimizer' if c.get('initial_policy') else 'frozen_checkpoint9',
            'selected_checkpoint_id': None if c.get('initial_policy') else result['selected_checkpoint']['id'],
            'main_final_checkpoint_id': result['main_final_checkpoint']['id'],
            'main_final_iteration': main_iteration, 'profile_iteration': None if c.get('initial_policy') else PROFILE_ITERATION,
            'saved_stop_sentinel_absent': result.get('saved_stop_sentinel_absent')}


def _owned_record(path):
    path = Path(path)
    if path.resolve() != path or not path.is_file() or path.stat().st_uid != 28644:
        raise RuntimeError('Evidence missing, symlinked, or foreign-owned: ' + str(path))
    if path.stat().st_size > 32 * 1024**2:
        raise RuntimeError('Oversized evidence JSON: ' + str(path))
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _timestamp(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise RuntimeError('Evidence timestamp lacks timezone')
    return result.timestamp()


def _saved_stop_evidence(c, main, metrics, exits, retained):
    """Accept evidence of an existing original-watchdog saved stop; never stop it."""
    r = Path(c['root'])
    if main.get('status') != 'STOPPED':
        raise RuntimeError('Saved-stop evidence requires actual Ray STOPPED')
    retained_record, retained_sha = _owned_record(r / 'artifact-retention-exit.json')
    if retained_record != retained or retained.get('exit_code') != 0 or retained.get('uid') != 28644:
        raise RuntimeError('Small-artifact retention is unsuccessful or changed')
    exit_hashes = {}
    for filename, code in exits.items():
        record, exit_hashes[filename] = _owned_record(r / filename)
        if record.get('exit_code') != code:
            raise RuntimeError('Launcher exit record changed during verification')
    expected = SAVED_STOP_PROVENANCE[c['platform']]
    watch, watch_sha = _owned_record(r / 'node-output/watchdog.json')
    launch, launch_sha = _owned_record(r / 'watchdog-launch.json')
    training, training_sha = _owned_record(r / 'train-launch.json')
    if (launch_sha != expected['watchdog_launch_sha256'] or training_sha != expected['train_launch_sha256']
            or training.get('source_sha256', {}).get('lab/rubin_two_node/watch_qwen3_run.py') != WATCHDOG_SOURCE_SHA256):
        raise RuntimeError('Original watchdog/launch source provenance differs')
    driver = r / expected['driver_file']
    if driver.resolve() != driver or driver.stat().st_uid != 28644 or hashlib.sha256(driver.read_bytes()).hexdigest() != expected['driver_sha256']:
        raise RuntimeError('Audited stage-driver exit mapping changed')
    if hashlib.sha256(main['entrypoint'].encode()).hexdigest() != expected['entrypoint_sha256']:
        raise RuntimeError('Original main entrypoint hash differs')
    required = {'state': 'finished', 'terminal_status': 'STOPPED', 'ray_status': 'STOPPED',
        'run_id': c['main_run'], 'submission_id': c['main_id'], 'pid': expected['pid'],
        'armed_at': expected['armed_at'], 'ray_address': 'http://' + c['ip'] + ':28265',
        'soft_deadline_at': expected['soft'], 'deadline_at': expected['hard'], 'lease_deadline_at': c['lease'],
        'expected_rollouts': 50, 'stop_reason': 'soft_deadline_checkpoint_confirmed', 'outcome': 'partial',
        'sentinel': '/run-output/checkpoint-now', 'save_dir': '/run-output/checkpoints',
        'sentinel_created': True, 'sentinel_seen': True, 'last_stop_response': True,
        'training_complete_verified': False}
    if any(watch.get(k) != v for k, v in required.items()) or launch.get('pid') != expected['pid']:
        raise RuntimeError('Not an original-watchdog checkpoint-confirmed STOPPED outcome')
    if type(watch.get('stop_request_count')) is not int or watch['stop_request_count'] < 1:
        raise RuntimeError('No original-watchdog stop request was recorded')
    if any(watch.get(key) for key in ('checkpoint_error_type', 'final_artifact_read_error', 'last_stop_error_type')):
        raise RuntimeError('Saved-stop evidence contains unresolved checkpoint/stop errors')
    baseline, latest = watch.get('checkpoint_baseline'), watch.get('latest_checkpoint')
    if not isinstance(baseline, dict) or not isinstance(latest, dict):
        raise RuntimeError('Missing numeric checkpoint baseline/confirmation')
    before, iteration = baseline.get('iteration'), latest.get('iteration')
    if (type(before) is not int or type(iteration) is not int or not 0 <= before < iteration < 50
            or iteration < PROFILE_ITERATION or latest.get('directory_exists') is not True
            or latest.get('directory') != f'/run-output/checkpoints/iter_{iteration:07d}'
            or baseline.get('directory_exists') is not True
            or baseline.get('directory') != f'/run-output/checkpoints/iter_{before:07d}'):
        raise RuntimeError('Save confirmation did not advance the exact numeric checkpoint')
    requested, confirmed, finished = (_timestamp(watch[k]) for k in
                                      ('checkpoint_requested_at', 'checkpoint_confirmed_at', 'completed_at'))
    soft, hard = _timestamp(expected['soft']), _timestamp(expected['hard'])
    if not (_timestamp(expected['armed_at']) < _timestamp(training['started_utc']) < soft
            <= requested <= confirmed <= finished < hard
            and finished <= _timestamp(retained['retained_at'])):
        raise RuntimeError('Saved-stop chronology differs from original soft/hard deadlines or retention')
    ended = main.get('end_time')
    if not isinstance(ended, (int, float)) or not confirmed <= ended / 1000 <= finished:
        raise RuntimeError('Ray terminal time is not after the confirmed save')
    # Source-confirmed mapping is deliberately narrow; signals/SSH failures are not success.
    if any(type(value) is not int or value != 0 for value in exits.values()):
        raise RuntimeError('Unexplained saved-stop launcher exit mapping')
    if main.get('driver_exit_code') is not None or watch.get('driver_exit_code') is not None:
        raise RuntimeError('Unexpected STOPPED Ray driver_exit_code; needs separate source review')
    watch_log = r / 'node-output/logs/watchdog.log'
    if watch_log.resolve() != watch_log or watch_log.stat().st_uid != 28644:
        raise RuntimeError('Watchdog transcript missing or foreign-owned')
    stopping_seen = terminal_seen = False
    with watch_log.open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            terminal_seen |= event == watch
            stopping_seen |= (event.get('state') == 'stopping' and event.get('submission_id') == c['main_id']
                and event.get('stop_reason') == required['stop_reason']
                and (event.get('latest_checkpoint') or {}).get('iteration') == iteration)
    if not stopping_seen or not terminal_seen:
        raise RuntimeError('Retained watchdog transcript does not corroborate saved-stop sequence')
    complete = metrics['completed_training_rollouts']
    if (metrics.get('parse_errors') or metrics.get('conflicts') or not complete
            or complete != list(range(len(complete))) or iteration not in complete):
        raise RuntimeError('Learning log lacks a complete, conflict-free prefix through saved checkpoint')
    steps = [step for row in metrics['rows'] for step in row['train_steps']]
    ids = sorted(step['logged_id'] for step in steps)
    if ids != list(range(len(ids))) or not 4 * len(complete) <= len(ids) <= min(200, 4 * len(complete) + 3):
        raise RuntimeError('Observed optimizer updates are not a contiguous learning prefix')
    for step in steps:
        for key in ('train/loss', 'train/grad_norm'):
            value = step['metrics'].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError('Non-finite/missing learning metric before saved stop: ' + key)
    fatal = re.compile(r'(?:torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory|outcome=(?:ERROR|FAILED)|valid_step=false)', re.I)
    with (r / 'logs/qwen3_train.log').open(errors='replace') as stream:
        if any(fatal.search(line) for line in stream):
            raise RuntimeError('Training failure/OOM/invalid step is not an eligible saved stop')
    return {'mode': 'original_watchdog_saved_STOPPED', 'main_status': 'STOPPED', 'partial': True,
            'checkpoint_iteration': iteration, 'saved_optimizer_updates': (iteration + 1) * 4,
            'observed_optimizer_updates': len(ids), 'completed_rollouts': complete,
            'watchdog_sha256': watch_sha, 'watchdog_launch_sha256': launch_sha, 'train_launch_sha256': training_sha,
            'watchdog_source_sha256': WATCHDOG_SOURCE_SHA256,
            'original_soft_deadline': expected['soft'], 'original_hard_deadline': expected['hard'],
            'original_lease_deadline': c['lease'], 'stop_reason': watch['stop_reason'],
            'launcher_exits': exits, 'launcher_exit_sha256': exit_hashes,
            'small_artifact_retention_sha256': retained_sha,
            'watchdog_transcript_sha256': hashlib.sha256(watch_log.read_bytes()).hexdigest(),
            'driver_source_sha256': expected['driver_sha256'],
            'ray_driver_exit_code': main.get('driver_exit_code'),
            'checkpoint_requested_at': watch['checkpoint_requested_at'],
            'checkpoint_confirmed_at': watch['checkpoint_confirmed_at'], 'terminal_at': watch['completed_at']}


def check_main(c, allow_saved_stopped=False):
    r = Path(c['root'])
    exits = {filename: json.loads((r / filename).read_text()).get('exit_code')
             for filename in ('train_exit.json', 'train-driver-exit.json')}
    for filename, value in exits.items():
        if value != 0:
            raise RuntimeError('Main launcher did not exit successfully: ' + filename)
    retained = json.loads((r / 'artifact-retention-exit.json').read_text())
    if retained.get('exit_code') != 0 or retained.get('uid') != 28644:
        raise RuntimeError('Main node-local evidence has not been retained by normal UID')
    current = jobs(c, 'http://' + c['ip'] + ':28265')
    if any(j.get('type') == 'SUBMISSION' and j.get('status') not in TERMINAL for j in current):
        raise RuntimeError('Main Ray cluster still has an active/unknown job')
    matches = [j for j in current if j.get('submission_id') == c['main_id']]
    if len(matches) != 1:
        raise RuntimeError('Exact main Ray identity is missing or duplicated')
    main = matches[0]
    saved_stop = allow_saved_stopped and main.get('status') == 'STOPPED'
    if main.get('status') != 'SUCCEEDED' and not saved_stop:
        raise RuntimeError('Exact main Ray job is not SUCCEEDED')
    if saved_stop and any(j.get('status') not in TERMINAL for j in current):
        raise RuntimeError('Saved-stop alternate requires every main-cluster job to be terminal')
    if main.get('runtime_env', {}).get('env_vars', {}).get('RUBIN_RUN_ID') != c['main_run']:
        raise RuntimeError('Main Ray identity mismatch')
    original = json.loads((r / 'original-ray-job.json').read_text())
    recorded_env = original.get('runtime_env', {}).get('env_vars', {})
    unexpected_routing = {k: recorded_env[k] for k in ('RAY_ADDRESS', 'RAY_API_SERVER_ADDRESS', 'MILES_SCRIPT_EXTERNAL_RAY') if k in recorded_env}
    if unexpected_routing:
        raise RuntimeError('Review recorded Ray routing before replay; will not rewrite it: ' + repr(unexpected_routing))
    if main.get('entrypoint') != original.get('entrypoint'):
        raise RuntimeError('Main entrypoint differs from saved source plan')
    spec = importlib.util.spec_from_file_location('qwen3_summary', REPO + '/lab/rubin_two_node/summarize_qwen3_runs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.summarize_run(c['platform'], r / 'logs/qwen3_train.log', {'status': main['status']})
    alternate = _saved_stop_evidence(c, main, metrics, exits, retained) if saved_stop else None
    if not saved_stop and (metrics['partial'] or metrics['completed_training_rollouts'] != list(range(50))):
        raise RuntimeError('Main log does not prove all 50 rollouts/200 updates')
    retained_checkpoint = None if c.get('initial_policy') else check_checkpoint_retention()
    if saved_stop:
        node_prerequisites = check_node_prerequisites(c, retained_checkpoint, alternate['checkpoint_iteration'], alternate)
    else:
        node_prerequisites = check_node_prerequisites(c, retained_checkpoint)
    return {'checked_at': utc(), 'ray_job': main, 'main_status': main['status'], 'partial': bool(saved_stop),
            'completion_mode': alternate['mode'] if saved_stop else 'natural_SUCCEEDED_50_200',
            'saved_stop_evidence': alternate, 'log_sha256': metrics['source_log_sha256'],
            'completed_rollouts': metrics['completed_training_rollouts'],
            'optimizer_updates': alternate['observed_optimizer_updates'] if saved_stop else 200,
            'checkpoint_retention_record_sha256': retained_checkpoint['record_sha256'] if retained_checkpoint else None,
            'node_prerequisites': node_prerequisites}


def inspect_profile(c, deadline=None):
    data = json.loads(remote(c, ['docker', 'inspect', c['name']], bounded(deadline, 60)))[0]
    if data['Config']['Labels'].get('miles.profile_run') != c['run_id'] or data['Config']['Image'] != c['image']:
        raise RuntimeError('Profile container identity/image mismatch')
    if data['Config']['User'] != '28644:30':
        raise RuntimeError('Unexpected profile writer UID')
    return data


def prepare(c, args):
    allocation(c)
    seconds = budget(c, args.max_runtime_seconds, args.retention_margin_seconds)
    evidence = check_main(c, allow_saved_stopped=getattr(args, 'allow_saved_stopped_main', False))
    if not args.allow_stop_completed_main:
        raise RuntimeError('prepare requires --allow-stop-completed-main after reviewing exact main evidence')
    durable = Path(c['durable'])
    durable.mkdir(parents=True, exist_ok=False)
    write_json(durable / 'main-completion.json', evidence)
    # Prove normal login writer/read/delete before creating a runtime output.
    probe = durable / '.writer-probe'
    probe.write_text('ok')
    if probe.stat().st_uid != 28644 or probe.read_text() != 'ok':
        raise RuntimeError('Durable NFS writer identity check failed')
    probe.unlink()
    code = """import hashlib,json,os,pathlib,shutil,socket
c=CONFIG
assert (os.getuid(),os.getgid())==(28644,30)
p=pathlib.Path(c['local']); p.mkdir(mode=0o700)
assert shutil.disk_usage(p).free > 50*1024**3
for d in ['run','run/inputs','cache']+[f'cache/{x}' for x in ['home','tmp','ray','triton','flashinfer','torch_extensions','cuda','huggingface','pycache','xdg']]:
 (p/d).mkdir(exist_ok=True)
for port in PORTS+list(range(27400,28000)):
 with socket.socket() as s: s.bind(('0.0.0.0',port))
for name,expected in HASHES.items():
 assert hashlib.sha256((pathlib.Path(c['input'])/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'uid':os.getuid(),'free_bytes':shutil.disk_usage(p).free}))
""".replace('CONFIG', repr(c)).replace('PORTS', repr(PORTS)).replace('HASHES', repr(INPUT_HASHES))
    preflight = node_python(c, code)
    # Scope stop by verified name + immutable original image, never process name.
    main = json.loads(remote(c, ['docker', 'inspect', c['main']]))[0]
    if main['Config']['Image'] != c['image'] or main['Config']['User'] != '28644:30':
        raise RuntimeError('Main container image/writer mismatch; do not stop it')
    # Validate this exact image's Ray CLI before stopping the completed main.
    help_text = remote(c, ['docker', 'exec', c['main'], 'ray', 'start', '--help'])
    for flag in ('--dashboard-agent-grpc-port', '--dashboard-agent-listen-port', '--metrics-export-port'):
        if flag not in help_text:
            raise RuntimeError('This Ray CLI lacks planned isolation flag: ' + flag)
    # Recheck immediately before the only main-container mutation.
    repeated = check_main(c, allow_saved_stopped=getattr(args, 'allow_saved_stopped_main', False))
    if (repeated['main_status'], repeated['log_sha256'], repeated['node_prerequisites']) != (evidence['main_status'], evidence['log_sha256'], evidence['node_prerequisites']):
        raise RuntimeError('Main terminal evidence changed during preparation; do not stop container')
    remote(c, ['docker', 'stop', '--time', '30', c['main']], 60)
    remaining_gpu = remote(c, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip()
    if remaining_gpu:
        raise RuntimeError('GPU processes remain after exact main stop; inspect without killing them: ' + remaining_gpu)
    remote(c, docker_command(c), 90)
    if c['platform'] == 'gb300':
        remote(c, ['docker', 'exec', '--user', '0:0', c['name'], 'chmod', 'o+x', '/root'])
    inspect_profile(c)
    writer = "import os,pathlib; assert (os.getuid(),os.getgid())==(28644,30); p=pathlib.Path('/run-output/.container-probe'); p.write_text('ok'); print(p.read_text())"
    remote(c, ['docker', 'exec', c['name'], 'python3', '-c', writer])
    content = remote(c, ['cat', c['local'] + '/run/.container-probe'])
    if content.strip() != 'ok':
        raise RuntimeError('Node-local writer/read check failed')
    probe.write_text(content)
    if probe.stat().st_uid != 28644:
        raise RuntimeError('Retained probe owner mismatch')
    probe.unlink()
    remote(c, ['rm', '--', c['local'] + '/run/.container-probe'])
    remote(c, ray_command(c), 120)
    plan_text = remote(c, replay_command(c, seconds), 120)
    (durable / 'replay-plan.txt').write_text(plan_text)
    write_json(durable / 'operator-prepared.json', {'config': c, 'prepared_at': utc(), 'preflight': preflight,
        'max_runtime_seconds_at_prepare': seconds, 'retention_margin_seconds': args.retention_margin_seconds,
        'main_completion_mode': evidence['completion_mode'],
        'saved_stop_opt_in_requested': getattr(args, 'allow_saved_stopped_main', False),
        'nofile_soft_hard': [65535, 65535], 'input_hashes': INPUT_HASHES,
        'source_commit': run(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip(),
        'source_plan_sha256': hashlib.sha256(Path(c['source_plan']).read_bytes()).hexdigest()})
    print(plan_text)


def submit(c, args):
    allocation(c)
    inspect_profile(c)
    identity = args.expected_initial_model_id if c.get('initial_policy') else args.expected_checkpoint_id
    if not re.fullmatch('[0-9a-f]{64}', identity):
        raise RuntimeError('Supply the reviewed 64-hex initial-model/checkpoint ID; reuse that exact ID for the other platform')
    if not Path(c['durable'], 'operator-prepared.json').is_file():
        raise RuntimeError('Missing prepare record')
    seconds = budget(c, args.max_runtime_seconds, args.retention_margin_seconds)
    prepared = json.loads(Path(c['durable'], 'operator-prepared.json').read_text())
    if prepared.get('config', {}).get('initial_policy', False) != c.get('initial_policy', False):
        raise RuntimeError('Profile policy differs from the prepared container')
    command = replay_command(c, seconds, True, args.expected_checkpoint_id, getattr(args, 'expected_initial_model_id', ''))
    text = remote(c, command, 120)
    (Path(c['durable']) / 'submit-result.txt').write_text(text)
    print(text)


def profile_terminal(c, deadline=None):
    inspect_profile(c, deadline)
    current = jobs(c, c['dashboard'], deadline)
    if any(j.get('type') == 'SUBMISSION' and j.get('status') not in TERMINAL for j in current):
        raise RuntimeError('Profile cluster has an active/unknown job; do not retain/stop yet')
    found = [j for j in current if j.get('submission_id') == c['submission_id']]
    if len(found) != 1 or found[0]['runtime_env']['env_vars'].get('MILES_PROFILE_RUN_ID') != c['run_id']:
        raise RuntimeError('Exact terminal profile identity not found')
    return found[0]


def manifest(c, deadline):
    code = _remote_deadline_preamble(deadline) + """import hashlib,json,os,pathlib
root=pathlib.Path(ROOT); result={}
def fail(e): raise e
for base,dirs,files in os.walk(root,onerror=fail,followlinks=False):
 _check_deadline()
 for name in dirs+files:
  p=pathlib.Path(base,name)
  if p.is_symlink(): raise RuntimeError('Refuse symlink: '+str(p))
 for name in files:
  _check_deadline()
  p=pathlib.Path(base,name); h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(4*1024**2),b''):
    _check_deadline(); h.update(block)
  result[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
_check_deadline()
print(json.dumps(result))
""".replace('ROOT', repr(c['local'] + '/run'))
    return node_python(c, code, deadline.remaining(600))


def _verify_retained(destination, files, deadline):
    for name, meta in files.items():
        deadline.remaining()
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe retained path: ' + name)
        p = destination / relative
        if p.resolve() != p or p.stat().st_uid != 28644 or not p.is_file():
            raise RuntimeError('Retained file owner/type/path mismatch: ' + name)
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                deadline.remaining()
                h.update(block)
        if p.stat().st_size != meta['bytes'] or h.hexdigest() != meta['sha256']:
            raise RuntimeError('Retained file mismatch: ' + name)
    deadline.remaining()


def _operation(c, args, action, function):
    deadline = OperationDeadline(c, args.retention_margin_seconds, action,
        lease_reserve_seconds=120 if action == 'retain' else 0)
    attempt = {'action': action, 'state': 'running', 'run_id': c['run_id'],
               'attempt_id': str(os.getpid()) + '-' + str(time.time_ns()),
               'requested_margin_seconds': args.retention_margin_seconds,
               'actual_total_bytes': None, 'trace_limit_is_polled_not_hard': True}
    audit_path = Path(c['durable'], action + '-attempt.json')
    def audit():
        attempt.update(deadline.describe())
        _write_before_deadline(audit_path, attempt, deadline)
    try:
        with deadline.enforce():
            audit()
            function(deadline, attempt, audit)
            attempt.update(state='verified', completed_at=utc())
            audit()
    except BaseException as exc:
        attempt.update(deadline.describe(), state='failed', error=type(exc).__name__ + ': ' + str(exc))
        # Once expired, do not start a new file write. Prior progress/partial files
        # remain; final failure is also emitted to the operator's retained stdout.
        try:
            with deadline.enforce():
                audit()
        except (TimeoutError, OSError):
            pass
        print(json.dumps(attempt), flush=True)
        raise
    print(json.dumps(attempt), flush=True)


def retain(c, args):
    def retain_work(deadline, attempt, audit):
        job = deadline.stage('terminal_identity', lambda: profile_terminal(c, deadline))
        audit()
        capture = _remote_deadline_preamble(deadline) + """import json,pathlib,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
url=URL
with opener.open(url+'/logs',timeout=min(30,_deadline-time.time())) as response: logs=json.load(response)
_check_deadline()
p=pathlib.Path('/run-output/replay')
(p/'job-driver.log').write_text(logs['logs'])
_check_deadline()
(p/'final-ray-job.json').write_text(json.dumps(JOB,indent=2)+'\\n')
_check_deadline()
print(json.dumps({'driver_log_bytes':(p/'job-driver.log').stat().st_size}))
""".replace('URL', repr(c['dashboard'] + '/api/jobs/' + c['submission_id'])).replace('JOB', repr(job))
        deadline.stage('capture_driver_evidence', lambda: remote(c,
            ['docker', 'exec', c['name'], 'python3', '-c', capture], deadline.remaining(45)))
        audit()
        destination = Path(c['durable'], 'node-output')
        deadline.remaining()
        destination.mkdir(exist_ok=True)
        before = deadline.stage('source_manifest_before', lambda: manifest(c, deadline))
        total = sum(v['bytes'] for v in before.values())
        attempt['actual_total_bytes'] = total
        audit()
        deadline.remaining()
        if shutil.disk_usage(destination).free < total + 2*1024**3:
            raise RuntimeError('Insufficient durable retention capacity')
        deadline.stage('rsync', lambda: subprocess.run(
            ['rsync', '-rt', '--no-perms', '--omit-dir-times', '--partial',
             c['node'] + ':' + c['local'] + '/run/', str(destination) + '/'],
            check=True, timeout=deadline.remaining()))
        audit()
        after = deadline.stage('source_manifest_after', lambda: manifest(c, deadline))
        if before != after:
            raise RuntimeError('Output changed while retaining; repeat retain after guard finishes')
        audit()
        deadline.stage('destination_hashes', lambda: _verify_retained(destination, after, deadline))
        audit()
        record = {'retained_at': utc(), 'uid': os.getuid(), 'state': 'verified',
            'run_id': c['run_id'], 'attempt_id': attempt['attempt_id'], 'terminal_ray_job': job,
            'bytes': total, 'actual_total_bytes': total, 'files': after,
            'operation_budget': deadline.describe(), 'attempt_audit': str(Path(c['durable'], 'retain-attempt.json')),
            'trace_limit_is_polled_not_hard': True}
        deadline.stage('verified_retention_record', lambda: _write_before_deadline(
            Path(c['durable'], 'retention.json'), record, deadline))
    _operation(c, args, 'retain', retain_work)


def stop(c, args):
    def stop_work(deadline, attempt, audit):
        job = deadline.stage('terminal_identity', lambda: profile_terminal(c, deadline))
        deadline.remaining()
        record = json.loads(Path(c['durable'], 'retention.json').read_text())
        retained = json.loads(Path(c['durable'], 'retain-attempt.json').read_text())
        if (record.get('state') != 'verified' or retained.get('state') != 'verified'
                or record.get('attempt_id') != retained.get('attempt_id')
                or record['run_id'] != c['run_id']
                or record['terminal_ray_job']['submission_id'] != job['submission_id']):
            raise RuntimeError('Retention record identity or completed verification mismatch')
        attempt['actual_total_bytes'] = record['bytes']
        if deadline.stage('source_manifest_before_stop', lambda: manifest(c, deadline)) != record['files']:
            raise RuntimeError('Artifacts changed since retention; retain again before stop')
        audit()
        # Every mutation retains the exact verified profile container namespace.
        deadline.stage('ray_stop', lambda: remote(c,
            ['docker', 'exec', c['name'], 'ray', 'stop', '--force'], deadline.remaining(60)))
        audit()
        deadline.stage('container_stop', lambda: remote(c,
            ['docker', 'stop', '--time', '30', c['name']], deadline.remaining(60)))
        audit()
        deadline.stage('stopped_record', lambda: _write_before_deadline(
            Path(c['durable'], 'operator-stopped.json'), {'stopped_at': utc(), 'container': c['name'],
                'operation_budget': deadline.describe(), 'attempt_id': attempt['attempt_id']}, deadline))
    _operation(c, args, 'stop', stop_work)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('platform', choices=RUNS)
    p.add_argument('--action', choices=['prepare', 'submit', 'retain', 'stop'], default='prepare')
    p.add_argument('--run-id', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--allow-stop-completed-main', action='store_true')
    p.add_argument('--allow-saved-stopped-main', action='store_true',
                   help='Opt in only to an already-terminal, original-watchdog checkpoint-confirmed STOPPED main')
    p.add_argument('--expected-checkpoint-id', default='')
    p.add_argument('--initial-policy', action='store_true', help='Profile initial HF/release policy with fresh optimizer; no checkpoint9 dependency')
    p.add_argument('--expected-initial-model-id', default='')
    p.add_argument('--max-runtime-seconds', type=int, default=4500)
    p.add_argument('--retention-margin-seconds', type=int, default=1200)
    args = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', args.run_id):
        p.error('Use a simple unique run ID of at most64 characters')
    if args.max_runtime_seconds < 900 or args.retention_margin_seconds < 600:
        p.error('Require >=900s requested runtime and >=600s retention reserve (default1200s)')
    if (args.initial_policy and args.expected_checkpoint_id) or (not args.initial_policy and args.expected_initial_model_id):
        p.error('Use only the identity selector matching the chosen profile policy')
    c = config(args)
    if not args.execute:
        print(json.dumps({'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'action': args.action, 'config': c,
            'saved_stopped_opt_in': args.allow_saved_stopped_main,
            'saved_stopped_alternate': ('exact original-watchdog saved STOPPED; advanced native checkpointR; actual partial counts; no active jobs'
                                        if args.allow_saved_stopped_main else None),
            'completion_gates': ['exact main Ray SUCCEEDED', 'both launcher exits0', '50rollouts/200updates',
                                 'main node-local evidence retained', 'own native final checkpoint49',
                                 ('shared initial HF/release model fingerprint verified' if args.initial_policy
                                  else 'frozen Rubin checkpoint9 retained and local copy fingerprint matches'),
                                 'local RAID model staging inventory and metadata verified',
                                 'no active Ray submissions'],
            'docker_argv': docker_command(c), 'ray_start_argv': ray_command(c),
            'replay_plan_argv': replay_command(c, args.max_runtime_seconds),
            'runtime_budget': 'min(requested, recorded lease remaining - retention margin), recomputed before submit',
            'retention_margin_seconds': args.retention_margin_seconds,
            'retention_deadline': 'min(retain start + requested margin, recorded lease -120s), shared across all stages',
            'stop_deadline': 'min(stop start + requested margin, recorded lease), shared across identity, hashing and scoped commands',
            'replay_workload': {'policy': 'initial_policy_fresh_optimizer' if args.initial_policy else 'frozen_checkpoint9',
                'rollout_ids': [0, 1] if args.initial_policy else [10, 11], 'rollouts': 2, 'optimizer_updates': 8,
                'original_training_horizon': 50 if args.initial_policy else None,
                'profile_step_start': 1, 'profile_step_end': 2,
                'trace_export': 'At second train end, before CPU backup; no third rollout required'},
            'caveats': [('Initial-policy profiling uses shared HF/release inputs and fresh optimizer/RNG; no late-policy performance claim.'
                         if args.initial_policy else 'Both runtimes replay frozen checkpoint9; cross-version resume untested.'),
                        'Main must be naturally SUCCEEDED or explicitly verified original-watchdog saved STOPPED; never interrupt a healthy job.',
                        'Both replays explicitly use4096 tokens/GPU; GB main used8192 and Rubin A2 main4096.',
                        'Snapshot capture/retention/local staging are external prerequisites; no automatic copy/overwrite.',
                        '10GiB is a polled trace limit; API failure may delay scoped stopping.',
                        'Cold loading is inside runtime budget; insufficient lease means skip, never extend.',
                        'SGLang bounded4-step API capture must target one replay engine, not the main job.',
                        'Plan uses current checked-out Miles source read-only; record its commit before execution.',
                        'GB /root traverse change is scoped to the new container; no image/package changes.']}, indent=2))
        return
    if (os.getuid(), os.getgid()) != (28644, 30) or not Path(REPO).is_dir():
        raise RuntimeError('Execution must run on dl3 with normal UID28644:GID30 and direct NFS repo')
    if args.action == 'prepare': prepare(c, args)
    elif args.action == 'submit': submit(c, args)
    elif args.action == 'retain': retain(c, args)
    else: stop(c, args)


if __name__ == '__main__':
    main()
