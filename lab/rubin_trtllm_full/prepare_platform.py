#!/usr/bin/env python3
"""Prepare a fixed TRTLLM allocation; plan-only unless --execute.

Run as UID 28644:GID 30 on dl3. Images are built/pulled externally first.
Supply --platform, --source, --source-commit, --source-manifest and --wait-until;
GB300 additionally requires --image (an immutable registry digest). --resume
reuses verified preparation receipts, never resets leases or resubmits training.
--execute-main submits the 50-rollout main driver only after all preflights pass.
No allocation, image build, container deletion or checkpoint deletion is done.
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
import subprocess
import sys
import tempfile
import time


BASE = Path('/home/scratch.kaixih_ent')
JOBS = {'rubin': '2212643', 'gb300': '2212644'}
RUBIN_IMAGE = ('gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:'
               '405315ba3add773be16cfe176a4dcd15a1071cbf17f3be255ab2c47324c65f34')
MODEL_RECORD = BASE / 'repro/miles-qwen3-cudagraph/20260916-gb300-j2203647-cg/preparation.json'
DATA_ROOT = BASE / 'repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs/inputs'
FINGERPRINTS = {
    'Qwen3-30B-A3B': 'd7dd1251002eafcb34b4cd19748fc198435bf567291a1b2950371baae487dbf7',
    'Qwen3-30B-A3B_torch_dist': '2941741527169e4ffaa95476e3a30ccbe045a7d0c5fd3edd635772def3c800a1',
}
DATA_HASHES = {
    'train.jsonl': 'f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8',
    'test-fixed-256.jsonl': '93ed3ccda6ecd09ce0665d0423bf8720b7db97823b0ce2a1c4d20483f410ce99',
}
RUNTIME_FILES = ['lab/rubin_two_node/' + name for name in (
    'orchestrate_rubin.py', 'run_qwen3_30b_a3b_gsm8k_rubin.py', 'gsm8k_verl_reward.py',
    'watch_qwen3_run.py', 'run_cudagraph_main.py')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'An explicit timezone is required')
    return parsed.timestamp()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_view(platform, path):
    path = str(path)
    require(path.startswith(str(BASE) + '/'), 'Shared path must be within the authorized scratch root')
    return '/mnt/cifs' + path if platform == 'gb300' else path


def validate_options(args):
    job = str(getattr(args, 'job_id', None) or JOBS[args.platform])
    require(re.fullmatch(r'[1-9][0-9]*', job), 'Job ID must be a positive decimal allocation ID')
    tag = getattr(args, 'attempt_tag', None)
    if tag is not None:
        require(re.fullmatch(r'[a-z][a-z0-9]{0,11}', tag) and args.platform == 'rubin'
                and getattr(args, 'job_id', None), 'Attempt tag requires an explicit Rubin job and a short lowercase tag')
    image = args.image or (RUBIN_IMAGE if args.platform == 'rubin' else '')
    require(re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image), 'An immutable registry image digest is required')
    if args.platform == 'rubin':
        require(image == RUBIN_IMAGE, 'Rubin must use the tested two-fix image')
    require(re.fullmatch(r'[0-9a-f]{40}', args.source_commit), 'Source commit must be exact')
    for name in ('source', 'source_manifest', 'model_record', 'data_root'):
        p = Path(getattr(args, name))
        require(p.is_absolute() and '..' not in p.parts and str(p).startswith(str(BASE) + '/'),
                name + ' must be a normalized scratch path')
    timestamp(args.wait_until)
    expected_manifest_sha = getattr(args, 'source_manifest_sha256', None)
    if expected_manifest_sha is not None:
        require(re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha), 'Expected source manifest SHA256 is invalid')
    if tag:
        require(expected_manifest_sha is not None, 'Tagged attempt requires an explicit source manifest SHA256')
    reuse = getattr(args, 'reuse_node_models', None)
    if reuse:
        p = Path(reuse)
        require(p.is_absolute() and '..' not in p.parts and p.name == 'models'
                and p.parent.parent in (Path('/tmp'), Path('/raid/tmp'), Path('/raid/dldata'))
                and re.fullmatch(r'miles-kaixih-j' + re.escape(job) + r'-trtllm(?:-[a-z][a-z0-9]{0,11})?', p.parent.name),
                'Reused models must be an explicit prior local model directory for this job')
    run_id = f'20260917-{args.platform}-j{job}-trtllm' + ('-' + tag if tag else '')
    result = {'platform': args.platform, 'job_id': job, 'run_id': run_id,
            'run_dir': str(BASE / 'repro/miles-qwen3-trtllm-full' / run_id),
            'image': image, 'source': str(args.source), 'source_commit': args.source_commit,
            'source_manifest': str(args.source_manifest), 'model_record': str(args.model_record),
            'data_root': str(args.data_root), 'wait_until': args.wait_until}
    if tag:
        result['attempt_tag'] = tag
    if expected_manifest_sha:
        result['source_manifest_sha256'] = expected_manifest_sha
    if reuse:
        result['reuse_node_models'] = str(reuse)
    return result


def allocation_guard(config, record, now, original=None):
    require(record.get('JobId') == config['job_id'], 'Wrong allocation ID')
    require(re.fullmatch(r'[^()]+\(28644\)', record.get('UserId', '')), 'Wrong allocation owner')
    require(re.fullmatch(r'[^()]+\(30\)', record.get('GroupId', '')), 'Wrong allocation group')
    state = record.get('JobState')
    if original:
        for key in ('JobId', 'NodeList', 'StartTime', 'EndTime'):
            require(record.get(key) == original.get(key), 'Allocation identity/lease changed: ' + key)
        require(state == 'RUNNING', 'Allocation is no longer running')
    if state == 'PENDING' and not original:
        return 'wait'
    require(state == 'RUNNING', 'Allocation is neither pending nor running')
    require(record.get('NumNodes') == '1', 'Exactly one allocated node is required')
    require(re.search(r'(?:^|,)gres/gpu=4(?:,|$)', record.get('AllocTRES', '')),
            'Exactly four allocated GPUs are required')
    node = record.get('NodeList', '')
    require(re.fullmatch(r'[A-Za-z0-9.-]+', node) and node not in ('None', 'null'), 'An actual single NodeList is required')
    require(record.get('TimeLimit') == '08:00:00', 'The actual allocation must have its original eight-hour lease')
    start, end = [timestamp(record[name] + 'Z') for name in ('StartTime', 'EndTime')]
    # Slurm can round the independently recorded start/end to adjacent seconds.
    # Always retain and enforce the actual EndTime; this does not extend a lease.
    require(8 * 3600 <= end - start <= 8 * 3600 + 1 and end - now > 120,
            'Unexpected or expiring actual lease')
    return 'ready'


def verify_source(config):
    source = Path(config['source'])
    raw = Path(config['source_manifest']).read_bytes()
    require(not config.get('source_manifest_sha256')
            or hashlib.sha256(raw).hexdigest() == config['source_manifest_sha256'], 'Frozen source manifest SHA256 mismatch')
    manifest = json.loads(raw)
    require(manifest['git_commit'] == config['source_commit'], 'Frozen source commit mismatch')
    require(manifest.get('source_root', str(source)) == str(source), 'Frozen source root mismatch')
    hashes = manifest['source_sha256']
    require(set(RUNTIME_FILES) <= hashes.keys(), 'Source manifest must include all five launcher/driver files')
    require(source.is_dir() and source.stat().st_uid == 28644 and not source.is_symlink(), 'Invalid frozen source root')
    for name, expected in hashes.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source manifest path')
        p = source / rel
        require(not any(parent.is_symlink() for parent in [p, *p.parents]), 'Symlink in verified source path')
        require(file_hash(p) == expected, 'Frozen source changed: ' + name)
    return manifest, {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'git_commit': config['source_commit'], 'files_checked': len(hashes)}


def check_models(models, expected):
    for name, fingerprint in FINGERPRINTS.items():
        files = expected[name]['files']
        require(hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == fingerprint,
                'Canonical model record fingerprint changed: ' + name)
        require(len(files) == (26 if name == 'Qwen3-30B-A3B' else 20), 'Canonical model count changed')
        require(all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in files), 'Unsafe model path')
        require(models[name]['fingerprint'] == fingerprint and models[name]['files'] == files,
                'Staged model inventory/metadata mismatch: ' + name)


def host_storage(config, existing=None):
    import json, os, shutil, socket, subprocess, tempfile
    from pathlib import Path
    assert (os.getuid(), os.getgid()) == (28644, 30)
    assert socket.gethostname().split('.')[0] == config['node'].split('.')[0]
    nic = 'mp0' if config['platform'] == 'rubin' else 'enP5p9s0'
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address'], text=True))
    addresses = [a['local'] for i in interfaces if i['ifname'] == nic
                 for a in i['addr_info'] if a['family'] == 'inet' and a['scope'] == 'global']
    assert len(addresses) == 1, 'Inspect unexpected compute interface'
    shared = ('/mnt/cifs' if config['platform'] == 'gb300' else '') + '/home/scratch.kaixih_ent'
    shared_mount = subprocess.check_output(['findmnt', '-T', shared, '-n', '-o', 'SOURCE,FSTYPE,TARGET'], text=True)
    assert ('cifs' if config['platform'] == 'gb300' else 'nfs') in shared_mount.lower()
    suffix = 'miles-kaixih-j' + config['job_id'] + '-trtllm' + ('-' + config['attempt_tag'] if config.get('attempt_tag') else '')
    if existing:
        root = Path(existing['local_root'])
        assert root.name == suffix and root.parent in [Path('/raid/dldata'), Path('/raid/tmp'), Path('/tmp')]
        assert root.is_dir() and not root.is_symlink() and root.stat().st_uid == 28644
        assert json.loads((root / 'prep-identity.json').read_text()) == config
        assert shutil.disk_usage(root).free >= 180 * 1024**3, 'Insufficient remaining local checkpoint/cache space'
    else:
        candidates = []
        for parent in map(Path, ['/raid/dldata', '/raid/tmp', '/tmp']):
            if not parent.is_dir() or not os.access(parent, os.W_OK):
                continue
            kind = subprocess.check_output(['findmnt', '-T', str(parent), '-n', '-o', 'FSTYPE'], text=True).strip()
            if kind in ('ext4', 'xfs', 'btrfs') and shutil.disk_usage(parent).free >= 400 * 1024**3:
                candidates.append(parent)
        assert candidates, 'Require verified local storage with at least 400 GiB free'
        root = candidates[0] / suffix
        assert not root.exists(), 'Unrecorded local root exists: inspect before resuming'
        root.mkdir(mode=0o700)
        (root / 'prep-identity.json').write_text(json.dumps(config, sort_keys=True) + '\n')
    for name in ['run', 'run/inputs', 'run/logs', 'run/preflight', 'cache', 'models', 'diagnostics']:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        assert not p.is_symlink() and p.stat().st_uid == 28644 and os.access(p, os.W_OK)
    with tempfile.TemporaryDirectory(prefix='host-probe-', dir=root) as d:
        p = Path(d) / 'nested'; p.mkdir(); f = p / 'ok'; f.write_text('ok')
        assert f.read_text() == 'ok' and (f.stat().st_uid, f.stat().st_gid) == (28644, 30)
        f.unlink(); p.rmdir()
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,compute_cap', '--format=csv'], text=True)
    rows = gpus.strip().splitlines()[1:]
    assert len(rows) == 4
    assert all(('10.7' if config['platform'] == 'rubin' else 'NVIDIA GB300') in r for r in rows)
    return {'local_root': str(root), 'node_ip': addresses[0], 'nic': nic,
            'uid': 28644, 'gid': 30, 'free_bytes': shutil.disk_usage(root).free,
            'local_mount': subprocess.check_output(['findmnt', '-T', str(root), '-n', '-o', 'SOURCE,FSTYPE,TARGET'], text=True),
            'shared_mount': shared_mount, 'gpus': gpus, 'host_write_read_delete': 'PASS'}


def host_model_inventory(root):
    import hashlib, json
    from pathlib import Path
    root = Path(root); result = {}
    for name in ['Qwen3-30B-A3B', 'Qwen3-30B-A3B_torch_dist']:
        base = root / name; files = {}
        assert base.is_dir() and not base.is_symlink() and base.stat().st_uid == 28644
        for path in sorted(base.rglob('*')):
            assert not path.is_symlink() and path.stat().st_uid == 28644, str(path)
            if path.is_dir():
                continue
            assert path.is_file()
            item = {'bytes': path.stat().st_size}
            if path.suffix not in ('.safetensors', '.distcp'):
                item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(base))] = item
        result[name] = {'root': str(base), 'files': files, 'file_count': len(files),
                        'total_bytes': sum(x['bytes'] for x in files.values()),
                        'fingerprint': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    assert (root / 'Qwen3-30B-A3B_torch_dist/latest_checkpointed_iteration.txt').read_text().strip() == 'release'
    return result


def python_program(function, *args):
    return 'import json\n' + inspect.getsource(function) + '\nprint(json.dumps(' + function.__name__ + '(' + ','.join(repr(a) for a in args) + ')))'


class Worker:
    def __init__(self, config, args):
        self.c, self.args = config, args
        self.root = Path(config['run_dir'])
        self.original = None
        self.attempt = None
        self.counter = 0

    def record(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        os.replace(temp, path)

    def read(self, name):
        path = self.root / name
        return json.loads(path.read_text()) if path.exists() else None

    def allocation(self):
        raw = subprocess.check_output(['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o', self.c['job_id']], text=True, timeout=20)
        return dict(re.findall(r'(\w+)=([^\s]+)', raw))

    def guard(self):
        current = self.allocation()
        allocation_guard(self.c, current, time.time(), self.original)
        return current

    def remote(self, argv, *, input=None, timeout=120, label='node-command'):
        self.guard()
        remaining = int(timestamp(self.original['EndTime'] + 'Z') - time.time() - 120)
        timeout = min(timeout, remaining)
        require(timeout > 20, 'Insufficient lease for a bounded node command')
        # A node-side timeout also terminates work when the SSH client is interrupted.
        bounded = ['timeout', '--signal=TERM', '--kill-after=10s', str(timeout) + 's', *argv]
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
                   self.original['NodeList'], shlex.join(bounded)]
        self.counter += 1
        log = self.attempt / f'{self.counter:03d}-{label}.log'
        receipt = {'argv': argv, 'started_at': utc(), 'timeout_seconds': timeout, 'log': str(log)}
        try:
            with log.open('x') as stream:
                p = subprocess.run(command, input=input, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout + 25)
            receipt.update(returncode=p.returncode, ended_at=utc())
            self.record(str(log.relative_to(self.root)) + '.json', receipt)
            require(p.returncode == 0, f'{label} failed; inspect {log}')
            return log.read_text()
        except Exception as exc:
            receipt.update(error=type(exc).__name__ + ': ' + str(exc), ended_at=utc())
            self.record(str(log.relative_to(self.root)) + '.json', receipt)
            raise

    def remote_json(self, function, *args, label='node-probe'):
        text = self.remote(['python3', '-B', '-c', python_program(function, *args)], label=label)
        return json.loads(text.strip().splitlines()[-1])

    def stage(self, name, function, verify=None):
        receipt = self.read('prep-stages/' + name + '.json')
        if receipt:
            require(receipt['status'] == 'PASS', 'Invalid completed preparation receipt')
            if verify:
                verify(receipt['result'])
            return receipt['result']
        self.record('prep-state.json', {'stage': name, 'at': utc(), 'attempt': str(self.attempt)})
        result = function()
        self.record('prep-stages/' + name + '.json', {'status': 'PASS', 'ended_at': utc(), 'result': result})
        return result

    def storage_probe(self, storage):
        local = storage['local_root']
        parents = ['/run-output'] if self.c['platform'] == 'gb300' else ['/run-output', '/durable-output']
        code = """import json,os,tempfile
from pathlib import Path
assert (os.getuid(),os.getgid())==(28644,30)
result=[]
for parent in PARENTS:
 p=Path(tempfile.mkdtemp(prefix='writer-probe-',dir=parent));(p/'nested').mkdir();f=p/'nested/ok';f.write_text('ok')
 result.append({'path':str(p),'uid':f.stat().st_uid,'gid':f.stat().st_gid})
print(json.dumps(result))
""".replace('PARENTS', repr(parents))
        probe_name = 'miles-trtllm-probe-j' + self.c['job_id'] + ('-' + self.c['attempt_tag'] if self.c.get('attempt_tag') else '')
        text = self.remote(['docker', 'run', '--rm', '--name', probe_name, '--user', '28644:30',
                            '--mount', f'type=bind,src={local}/run,dst=/run-output',
                            '--mount', f'type=bind,src={node_view(self.c["platform"], self.root)},dst=/durable-output',
                            self.c['image'], 'python3', '-B', '-c', code], label='container-writer-probe')
        probes = json.loads(text.strip().splitlines()[-1])
        for item in probes:
            require((item['uid'], item['gid']) == (28644, 30), 'Unexpected actual container writer identity')
            path = Path(item['path']); require(path.name.startswith('writer-probe-'), 'Unexpected probe path')
            if path.parent == Path('/durable-output'):
                actual = self.root / path.name
                require((actual / 'nested/ok').read_text() == 'ok' and (actual / 'nested/ok').stat().st_uid == 28644, 'NFS probe not owned/readable')
                (actual / 'nested/ok').unlink(); (actual / 'nested').rmdir(); actual.rmdir()
                node_path = node_view(self.c['platform'], actual)
                self.remote(['python3', '-c', 'from pathlib import Path;assert not Path(' + repr(node_path) + ').exists()'], label='shared-probe-removed')
            else:
                require(path.parent == Path('/run-output'), 'Probe escapes intended mount')
                node_path = local + '/run/' + path.name
                if self.c['platform'] == 'gb300':
                    # The CIFS service identity cannot write normal-user NFS
                    # directories. Main outputs use local storage; actual durable
                    # retention is SSH -> login NFS, owned by this normal UID.
                    target = self.root / ('retention-' + path.name)
                    self.guard()
                    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes',
                                    self.original['NodeList'] + ':' + node_path + '/nested/ok', str(target)],
                                   check=True, timeout=30)
                    require(target.read_text() == 'ok' and target.stat().st_uid == 28644,
                            'Actual node-local to login-NFS retention probe failed')
                    target.unlink()
                self.remote(['python3', '-c', 'from pathlib import Path;p=Path(' + repr(node_path) + ");f=p/'nested/ok';assert f.read_text()=='ok' and f.stat().st_uid==28644;f.unlink();(p/'nested').rmdir();p.rmdir();assert not p.exists()"], label='local-probe-removed')
        return {'status': 'PASS', 'writer_uid': 28644, 'writer_gid': 30, 'removed_probes': probes,
                'durable_route': 'SSH to login direct NFS' if self.c['platform'] == 'gb300' else 'container direct NFS'}

    def models(self, storage, expected):
        explicit = self.c.get('reuse_node_models')
        if explicit:
            models = self.remote_json(host_model_inventory, explicit, label='explicit-reuse-model-inventory')
            check_models(models, expected)
            return {'root': explicit, 'models': models, 'mode': 'explicit_reused_verified_readonly_local_inputs'}
        candidates = (['/raid/tmp/miles-kaixih-j2198810-profile/models'] if self.c['platform'] == 'gb300' else [])
        for candidate in candidates:
            try:
                models = self.remote_json(host_model_inventory, candidate, label='reuse-model-inventory')
                check_models(models, expected)
                return {'root': candidate, 'models': models, 'mode': 'reused_verified_readonly_local_inputs'}
            except (ValueError, subprocess.SubprocessError):
                pass
        destination = storage['local_root'] + '/models'
        for name in FINGERPRINTS:
            self.remote(['mkdir', '-p', destination + '/' + name], label='create-model-output')
            names = sorted(expected[name]['files'])
            source = node_view(self.c['platform'], BASE / 'models' / name) + '/'
            seconds = min(5400, int(timestamp(self.original['EndTime'] + 'Z') - time.time() - 4 * 3600))
            require(seconds > 60, 'Insufficient lease for staging plus the full training run')
            self.remote(['rsync', '-rt', '--no-perms', '--omit-dir-times', '--partial', '--files-from=-', source,
                         destination + '/' + name + '/'], input='\n'.join(names) + '\n', timeout=seconds, label='stage-' + name)
        models = self.remote_json(host_model_inventory, destination, label='staged-model-inventory')
        check_models(models, expected)
        return {'root': destination, 'models': models, 'mode': 'copied_canonical_inventory_to_local_storage'}

    def driver_config(self, storage, models):
        result = {'platform': self.c['platform'], 'run_id': self.c['run_id'], 'job_id': self.c['job_id'],
                'node': self.original['NodeList'], 'node_ip': storage['node_ip'], 'image': self.c['image'],
                'repo': self.c['source'], 'node_repo': node_view(self.c['platform'], self.c['source']),
                'models': str(BASE / 'models'), 'node_models': models['root'], 'run_dir': str(self.root),
                'node_run_dir': storage['local_root'] + '/run', 'cache_dir': storage['local_root'] + '/cache',
                'container_prefix': 'miles-' + self.c['platform'] + '-qwen3-trtllm-j' + self.c['job_id']
                                    + ('-' + self.c['attempt_tag'] if self.c.get('attempt_tag') else ''),
                'ray_port': 26379, 'dashboard_port': 28265,
                'megatron_path': '/root/Megatron-LM' if self.c['platform'] == 'gb300' else '/opt/Megatron-LM',
                'nccl_iface': storage['nic'], 'lease_deadline': self.original['EndTime'] + 'Z',
                'gate_log': str(self.root / 'single-node-allreduce.log'), 'source_commit': self.c['source_commit'],
                'train_sha256': DATA_HASHES['train.jsonl'], 'eval_sha256': DATA_HASHES['test-fixed-256.jsonl'],
                'source_manifest': str(self.root / 'source-manifest.json'), 'uid': 28644, 'gid': 30,
                'sglang_moe_runner_backend': 'flashinfer_trtllm', 'save_optimizer': False}
        return result

    def inspect_container(self, c):
        data = json.loads(self.remote(['docker', 'inspect', c['container_prefix'] + '-0'], label='owned-container-inspect'))[0]
        require(data['State']['Running'] and data['Config']['Image'] == c['image'], 'Wrong/stopped experiment container')
        require(data['Config']['User'] == '28644:30', 'Container normal UID mismatch')
        mounts = {m['Destination']: m for m in data['Mounts']}
        for dest, source, writable in [('/opt/miles', c['node_repo'], False), (c['models'], c['node_models'], False),
                                       ('/run-output', c['node_run_dir'], True), ('/cache', c['cache_dir'], True)]:
            require(mounts[dest]['Source'] == source and mounts[dest]['RW'] == writable, 'Container mount mismatch: ' + dest)
        require('/root' not in mounts, 'The GB300 /root traversal fix must only affect the container filesystem')
        return data

    def bootstrap(self, c):
        from importlib.util import module_from_spec, spec_from_file_location
        driver = Path(c['repo']) / 'lab/rubin_two_node/run_cudagraph_main.py'
        spec = spec_from_file_location('verified_main_driver', driver)
        module = module_from_spec(spec); spec.loader.exec_module(module)
        normalized = module._config(c)
        plan = module._plan(normalized)
        command = ['python3', '-B', str(Path(c['repo']) / 'lab/rubin_two_node/orchestrate_rubin.py'),
                   'bootstrap', *plan['common_orchestrator_args']]
        self.record('bootstrap-command.json', command)
        self.guard()
        log = self.attempt / 'bootstrap-driver.log'
        with log.open('x') as stream:
            p = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        self.record('bootstrap-driver-exit.json', {'returncode': p.returncode, 'ended_at': utc(), 'log': str(log)})
        require(p.returncode == 0, 'Bootstrap failed; preserve container and inspect before retry')
        evidence = self.read('bootstrap.json')
        module._bootstrap(normalized, evidence)
        self.inspect_container(c)
        return {'container': c['container_prefix'] + '-0', 'image_id': evidence['image_id'], 'state': evidence['state']}

    def preflight(self, c):
        container = c['container_prefix'] + '-0'
        self.inspect_container(c)
        if c['platform'] == 'gb300':
            self.remote(['docker', 'exec', '--user', '0', container, 'chmod', 'o+x', '/root'], label='owned-container-root-traversal')
        def call(argv, label, timeout=180):
            return self.remote(['docker', 'exec', '--user', '28644:30', container,
                                'timeout', '--kill-after=10s', str(timeout) + 's', *argv], timeout=timeout + 20, label=label)
        call(['python3', '-B', '-m', 'unittest', 'lab.rubin_two_node.test_qwen3_launcher'], 'launcher-tests')
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location('verified_main_preflight', Path(c['repo']) / 'lab/rubin_two_node/run_cudagraph_main.py')
        driver = module_from_spec(spec); spec.loader.exec_module(driver)
        launcher = ['python3', '-B', 'lab/rubin_two_node/run_qwen3_30b_a3b_gsm8k_rubin.py',
                    '--model-dir', c['models'], '--data-dir', c['models'], '--output-dir', '/run-output',
                    '--num-nodes', '1', '--num-rollout', '50', *driver._launcher_args(driver._config(c)), '--print-only']
        text = call(launcher, 'launcher-print-only')
        printed = json.loads(text[text.index('{'):]); actual = printed['argv']
        require(actual[actual.index('--sglang-moe-runner-backend') + 1] == 'flashinfer_trtllm', 'Actual launcher backend mismatch')
        require('--sglang-disable-cuda-graph' not in actual and '--sglang-disable-piecewise-cuda-graph' in actual,
                'Actual launcher graph configuration mismatch')
        require('--no-save-optim' in actual and printed['planned_optimizer_steps'] == 200, 'Actual launcher checkpoint/step recipe mismatch')
        self.record('preflight/launcher-argv.json', printed)
        code = """import torch,torch.distributed as d,os,time
assert torch.cuda.device_count()==4
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
d.init_process_group('nccl',device_id=torch.device('cuda',rank));assert d.get_world_size()==4
for size in [1<<20,64<<20,512<<20]:
 x=torch.empty(size//4,device='cuda');times=[]
 for repeat in range(5):
  x.fill_(d.get_rank()+1);d.barrier();t=time.monotonic();d.all_reduce(x);torch.cuda.synchronize();times.append(time.monotonic()-t)
  assert torch.all(x==10).item()
 if d.get_rank()==0:print(f'ALLREDUCE_OK world=4 bytes={size} value={x[0].item()} mean_ms={sum(times[1:])/4*1000}',flush=True)
d.destroy_process_group()
"""
        call(['python3', '-B', '-c', "from pathlib import Path;Path('/run-output/preflight/allreduce-verified.py').write_text(" + repr(code) + ')'], 'write-allreduce', 30)
        text = call(['python3', '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=4',
                     '--master-addr=127.0.0.1', '--master-port=29500', '/run-output/preflight/allreduce-verified.py'], 'allreduce', 240)
        require(text.count('ALLREDUCE_OK world=4') == 3, 'Four-rank allreduce verification incomplete')
        (self.root / 'single-node-allreduce.log').write_text(text)
        self.record('single-node-allreduce-summary.json', {'status': 'PASS', 'world': 4, 'cases': 3, 'utc': utc()})
        code = """import importlib.metadata as m,json,os,torch,hashlib
from pathlib import Path
versions={}
for n in ['torch','sglang','transformer-engine','flash-attn','flashinfer-python','triton','ray']:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
first=json.loads(Path('/opt/miles-moe/provenance.json').read_text())
last=json.loads(Path('/opt/miles-moe/followup-idempotent-postload/provenance.json').read_text())
assert first['pr_head']=='89f25a0b9772576fbd427a495a1144f544c90928'
assert first['native_payloads_unchanged'] and first['native_distributions_unchanged']
assert last['after_git_blob']=='0ff99553743c61711e28b0dfde2bfd52ddd563fb'
assert last['real_flashinfer_cpu_tests_passed']==4 and last['native_packages_and_binaries_unchanged']
assert (os.getuid(),os.getgid())==(28644,30)
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'versions':versions,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'pr33743':first,'postload':last,'data':{n:hashlib.sha256((Path('/run-output/inputs')/n).read_bytes()).hexdigest() for n in ['train.jsonl','test-fixed-256.jsonl']}}))
"""
        text = call(['python3', '-B', '-c', code], 'runtime-provenance', 90)
        runtime = json.loads(text.strip().splitlines()[-1])
        require(runtime['data'] == DATA_HASHES, 'Runtime dataset hashes differ')
        self.record('preflight/runtime-provenance.json', runtime)
        result = {'status': 'PASS', 'utc': utc(), 'model': 'Qwen3-30B-A3B', 'backend': 'flashinfer_trtllm',
                  'decode_graph_requested': True, 'prefill_graph_requested': False, 'gpu_generation_validated': False}
        self.record('preflight/complete.json', result)
        return result

    def prepare(self):
        require(timestamp(self.original['EndTime'] + 'Z') - time.time() > 4 * 3600,
                'Less than four hours remain for preparation and the full main run')
        manifest, proof = verify_source(self.c)
        prior_proof = self.read('source-verification.json')
        require(prior_proof is None or prior_proof == proof, 'Frozen source manifest changed across attempts')
        self.record('source-manifest.json', manifest); self.record('source-verification.json', proof)
        def verify_image(prior):
            require(self.image_receipt()['image_id'] == prior['image_id'], 'Local image identity changed')
        self.stage('image', lambda: self.image_receipt(), verify_image)
        host_config = {**self.c, 'node': self.original['NodeList']}
        storage = self.stage('storage', lambda: self.remote_json(host_storage, host_config, label='storage-prepare'),
                             lambda result: self.remote_json(host_storage, host_config, result, label='storage-reverify'))
        self.record('node-inventory.json', storage)
        self.stage('writer-probe', lambda: self.storage_probe(storage))
        expected = json.loads(Path(self.c['model_record']).read_text())['models']
        check_models(expected, expected)
        model = self.stage('models', lambda: self.models(storage, expected),
                           lambda r: check_models(self.remote_json(host_model_inventory, r['root'], label='models-reverify'), expected))
        data = {}
        for name, digest in DATA_HASHES.items():
            raw = (Path(self.c['data_root']) / name).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == digest, 'Prepared dataset mismatch: ' + name)
            target = self.root / 'inputs' / name
            if target.exists():
                require(target.read_bytes() == raw, 'Existing retained data differs')
            else:
                target.write_bytes(raw)
            data[name] = {'bytes': len(raw), 'sha256': digest}
        self.remote(['rsync', '-rt', '--no-perms', '--omit-dir-times', node_view(self.c['platform'], self.root / 'inputs') + '/',
                     storage['local_root'] + '/run/inputs/'], label='stage-data')
        c = self.driver_config(storage, model)
        existing = self.read('driver-config.json')
        require(existing is None or existing == c, 'Driver config changed across preparation attempts')
        self.record('driver-config.json', c)
        self.record('preparation.json', {'run_id': self.c['run_id'], 'source_commit': self.c['source_commit'],
                    'slurm': self.original, 'node': c['node'], 'image': c['image'], 'models': model['models'],
                    'model_staging': model['mode'], 'data': data, 'storage': storage, 'prepared_at': utc(),
                    'inputs_staged': True, 'background_bulk_io': False,
                    'checkpoint_retention': 'Final model weights only, node-local. Caller must copy and verify final checkpoint on durable storage before releasing the node; optimizer checkpoint state is disabled.'})
        if not self.read('prep-stages/bootstrap.json'):
            occupied = self.remote(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], label='gpu-occupancy')
            require(not occupied.strip(), 'Existing GPU processes must be inspected before bootstrap')
        def verify_bootstrap(prior):
            require(self.inspect_container(c)['Image'] == prior['image_id'], 'Owned container image ID changed')
        self.stage('bootstrap', lambda: self.bootstrap(c), verify_bootstrap)
        self.stage('preflight', lambda: self.preflight(c), lambda r: self.inspect_container(c))
        self.guard()
        final = {'state': 'READY_NOT_TRAINING', 'at': utc(), 'run_id': c['run_id'], 'node': c['node'],
                 'config': str(self.root / 'driver-config.json'), 'lease_deadline': c['lease_deadline']}
        self.record('prep-state.json', final); print(json.dumps(final), flush=True)
        if self.args.execute_main:
            return self.execute_main(c)
        return 0

    def image_receipt(self):
        data = json.loads(self.remote(['docker', 'image', 'inspect', self.c['image']], label='image-inspect'))[0]
        require(self.c['image'] in data.get('RepoDigests', []), 'Image digest is not present locally; pull externally')
        return {'image': self.c['image'], 'image_id': data['Id'], 'repo_digests': data['RepoDigests'], 'observed_at': utc()}

    def execute_main(self, c):
        self.guard()
        for name in ('main-worker-launch.json', 'train-launch.json', 'train_exit.json', 'train-driver-exit.json'):
            require(not (self.root / name).exists(), 'Existing main-run evidence: never resubmit automatically')
        command = ['python3', '-u', str(Path(c['repo']) / 'lab/rubin_two_node/run_cudagraph_main.py'),
                   '--config', str(self.root / 'driver-config.json'), '--execute']
        self.record('main-worker-launch.json', {'command': command, 'at': utc(), 'pid': os.getpid()})
        log = self.attempt / 'main-driver-console.log'
        with log.open('x') as stream:
            p = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        self.record('main-worker-exit.json', {'returncode': p.returncode, 'at': utc(), 'log': str(log),
                                             'completion_not_proven_by_exit_code': True})
        return p.returncode

    def execute(self):
        require((os.getuid(), os.getgid()) == (28644, 30), 'Execute on dl3 as normal UID 28644:GID 30')
        self.root.mkdir(parents=True, exist_ok=True)
        require(self.root.stat().st_uid == 28644 and not self.root.is_symlink(), 'Unexpected durable output owner/path')
        for part in [self.root, *self.root.parents]:
            require(not part.is_symlink(), 'Symlink in durable output path')
        for name in ('logs', 'inputs', 'preflight', 'prep-stages', 'prep-attempts'):
            (self.root / name).mkdir(exist_ok=True)
        with (self.root / 'prep-worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prior = self.read('prep-invocation.json')
            if prior:
                require(self.args.resume and prior == self.c, 'Use --resume with the unchanged original preparation arguments')
            else:
                require(not self.args.resume, 'No preparation invocation exists to resume')
                self.record('prep-invocation.json', self.c)
            for name in ('main-worker-launch.json', 'train-launch.json', 'train_exit.json', 'train-driver-exit.json'):
                require(not (self.root / name).exists(), 'Main run was already submitted; inspect rather than repeat preparation')
            self.attempt = self.root / 'prep-attempts' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid()))
            self.attempt.mkdir()
            self.record(str(self.attempt.relative_to(self.root) / 'worker-source.json'),
                        {'path': str(Path(__file__).resolve()), 'sha256': file_hash(Path(__file__))})
            with tempfile.TemporaryDirectory(prefix='login-probe-', dir=self.root) as tmp:
                p = Path(tmp) / 'ok'; p.write_text('ok')
                require(p.read_text() == 'ok' and p.stat().st_uid == 28644, 'Normal-user durable write/read probe failed')
                p.unlink()
            try:
                original = self.read('allocation.json')
                while True:
                    record = self.allocation()
                    decision = allocation_guard(self.c, record, time.time(), original)
                    if decision == 'ready':
                        self.original = original or record
                        if not original:
                            self.record('allocation.json', record)
                        break
                    require(time.time() < timestamp(self.c['wait_until']), 'Original queue wait deadline expired')
                    self.record('prep-state.json', {'state': 'WAITING_FOR_OWN_ALLOCATION', 'at': utc(),
                                'job_id': self.c['job_id'], 'slurm': record, 'no_node_access_while_pending': True})
                    time.sleep(min(60, max(1, timestamp(self.c['wait_until']) - time.time())))
                return self.prepare()
            except Exception as exc:
                failure = {'state': 'FAILED', 'at': utc(), 'error': type(exc).__name__ + ': ' + str(exc),
                           'attempt': str(self.attempt), 'partial_evidence_preserved': True}
                self.record(str(self.attempt.relative_to(self.root) / 'failure.json'), failure)
                self.record('prep-state.json', failure)
                raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=sorted(JOBS), required=True)
    p.add_argument('--job-id', help='Explicit existing allocation; omitted keeps the original campaign job')
    p.add_argument('--attempt-tag', help='Explicit fresh Rubin attempt, e.g. r2; never resumes an earlier run')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', help='Expected frozen manifest hash; required for a tagged retry')
    p.add_argument('--reuse-node-models', type=Path, help='Verify and reuse this prior job-local models directory; fail instead of copying on mismatch')
    p.add_argument('--image')
    p.add_argument('--model-record', type=Path, default=MODEL_RECORD)
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--wait-until', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--execute-main', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def main():
    args = parser().parse_args(); config = validate_options(args)
    if not args.execute:
        print(json.dumps({'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'config': config,
                          'execute_main_after_preflight': args.execute_main,
                          'recipe': {'rollouts': 50, 'updates': 200, 'moe_backend': 'flashinfer_trtllm',
                                     'decode_graph': True, 'prefill_graph': False, 'save_optimizer': False},
                          'capacity_gib': 400, 'retention': 'Caller copies/verifies final model weights before node release'}, indent=2))
        return 0
    return Worker(config, args).execute()


if __name__ == '__main__':
    raise SystemExit(main())
