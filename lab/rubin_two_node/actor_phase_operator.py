#!/usr/bin/env python3
"""Scoped actor diagnostic host operator. No actions without --execute.

Each phase owns a fresh container, cache, output directory and absolute guard.
The model/Miles source and prepared inputs are mounted read-only. Run on dl3 as
UID28644:GID30. This does not submit or repeat the completed learning runs.
"""
import argparse
import datetime as dt
import hashlib
import inspect
import math
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import stat
import time
import urllib.request


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            if deadline is not None:
                remaining(deadline)
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def remote(c, argv, timeout=60):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                   c['node'], shlex.join(argv)], text=True, timeout=timeout)


def node_python(c, code, timeout=60):
    # Bound the remote child too, not only the local SSH client.
    return json.loads(remote(c, ['timeout', '--signal=TERM', '--kill-after=5', str(max(1, int(timeout)-5)),
                                 'python3', '-c', code], timeout).strip().splitlines()[-1])


def remaining(deadline, maximum=60):
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError('Original phase/lease budget exhausted')
    return min(maximum, seconds)


def config_sha(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pinned_phase(c):
    record = json.loads((Path(c['durable_phase']) / 'phase-plan.json').read_text())
    if record['config'] != c or record.get('config_sha256') != config_sha(c):
        raise ValueError('Original phase config changed')
    deadline = record['absolute_deadline']
    lease = dt.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00')).timestamp()
    if not math.isfinite(deadline) or deadline > lease - 300:
        raise ValueError('Original phase deadline invalid')
    return record


def proc_identity(pid):
    root = Path('/proc') / str(pid)
    raw = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(raw[19]), 'process_state': raw[0],
            'uid': root.stat().st_uid, 'argv': (root / 'cmdline').read_bytes().decode().split('\0')[:-1]}


def phase_guard_snapshot(c, identity, deadline):
    code = ('import json,os;from pathlib import Path\n' + inspect.getsource(proc_identity)
            + '\ns=json.loads(Path(' + repr(c['node_phase'] + '/guard-state.json') + ').read_text());'
            + 'print(json.dumps({"state":s,"process":proc_identity(s["pid"])}))')
    snapshot = node_python(c, code, timeout=remaining(deadline, 20))
    state, process = snapshot['state'], snapshot['process']
    expected = identity['guard']
    if (state.get('state') != 'armed' or state.get('container_id') != identity['container_id']
            or state.get('deadline') != deadline or state.get('config_sha256') != config_sha(c)
            or state.get('pid') != expected['pid'] or state.get('start_ticks') != expected['start_ticks']
            or process['pid'] != expected['pid'] or process['start_ticks'] != expected['start_ticks']
            or process['uid'] != 28644 or process['process_state'] in ('Z', 'X')
            or process['argv'] != ['python3', '-u', c['node_phase'] + '/guard.py']
            or not 0 <= time.time() - state.get('heartbeat_epoch', 0) <= 20):
        raise ValueError('Independent guard identity/heartbeat/config mismatch')
    if deadline - time.time() < 300:
        raise ValueError('Insufficient original phase execution budget')
    return state


def allocation(c, minimum=300):
    raw = subprocess.check_output(['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o',
                                   str(c['job_id'])], text=True, timeout=20)
    f = dict(x.split('=', 1) for x in raw.split() if '=' in x)
    if (f.get('JobId') != str(c['job_id']) or f.get('JobState') != 'RUNNING'
            or f.get('NodeList') != c['node'] or not f.get('UserId', '').endswith('(28644)')):
        raise ValueError('Exact allocation node/UID/state mismatch')
    end = dt.datetime.fromisoformat(f['EndTime']).replace(tzinfo=dt.timezone.utc).timestamp()
    expected = dt.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00')).timestamp()
    if abs(end - expected) > 1 or end - time.time() < minimum:
        raise ValueError('Original lease changed or insufficient time remains')
    return {'at': utc(), 'raw': raw, 'lease_timestamp': end}


def mounts(c):
    result = [(c['node_repo'], '/opt/miles', False),
              (c['node_models'], c['models'], False),
              (c['node_inputs'], '/inputs', False),
              (c['node_helpers'], '/opt/actor-profile', False),
              (c['node_output'], '/run-output', True),
              (c['node_cache'], '/cache', True)]
    for p in ['/dev/nvidia-caps', '/dev/nvidia-caps-imex-channels']:
        result.append((p, p, True))
    return result


def validate_container(c, info, running=True, exact_id=None):
    if (exact_id is not None and info.get('Id') != exact_id):
        raise ValueError('Exact container ID mismatch')
    if (info['Name'] != '/' + c['container'] or info['Image'] != c['image_id']
            or info['Config']['Image'] != c['image'] or info['Config']['User'] != '28644:30'
            or info['Config'].get('Labels', {}).get('miles.actor_phase') != c['phase_id']
            or (running and not info['State']['Running'])):
        raise ValueError('Exact phase container identity mismatch')
    actual = {m['Destination']: m for m in info['Mounts']}
    for source, target, writable in mounts(c):
        item = actual.get(target, {})
        if item.get('Type') != 'bind' or item.get('Source') != source or item.get('RW') != writable:
            raise ValueError('Phase mount mismatch: ' + target)
    if set(actual) != {target for _, target, _ in mounts(c)}:
        raise ValueError('Unexpected phase mounts forbidden')


def docker_argv(c):
    argv = ['docker', 'run', '--detach', '--name', c['container'], '--label',
            'miles.actor_phase=' + c['phase_id'], '--user', '28644:30', '--gpus', 'all',
            '--network', 'host', '--ipc', 'host', '--privileged', '--ulimit', 'memlock=-1',
            '--ulimit', 'stack=67108864', '--ulimit', 'nofile=65535:65535', '--shm-size', '16g',
            '--workdir', '/opt/miles']
    for source, target, writable in mounts(c):
        argv += ['--mount', 'type=bind,src=' + source + ',dst=' + target + ('' if writable else ',readonly')]
    env = {'HOME': '/cache/home', 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': '/opt/actor-profile:/opt/miles:' + c['megatron_path'],
           'MILES_SCRIPT_EXTERNAL_RAY': '1', 'RAY_USAGE_STATS_ENABLED': '0', 'RAY_DEDUP_LOGS': '0',
           'RAY_ADDRESS': c['node_ip'] + ':26379', 'RAY_API_SERVER_ADDRESS': c['dashboard'],
           'MASTER_ADDR': c['node_ip'], 'NCCL_SOCKET_IFNAME': c['nccl_iface'],
           'GLOO_SOCKET_IFNAME': c['nccl_iface'], 'NCCL_CUMEM_ENABLE': '1', 'NCCL_NVLS_ENABLE': '0',
           'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'MAX_JOBS': '8', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'no_proxy': 'localhost,127.0.0.1,' + c['node_ip'],
           'NO_PROXY': 'localhost,127.0.0.1,' + c['node_ip']}
    for key, leaf in {'TMPDIR': 'tmp', 'RAY_TMPDIR': 'ray', 'TRITON_CACHE_DIR': 'triton',
                      'FLASHINFER_WORKSPACE_BASE': 'flashinfer', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
                      'CUDA_CACHE_PATH': 'cuda', 'HF_HOME': 'huggingface',
                      'PYTHONPYCACHEPREFIX': 'pycache', 'XDG_CACHE_HOME': 'xdg'}.items():
        env[key] = '/cache/' + leaf
    for key, value in env.items():
        argv += ['--env', key + '=' + value]
    return argv + [c['image'], 'sleep', 'infinity']


def output_size(path):
    total = 0
    for folder, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            try:
                item = (Path(folder) / name).lstat()
            except FileNotFoundError:
                continue  # Atomic trace/receipt rename is normal.
            if stat.S_ISLNK(item.st_mode):
                raise ValueError('Output symlink')
            if stat.S_ISREG(item.st_mode):
                total += item.st_size
    return total


def guard(c, container_id, deadline):
    """Detached guard; retries transient reads and only stops its immutable ID."""
    root = Path(c['node_phase'])
    state = root / 'guard-state.json'
    process = proc_identity(os.getpid())
    def save(value):
        data = {'at': utc(), 'heartbeat_epoch': time.time(), 'container_id': container_id,
                'deadline': deadline, 'config_sha256': config_sha(c), 'pid': os.getpid(),
                'start_ticks': process['start_ticks'], **value}
        try:
            tmp = state.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2) + '\n')
            tmp.replace(state)
        except OSError as error:
            print('guard state write failed: ' + repr(error), flush=True)
    errors = 0
    while True:
        reason, size, mem = None, None, {}
        try:
            info = json.loads(subprocess.check_output(['docker', 'inspect', container_id], text=True, timeout=8))[0]
            validate_container(c, info, running=False, exact_id=container_id)
            if not info['State']['Running']:
                save({'state': 'container_stopped'})
                return
            size = output_size(c['node_output'])
            mem = {line.split(':')[0]: int(line.split()[1]) * 1024
                   for line in Path('/proc/meminfo').read_text().splitlines()}
            reason = ('absolute_deadline' if time.time() >= deadline else
                      'output_size' if size > c['max_output_bytes'] else
                      'host_memory' if mem['MemAvailable'] < 0.12 * mem['MemTotal'] else None)
            errors = 0
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.SubprocessError) as error:
            errors += 1
            save({'state': 'degraded', 'read_errors': errors, 'error': repr(error)})
            if errors >= 3 or time.time() >= deadline:
                reason = 'guard_read_failure'
        if reason:
            # This exact Docker ID was validated before guard launch. Docker IDs
            # cannot be reassigned; never substitute a name after inspect errors.
            try:
                subprocess.run(['docker', 'stop', '--time', '15', container_id], check=True, timeout=25)
                save({'state': 'stopped_by_guard', 'reason': reason, 'output_bytes': size,
                      'host_available_bytes': mem.get('MemAvailable')})
                return
            except (OSError, subprocess.SubprocessError) as error:
                save({'state': 'stop_retry', 'reason': reason, 'error': repr(error)})
        elif not errors:
            save({'state': 'armed', 'output_bytes': size, 'host_available_bytes': mem['MemAvailable']})
        time.sleep(2 if errors or reason else 5)


def start_guard(c, container_id, deadline):
    body = '\n\n'.join(inspect.getsource(f) for f in [utc, config_sha, proc_identity, mounts, validate_container, output_size, guard])
    code = ('import datetime as dt,hashlib,json,os,stat,subprocess,time\nfrom pathlib import Path\n' + body
            + '\nguard(' + repr(c) + ',' + repr(container_id) + ',' + repr(deadline) + ')\n')
    script = c['node_phase'] + '/guard.py'
    launch = ('import json,subprocess;from pathlib import Path;'
              'p=Path(' + repr(script) + ');p.open("x").write(' + repr(code) + ');'
              'f=open(' + repr(c['node_phase'] + '/guard.log') + ',"x");'
              'p=subprocess.Popen(["python3","-u",str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True);'
              'print(json.dumps({"pid":p.pid}))')
    result = node_python(c, launch)
    for _ in range(20):
        try:
            state = json.loads(remote(c, ['cat', c['node_phase'] + '/guard-state.json']))
            if (state['container_id'] == container_id and state['deadline'] == deadline
                    and state['state'] == 'armed' and state['pid'] == result['pid']
                    and state['config_sha256'] == config_sha(c)):
                return {**result, **state}
        except (subprocess.SubprocessError, ValueError, KeyError):
            pass
        time.sleep(.2)
    raise RuntimeError('Independent phase guard not armed')


def manifest_code(path):
    return ('import hashlib,json;from pathlib import Path;r=Path(' + repr(path) + ');d={};'
            '\nfor p in sorted(r.rglob("*")):\n'
            ' if p.is_symlink(): raise ValueError("Output symlink")\n'
            ' if p.is_file():\n'
            '  h=hashlib.sha256()\n'
            '  with p.open("rb") as f:\n'
            '   for b in iter(lambda:f.read(1048576),b""):h.update(b)\n'
            '  d[str(p.relative_to(r))]={"bytes":p.stat().st_size,"sha256":h.hexdigest()}\n'
            'print(json.dumps(d))')


def bootstrap(c):
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError('Host operator requires normal dl3 UID28644:GID30')
    alloc = allocation(c, minimum=1200)
    root = Path(c['durable_phase'])
    root.mkdir()
    deadline = min(time.time() + c['max_runtime_seconds'], alloc['lease_timestamp'] - 300)
    write(root / 'phase-plan.json', {'at': utc(), 'config': c, 'allocation': alloc,
                                    'absolute_deadline': deadline, 'config_sha256': config_sha(c), 'operator_sha256': sha(__file__)})
    prepare = ('from pathlib import Path;import json,os,socket;'
               'assert(os.getuid(),os.getgid())==(28644,30);'
               'r=Path(' + repr(c['node_phase']) + ');r.mkdir();'
               '[Path(p).mkdir() for p in ' + repr([c['node_output'], c['node_cache']]) + '];'
               '[Path(' + repr(c['node_cache']) + ',p).mkdir() for p in '
               + repr(['home','tmp','ray','triton','flashinfer','torch_extensions','cuda','huggingface','pycache','xdg']) + '];'
               '\nfor port in [26379,26380,26381,26382,28265]:\n'
               ' with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(("0.0.0.0",port));s.listen(1)\n'
               'print(json.dumps({"ready":True}))')
    node_python(c, prepare)
    container_id = remote(c, docker_argv(c)).strip()
    if not re.fullmatch('[0-9a-f]{64}', container_id):
        raise ValueError('Docker did not return an exact container ID')
    info = json.loads(remote(c, ['docker', 'inspect', container_id]))[0]
    validate_container(c, info, exact_id=container_id)
    armed = start_guard(c, container_id, deadline)
    write(root / 'container-identity.json', {'at': utc(), 'container_id': container_id,
                                           'inspect': info, 'guard': armed})
    probe = ('import os,json;from pathlib import Path;assert(os.getuid(),os.getgid())==(28644,30);'
             'p=Path("/run-output/writer-probe");p.write_text("28644");'
             'assert p.stat().st_uid==28644;p.unlink();print(json.dumps({"writer_ok":True}))')
    remote(c, ['docker', 'exec', container_id, 'python3', '-c', probe])
    if c['platform'] == 'gb300':
        current = json.loads(remote(c, ['docker', 'inspect', container_id]))[0]
        validate_container(c, current)
        # Only make the image's existing editable installation traversable.
        code = ('import os,stat,json;fd=os.open("/root",os.O_DIRECTORY|os.O_NOFOLLOW);s=os.fstat(fd);'
                'assert s.st_uid==0;os.fchmod(fd,stat.S_IMODE(s.st_mode)|1);'
                'print(json.dumps({"before":oct(stat.S_IMODE(s.st_mode)),"after":oct(stat.S_IMODE(os.fstat(fd).st_mode))}));os.close(fd)')
        result = remote(c, ['docker', 'exec', '--user', '0', container_id, 'python3', '-c', code])
        write(root / 'gb-root-traversal.json', {'at': utc(), 'container_id': container_id, 'result': json.loads(result)})
    import_probe = ('import os,json;assert(os.getuid(),os.getgid())==(28644,30);'
                    'import megatron.core;from megatron.core import parallel_state;'
                    'print(json.dumps({"uid":os.getuid(),"gid":os.getgid(),"megatron_core":megatron.core.__file__}))')
    imported = json.loads(remote(c, ['docker', 'exec', '--user', '28644:30', container_id,
                                    'python3', '-c', import_probe], timeout=remaining(deadline, 90)))
    write(root / 'normal-user-import.json', imported)
    phase_guard_snapshot(c, {'container_id': container_id, 'guard': armed}, deadline)
    ray = ['docker', 'exec', container_id, 'ray', 'start', '--head', '--node-ip-address', c['node_ip'],
           '--num-gpus', '4', '--num-cpus', '32', '--disable-usage-stats', '--port', '26379',
           '--node-manager-port', '26380', '--object-manager-port', '26381',
           '--runtime-env-agent-port', '26382', '--min-worker-port', '26400', '--max-worker-port', '26999',
           '--dashboard-host', c['node_ip'], '--dashboard-port', '28265', '--temp-dir', '/cache/ray']
    output = remote(c, ray, timeout=120)
    (root / 'ray-start.log').write_text(output)
    write(root / 'ready.json', {'at': utc(), 'container_id': container_id, 'deadline': deadline,
                               'state': 'RAY_READY_NO_SUBMISSION', 'config_sha256': config_sha(c),
                               'normal_uid_megatron_import': imported})
    return {'container_id': container_id, 'deadline': deadline}


def api(c, method, path, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(c['dashboard'] + '/api/jobs/' + path, data=body, method=method,
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def submission_ready(c, identity, deadline):
    info = json.loads(remote(c, ['docker', 'inspect', identity['container_id']], timeout=remaining(deadline, 20)))[0]
    validate_container(c, info, exact_id=identity['container_id'])
    return phase_guard_snapshot(c, identity, deadline)


def submit(c, plan_path):
    allocation(c)
    root = Path(c['durable_phase'])
    pinned = pinned_phase(c)
    deadline = pinned['absolute_deadline']
    identity = json.loads((root / 'container-identity.json').read_text())
    ready = json.loads((root / 'ready.json').read_text())
    if (ready.get('container_id') != identity['container_id'] or ready.get('deadline') != deadline
            or ready.get('config_sha256') != config_sha(c) or ready.get('state') != 'RAY_READY_NO_SUBMISSION'
            or ready.get('normal_uid_megatron_import', {}).get('uid') != 28644
            or ready.get('normal_uid_megatron_import', {}).get('gid') != 30):
        raise ValueError('Original bootstrap readiness/import mismatch')
    submission_ready(c, identity, deadline)
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw)
    argv = shlex.split(plan['entrypoint'])
    runtime = plan.get('runtime_env', {})
    env = runtime.get('env_vars', {})
    if (plan['submission_id'] != c['submission_id'] or argv[:2] != ['python3', '/opt/miles/train.py']
            or shlex.join(argv) != plan['entrypoint'] or set(runtime) != {'env_vars'}
            or env.get('RUBIN_RUN_ID') != c['phase_id']
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())):
        raise ValueError('Reviewed phase submission identity mismatch')
    jobs = api(c, 'GET', '', timeout=remaining(deadline, 30))
    if any(j.get('status') not in ['SUCCEEDED', 'FAILED', 'STOPPED'] for j in jobs):
        raise ValueError('Another Ray job is active')
    if any(j.get('submission_id') == c['submission_id'] for j in jobs):
        raise ValueError('Phase submission ID already exists')
    write(root / 'reviewed-submission.json', {'at': utc(), 'plan_sha256': hashlib.sha256(raw).hexdigest(), 'plan': plan})
    # Revalidate after API calls, immediately before the only submission.
    pinned_phase(c)
    allocation(c)
    submission_ready(c, identity, deadline)
    result = api(c, 'POST', '', {k: plan[k] for k in ['submission_id', 'entrypoint', 'runtime_env']},
                 timeout=remaining(deadline, 30))
    write(root / 'submitted.json', {'at': utc(), 'response': result})
    return result


def retain_guard_evidence(c, root, label, deadline):
    code = ('import base64,json;from pathlib import Path;r=Path(' + repr(c['node_phase']) + ');out={}'
            + '\nfor name in ["guard.py","guard-state.json","guard.log"]:\n'
            + ' p=r/name\n if p.is_symlink():raise ValueError("Guard evidence symlink")\n'
            + ' if p.exists():\n'
            + '  if p.stat().st_size>16*1024**2:raise ValueError("Guard evidence too large")\n'
            + '  out[name]=base64.b64encode(p.read_bytes()).decode()\n'
            + ' else:out[name]=None\nprint(json.dumps(out))')
    files = node_python(c, code, timeout=remaining(deadline, 30))
    write(root / ('guard-evidence-' + label + '.json'), {'at': utc(), 'node': c['node'],
          'source_root': c['node_phase'], 'encoding': 'base64', 'files': files})


def verify_retention_destination(target):
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError('Retention requires normal dl3 UID28644:GID30')
    item = target.lstat()
    if not stat.S_ISDIR(item.st_mode) or (item.st_uid, item.st_gid) != (28644, 30):
        raise ValueError('Retention destination must be a normal-UID owned directory')


def stop_retain(c):
    alloc = allocation(c, minimum=90)
    root = Path(c['durable_phase'])
    pinned_phase(c)
    # One nonrenewable budget includes both manifests, transfer, hashes and audit.
    deadline = min(time.time() + 900, alloc['lease_timestamp'] - 60)
    identity = json.loads((root / 'container-identity.json').read_text())
    exact = identity['container_id']
    try:
        retain_guard_evidence(c, root, 'before', deadline)
        info = json.loads(remote(c, ['docker', 'inspect', exact], timeout=remaining(deadline, 20)))[0]
        validate_container(c, info, running=False, exact_id=exact)
        if info['State']['Running']:
            jobs = api(c, 'GET', '', timeout=remaining(deadline, 30))
            if any(j.get('status') not in ['SUCCEEDED', 'FAILED', 'STOPPED'] for j in jobs):
                raise ValueError('Refuse stop while a Ray phase is active')
            write(root / 'final-ray-jobs.json', jobs)
            if any(j.get('submission_id') == c['submission_id'] for j in jobs):
                logs = api(c, 'GET', c['submission_id'] + '/logs', timeout=remaining(deadline, 30))
                (root / 'job-driver.log').write_text(logs['logs'])
            # Repeat immediately before exact-container stop; never submit Ray stop.
            info = json.loads(remote(c, ['docker', 'inspect', exact], timeout=remaining(deadline, 20)))[0]
            validate_container(c, info, running=False, exact_id=exact)
            jobs = api(c, 'GET', '', timeout=remaining(deadline, 30))
            if any(j.get('status') not in ['SUCCEEDED', 'FAILED', 'STOPPED'] for j in jobs):
                raise ValueError('A Ray job became active before stop')
            remote(c, ['docker', 'stop', '--time', '15', exact], timeout=remaining(deadline, 40))
        stopped = json.loads(remote(c, ['docker', 'inspect', exact], timeout=remaining(deadline, 20)))[0]
        validate_container(c, stopped, running=False, exact_id=exact)
        if stopped['State']['Running']:
            raise ValueError('Exact container is not stopped')
        write(root / 'stopped.json', {'at': utc(), 'container_id': exact})
        before = node_python(c, manifest_code(c['node_output']), timeout=remaining(deadline, 180))
        if sum(x['bytes'] for x in before.values()) > c['max_output_bytes'] * 2:
            raise ValueError('Retention exceeds phase output bound')
        target = root / 'node-output'
        target.mkdir()
        verify_retention_destination(target)
        transfer_seconds = remaining(deadline, 600)
        remote_seconds = max(1, int(transfer_seconds) - 5)
        subprocess.run(['rsync', '-rt', '--no-perms', '--no-owner', '--no-group', '--partial',
                        '--timeout=30', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                        '--rsync-path=timeout --signal=TERM --kill-after=5s ' + str(remote_seconds) + 's rsync',
                        c['node'] + ':' + c['node_output'].rstrip('/') + '/', str(target) + '/'],
                       check=True, timeout=transfer_seconds)
        after = node_python(c, manifest_code(c['node_output']), timeout=remaining(deadline, 180))
        if before != after:
            raise ValueError('Stopped phase output changed during retention')
        for name, item in before.items():
            path = target / name
            if path.is_symlink() or path.stat().st_size != item['bytes'] or sha(path, deadline) != item['sha256']:
                raise ValueError('Retained file mismatch: ' + name)
        remaining(deadline)
        write(root / 'retention.json', {'at': utc(), 'container_id': exact, 'files': before,
              'retention_deadline': deadline, 'stopped_before_retention': True, 'source_destination_verified': True})
        return {'retained_files': len(before), 'retained_bytes': sum(v['bytes'] for v in before.values())}
    finally:
        # Includes already-guard-stopped containers and failed/partial transfers.
        try:
            retain_guard_evidence(c, root, 'after', deadline)
        except Exception as error:
            write(root / 'guard-evidence-after-error.json', {'at': utc(), 'error': repr(error),
                  'retention_deadline': deadline, 'before_evidence_preserved': (root / 'guard-evidence-before.json').exists()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['bootstrap', 'submit', 'status', 'stop-retain'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--plan', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if not a.execute:
        print(json.dumps({'action': a.action, 'config': c, 'docker_argv': docker_argv(c)}, indent=2))
        return
    if a.action == 'bootstrap': result = bootstrap(c)
    elif a.action == 'submit': result = submit(c, a.plan)
    elif a.action == 'status':
        allocation(c)
        pinned_phase(c)
        result = api(c, 'GET', '')
    else: result = stop_retain(c)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
