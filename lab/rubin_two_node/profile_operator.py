#!/usr/bin/env python3
"""Experiment-specific profiling operator; PLAN ONLY unless --execute is explicit.

Copy this file to dl3 and run there as UID28644:GID30. It does not tune either
recipe or change either image. Actions: prepare -> inspect printed replay plan ->
submit --expected-checkpoint-id ID -> retain -> stop. Each action plans by default.
Prepare requires completed learning evidence and explicit permission to stop the
exact completed main container. Never use this to interrupt a main run.

A replay is 3 rollouts/12 updates, not 3 optimizer steps. Both platforms read the
same Rubin final checkpoint. Actual cross-version optimizer resume remains an
execution check. SGLang profiling is separately triggered on an actual replay
engine URL using the bounded payload printed by profile_qwen3_replay.py.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

BASE = '/home/scratch.kaixih_ent'
REPO = BASE + '/repo/miles-rubin-cu134'
RUBIN_ROOT = BASE + '/repro/miles-rubin-qwen3-gsm8k/20260915-j2198331'
CHECKPOINT = RUBIN_ROOT + '/checkpoints'
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED'}
RUNS = {
    'rubin': dict(node='vr-nvl72-ts2-l11-038-c15', ip='10.102.74.82', nic='mp0',
        job='2198331', main='miles-rubin-qwen3-j2198331-0', root=RUBIN_ROOT,
        main_id='raysubmit_vZnATxienP7Z3MSp', main_run='20260915-j2198331-qwen3-a1',
        lease='2026-09-16T06:09:18+00:00', megatron='/opt/Megatron-LM',
        image='gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c'),
    'gb300': dict(node='gb300-nvl-012-compute04', ip='10.85.212.9', nic='enP5p9s0',
        job='2198810', main='miles-gb300-qwen3-j2198810-0',
        root=BASE + '/repro/miles-gb300-qwen3-gsm8k/20260915-j2198810-nfs',
        main_id='raysubmit_W7k1mh2Y9Uh9wmmk', main_run='20260915-j2198810-qwen3-a1',
        lease='2026-09-16T07:09:02+00:00', megatron='/root/Megatron-LM',
        image='radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986'),
}
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
    c.update(platform=args.platform, run_id=args.run_id)
    c['name'] = 'miles-profile-' + args.run_id
    c['local'] = '/tmp/' + c['name']
    c['durable'] = c['root'] + '/profiles/' + args.run_id
    c['input'] = c['root'] + '/inputs' if args.platform == 'rubin' else '/tmp/miles-gb300-j2198810/run/inputs'
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
              (host_input(c, BASE + '/models'), BASE + '/models', True),
              (host_input(c, CHECKPOINT), '/profile-checkpoint', True),
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


def replay_command(c, seconds, execute=False, checkpoint_id=''):
    cmd = ['docker', 'exec', c['name'], 'python3', '/opt/miles/lab/rubin_two_node/profile_qwen3_replay.py',
           '--source-plan', '/profile-source/original-ray-job.json', '--checkpoint-root', '/profile-checkpoint',
           '--checkpoint-iteration', '49', '--output-dir', '/run-output/replay', '--run-id', c['run_id'],
           '--ray-address', c['dashboard'], '--megatron-path', c['megatron'],
           '--max-runtime-seconds', str(seconds), '--max-trace-gib', '10']
    return cmd + (['--expected-checkpoint-id', checkpoint_id, '--execute-run'] if execute else ['--print-only'])


def jobs(c, dashboard):
    return node_python(c, "import json,urllib.request; print(json.dumps(json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(" +
                       repr(dashboard + '/api/jobs/') + ",timeout=15))))", 25)


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


def check_main(c):
    r = Path(c['root'])
    for filename in ['train_exit.json', 'train-driver-exit.json']:
        if json.loads((r / filename).read_text()).get('exit_code') != 0:
            raise RuntimeError('Main launcher did not exit successfully: ' + filename)
    if c['platform'] == 'gb300':
        retained = json.loads((r / 'artifact-retention-exit.json').read_text())
        if retained.get('exit_code') != 0 or retained.get('uid') != 28644:
            raise RuntimeError('GB main evidence has not been retained by normal UID')
    spec = importlib.util.spec_from_file_location('qwen3_summary', REPO + '/lab/rubin_two_node/summarize_qwen3_runs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.summarize_run(c['platform'], r / 'logs/qwen3_train.log', {'status': 'SUCCEEDED'})
    if metrics['partial'] or metrics['completed_training_rollouts'] != list(range(50)):
        raise RuntimeError('Main log does not prove all 50 rollouts/200 updates')
    current = jobs(c, 'http://' + c['ip'] + ':28265')
    if any(j.get('type') == 'SUBMISSION' and j.get('status') not in TERMINAL for j in current):
        raise RuntimeError('Main Ray cluster still has an active/unknown job')
    matches = [j for j in current if j.get('submission_id') == c['main_id']]
    if len(matches) != 1 or matches[0].get('status') != 'SUCCEEDED':
        raise RuntimeError('Exact main Ray job is not SUCCEEDED')
    main = matches[0]
    if main.get('runtime_env', {}).get('env_vars', {}).get('RUBIN_RUN_ID') != c['main_run']:
        raise RuntimeError('Main Ray identity mismatch')
    original = json.loads((r / 'original-ray-job.json').read_text())
    recorded_env = original.get('runtime_env', {}).get('env_vars', {})
    unexpected_routing = {k: recorded_env[k] for k in ('RAY_ADDRESS', 'RAY_API_SERVER_ADDRESS', 'MILES_SCRIPT_EXTERNAL_RAY') if k in recorded_env}
    if unexpected_routing:
        raise RuntimeError('Review recorded Ray routing before replay; will not rewrite it: ' + repr(unexpected_routing))
    if main.get('entrypoint') != original.get('entrypoint'):
        raise RuntimeError('Main entrypoint differs from saved source plan')
    tracker = Path(CHECKPOINT, 'latest_checkpointed_iteration.txt').read_text().strip()
    if tracker != '49':
        raise RuntimeError('Shared Rubin checkpoint is not final iteration49')
    return {'checked_at': utc(), 'ray_job': main, 'log_sha256': metrics['source_log_sha256'],
            'completed_rollouts': metrics['completed_training_rollouts'], 'optimizer_updates': 200}


def inspect_profile(c):
    data = json.loads(remote(c, ['docker', 'inspect', c['name']]))[0]
    if data['Config']['Labels'].get('miles.profile_run') != c['run_id'] or data['Config']['Image'] != c['image']:
        raise RuntimeError('Profile container identity/image mismatch')
    if data['Config']['User'] != '28644:30':
        raise RuntimeError('Unexpected profile writer UID')
    return data


def prepare(c, args):
    allocation(c)
    seconds = budget(c, args.max_runtime_seconds, args.retention_margin_seconds)
    evidence = check_main(c)
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
    check_main(c)
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
        'nofile_soft_hard': [65535, 65535], 'input_hashes': INPUT_HASHES,
        'source_commit': run(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip(),
        'source_plan_sha256': hashlib.sha256(Path(c['source_plan']).read_bytes()).hexdigest()})
    print(plan_text)


def submit(c, args):
    allocation(c)
    inspect_profile(c)
    if not re.fullmatch('[0-9a-f]{64}', args.expected_checkpoint_id):
        raise RuntimeError('Supply the reviewed 64-hex checkpoint ID; reuse that exact ID for the other platform')
    if not Path(c['durable'], 'operator-prepared.json').is_file():
        raise RuntimeError('Missing prepare record')
    seconds = budget(c, args.max_runtime_seconds, args.retention_margin_seconds)
    command = replay_command(c, seconds, True, args.expected_checkpoint_id)
    text = remote(c, command, 120)
    (Path(c['durable']) / 'submit-result.txt').write_text(text)
    print(text)


def profile_terminal(c):
    inspect_profile(c)
    current = jobs(c, c['dashboard'])
    if any(j.get('type') == 'SUBMISSION' and j.get('status') not in TERMINAL for j in current):
        raise RuntimeError('Profile cluster has an active/unknown job; do not retain/stop yet')
    found = [j for j in current if j.get('submission_id') == c['submission_id']]
    if len(found) != 1 or found[0]['runtime_env']['env_vars'].get('MILES_PROFILE_RUN_ID') != c['run_id']:
        raise RuntimeError('Exact terminal profile identity not found')
    return found[0]


def manifest(c):
    code = """import hashlib,json,os,pathlib
root=pathlib.Path(ROOT); result={}
def fail(e): raise e
for base,dirs,files in os.walk(root,onerror=fail,followlinks=False):
 for name in dirs+files:
  p=pathlib.Path(base,name)
  if p.is_symlink(): raise RuntimeError('Refuse symlink: '+str(p))
 for name in files:
  p=pathlib.Path(base,name); h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
  result[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(result))
""".replace('ROOT', repr(c['local'] + '/run'))
    return node_python(c, code, 600)


def retain(c, args):
    job = profile_terminal(c)
    # Retain actual driver evidence, not only the profiler files. This executes
    # only after exact terminal identity was checked; normal container UID writes.
    capture = """import json,pathlib,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
url=URL
with opener.open(url+'/logs',timeout=30) as response: logs=json.load(response)
p=pathlib.Path('/run-output/replay')
(p/'job-driver.log').write_text(logs['logs'])
(p/'final-ray-job.json').write_text(json.dumps(JOB,indent=2)+'\\n')
print(json.dumps({'driver_log_bytes':(p/'job-driver.log').stat().st_size}))
""".replace('URL', repr(c['dashboard'] + '/api/jobs/' + c['submission_id'])).replace('JOB', repr(job))
    remote(c, ['docker', 'exec', c['name'], 'python3', '-c', capture], 45)
    destination = Path(c['durable'], 'node-output')
    destination.mkdir(exist_ok=True)
    before = manifest(c)
    total = sum(v['bytes'] for v in before.values())
    if shutil.disk_usage(destination).free < total + 2*1024**3:
        raise RuntimeError('Insufficient durable retention capacity')
    # Invoked ON dl3 as normal UID; never a GB CIFS destination, no 100MiB filter.
    subprocess.run(['rsync', '-rt', '--no-perms', '--omit-dir-times', '--partial',
                    c['node'] + ':' + c['local'] + '/run/', str(destination) + '/'], check=True,
                   timeout=args.retention_margin_seconds)
    after = manifest(c)
    if before != after:
        raise RuntimeError('Output changed while retaining; repeat retain after guard finishes')
    for name, meta in after.items():
        p = destination / name
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2), b''): h.update(block)
        if p.stat().st_uid != 28644 or p.stat().st_size != meta['bytes'] or h.hexdigest() != meta['sha256']:
            raise RuntimeError('Retained file mismatch: ' + name)
    write_json(Path(c['durable'], 'retention.json'), {'retained_at': utc(), 'uid': os.getuid(),
        'run_id': c['run_id'], 'terminal_ray_job': job, 'bytes': total, 'files': after,
        'trace_limit_is_polled_not_hard': True})


def stop(c):
    job = profile_terminal(c)
    record = json.loads(Path(c['durable'], 'retention.json').read_text())
    if record['run_id'] != c['run_id'] or record['terminal_ray_job']['submission_id'] != job['submission_id']:
        raise RuntimeError('Retention record identity mismatch')
    if manifest(c) != record['files']:
        raise RuntimeError('Artifacts changed since retention; retain again before stop')
    # Exact container namespace only. Do not run ray stop/pkill on the host.
    remote(c, ['docker', 'exec', c['name'], 'ray', 'stop', '--force'], 60)
    remote(c, ['docker', 'stop', '--time', '30', c['name']], 60)
    write_json(Path(c['durable'], 'operator-stopped.json'), {'stopped_at': utc(), 'container': c['name']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('platform', choices=RUNS)
    p.add_argument('--action', choices=['prepare', 'submit', 'retain', 'stop'], default='prepare')
    p.add_argument('--run-id', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--allow-stop-completed-main', action='store_true')
    p.add_argument('--expected-checkpoint-id', default='')
    p.add_argument('--max-runtime-seconds', type=int, default=4500)
    p.add_argument('--retention-margin-seconds', type=int, default=1200)
    args = p.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', args.run_id):
        p.error('Use a simple unique run ID of at most64 characters')
    if args.max_runtime_seconds < 900 or args.retention_margin_seconds < 1200:
        p.error('Require >=900s requested runtime and >=1200s retention reserve')
    c = config(args)
    if not args.execute:
        print(json.dumps({'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'action': args.action, 'config': c,
            'completion_gates': ['exact main Ray SUCCEEDED', 'both launcher exits0', '50rollouts/200updates',
                                 'GB main evidence retained', 'shared Rubin tracker49', 'no active Ray submissions'],
            'docker_argv': docker_command(c), 'ray_start_argv': ray_command(c),
            'replay_plan_argv': replay_command(c, args.max_runtime_seconds),
            'runtime_budget': 'min(requested, recorded lease remaining - retention margin), recomputed before submit',
            'retention_margin_seconds': args.retention_margin_seconds,
            'caveats': ['Both versioned runtimes read the same checkpoint; cross-version optimizer resume untested.',
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
    else: stop(c)


if __name__ == '__main__':
    main()
