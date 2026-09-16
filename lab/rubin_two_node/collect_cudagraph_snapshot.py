#!/usr/bin/env python3
"""Read-only, transactional snapshots of explicitly configured CUDA-graph main runs.

Config JSON maps rubin/gb300 to driver-config paths (null means pending). The only
remote writes are none: stdout transports bounded evidence to this local output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import inspect
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_qwen3_snapshot import (actual_recipe_and_save_events, scalar_flag, sha,
                                   decode_packet, collection_lock, commit_files)
from summarize_qwen3_runs import summarize_run, report, json_safe

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT = WORKSPACE / 'outputs/rubin-gb300-qwen3-cudagraph'
LOG = 'logs/qwen3_train.log'
FILES = ('driver-config.json', 'preparation.json', 'source-manifest.json', 'train-launch.json',
         'train_exit.json', 'train-driver-exit.json', 'watchdog.json', 'ray-job.json',
         'preflight/runtime-provenance.json', 'preflight/complete.json', 'single-node-allreduce-summary.json')
LIMIT = 1024**3


def encoded(value):
    return (json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n').encode()


def validate_config(platform, c):
    from pathlib import PurePosixPath
    import ipaddress
    ipaddress.ip_address(c['node_ip'])
    if platform not in ('rubin', 'gb300') or c.get('platform') != platform:
        raise ValueError('Unknown or mismatched platform')
    if not re.fullmatch(r'20260916-' + platform + r'-j\d+-cg', c.get('run_id', '')):
        raise ValueError('Run ID is outside the new graph experiment')
    if not str(c.get('job_id', '')).isdigit() or '-j' + str(c['job_id']) + '-cg' not in c['run_id']:
        raise ValueError('Job and run ID differ')
    if not re.fullmatch(r'[a-zA-Z0-9.-]+', c.get('node', '')):
        raise ValueError('Unsafe node')
    expected = '/home/scratch.kaixih_ent/repro/miles-qwen3-cudagraph/' + c['run_id']
    if c.get('run_dir') != expected:
        raise ValueError('Durable root must exactly match this run')
    p = PurePosixPath(c['node_run_dir'])
    if not p.is_absolute() or '..' in p.parts or str(p) != c['node_run_dir'] or ('j' + str(c['job_id'])) not in str(p):
        raise ValueError('Unsafe or unbound host watchdog path')
    if not str(p).startswith(('/tmp/', '/raid/')) or p.name != 'run':
        raise ValueError('Host output root must be a job-specific run directory')
    if not 1024 <= int(c['dashboard_port']) <= 65535:
        raise ValueError('Invalid Ray API port')
    if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', c.get('image', '')):
        raise ValueError('Image must be pinned by digest')
    if dt.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('Lease must have an explicit timezone')


def allocation_guard(c, text, now):
    """No node access unless this exact normal-user allocation is still live."""
    import datetime, re
    values = dict(re.findall(r'(\w+)=([^\s]+)', text))
    expected = {'JobId': str(c['job_id']), 'JobState': 'RUNNING', 'NodeList': c['node'], 'NumNodes': '1'}
    for key, value in expected.items():
        if values.get(key) != value:
            return False, 'allocation_' + key + '_mismatch'
    if not re.fullmatch(r'[^()]+\(28644\)', values.get('UserId', '')):
        return False, 'allocation_owner_mismatch'
    try:
        end = datetime.datetime.fromisoformat(values['EndTime']).replace(tzinfo=datetime.timezone.utc)
        lease = datetime.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00'))
    except (KeyError, ValueError):
        return False, 'allocation_deadline_unreadable'
    if end != lease or (end - now).total_seconds() < 35:
        return False, 'allocation_deadline_mismatch_or_expired'
    return True, 'verified_exact_allocation'


def node_read(c):
    import datetime, json, os, pathlib, socket, subprocess, urllib.request
    now = datetime.datetime.now(datetime.timezone.utc)
    if (datetime.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00')) - now).total_seconds() < 20:
        raise ValueError('Lease expired before node access')
    if socket.gethostname().split('.')[0] != c['node'].split('.')[0]:
        raise ValueError('Host identity mismatch')
    interfaces = json.loads(subprocess.check_output(['ip', '-j', 'address'], timeout=3))
    if c['node_ip'] not in {a.get('local') for iface in interfaces for a in iface.get('addr_info', [])}:
        raise ValueError('Ray API address is not a local interface on the verified host')
    path = pathlib.Path(c['node_run_dir']) / 'watchdog.json'
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError('Symlink in exact watchdog host path')
    watchdog = None
    if path.exists():
        if path.stat().st_uid != 28644 or path.stat().st_size > 8 * 1024**2:
            raise ValueError('Watchdog owner/size mismatch')
        watchdog = json.loads(path.read_bytes())
        if watchdog.get('run_id') != c['run_id']:
            raise ValueError('Watchdog run identity mismatch')
    result = {'observed_at': now.isoformat(), 'host': c['node'], 'watchdog_host_path': str(path),
              'watchdog': watchdog, 'ray': None, 'ray_error': None}
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://' + c['node_ip'] + ':' + str(c['dashboard_port']) + '/api/jobs/', timeout=8) as response:
            raw = response.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise ValueError('Ray response exceeds limit')
        matches = []
        for job in json.loads(raw):
            env = job.get('runtime_env') or {}
            if isinstance(env, str):
                env = json.loads(env)
            run_id = (env.get('env_vars') or {}).get('RUBIN_RUN_ID')
            if watchdog and job.get('submission_id') == watchdog.get('submission_id') and run_id != c['run_id']:
                raise ValueError('Ray submission has a different run identity')
            if run_id == c['run_id']:
                matches.append({k: job.get(k) for k in ('status', 'entrypoint', 'submission_id', 'job_id',
                                                       'driver_exit_code', 'start_time', 'end_time')})
        if len(matches) > 1:
            raise ValueError('More than one Ray job for the exact run ID')
        if matches:
            result['ray'] = dict(matches[0], run_id=c['run_id'], observed_at=now.isoformat())
    except (OSError, TimeoutError) as exc:
        result['ray_error'] = type(exc).__name__
    return result


def remote_read(c, prefix, files, node_source):
    import base64, datetime, hashlib, json, os, pathlib, shlex, signal, subprocess, zlib
    signal.alarm(80)
    def digest(raw):
        return hashlib.sha256(raw).hexdigest()
    def packet(raw):
        return {'bytes': len(raw), 'sha256': digest(raw), 'data': base64.b64encode(zlib.compress(raw, 3)).decode()}
    root = pathlib.Path(c['run_dir'])
    def safe(path):
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('Symlink in retained evidence path')
    records = {}
    for name in files:
        path = root / name; safe(path)
        if path.exists():
            if not path.is_file() or path.stat().st_size > 8 * 1024**2:
                raise ValueError('Auxiliary file exceeds limit')
            records[name] = packet(path.read_bytes())
    path = root / 'logs/qwen3_train.log'; safe(path)
    old_size, old_hash = prefix['bytes'], prefix['sha256']
    offset, full, tail, inode = 0, hashlib.sha256(), bytearray(), None
    if path.exists():
        with path.open('rb') as stream:
            stat = os.fstat(stream.fileno()); size, inode = stat.st_size, stat.st_ino
            if not 0 <= size <= 1024**3:
                raise ValueError('Log exceeds limit')
            if old_size <= size:
                while offset < old_size:
                    chunk = stream.read(min(1024**2, old_size - offset))
                    if not chunk: raise ValueError('Truncated prefix')
                    full.update(chunk); offset += len(chunk)
            if offset != old_size or full.hexdigest() != old_hash:
                stream.seek(0); offset, full = 0, hashlib.sha256()
            while offset + len(tail) < size:
                chunk = stream.read(min(1024**2, size - offset - len(tail)))
                if not chunk: raise ValueError('Truncated log')
                full.update(chunk); tail.extend(chunk)
            if path.stat().st_ino != inode or os.fstat(stream.fileno()).st_size < size:
                raise ValueError('Rotated log')
    else:
        size = 0
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    live, allocation = None, {'allowed': False, 'reason': 'allocation_query_failed'}
    try:
        command = ['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o', str(c['job_id'])]
        slurm = subprocess.check_output(command, timeout=10, stderr=subprocess.DEVNULL).decode()
        allowed, reason = allocation_guard(c, slurm, datetime.datetime.now(datetime.timezone.utc))
        allocation = {'allowed': allowed, 'reason': reason, 'slurm': slurm.strip(), 'observed_at': observed}
        if allowed:
            code = node_source + '\nprint(json.dumps(node_read(' + repr(c) + ')))'
            raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', c['node'],
                                           shlex.join(['python3', '-c', 'import json\n' + code])], timeout=25)
            live = json.loads(raw)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        allocation['read_error'] = type(exc).__name__
    return {'run_id': c['run_id'], 'root': str(root), 'observed_at': observed, 'files': records,
            'log': {'bytes': size, 'sha256': full.hexdigest(), 'offset': offset,
                    'prefix_sha256': old_hash if offset else digest(b''), 'tail': packet(tail), 'inode': inode},
            'allocation': allocation, 'live': live}


def decode(c, payload, previous):
    if payload.get('run_id') != c['run_id'] or payload.get('root') != c['run_dir']:
        raise ValueError('Payload identity mismatch')
    log = payload['log']; offset = log['offset']
    if offset not in (0, len(previous)) or log['prefix_sha256'] != sha(previous[:offset]):
        raise ValueError('Prefix mismatch')
    raw = previous[:offset] + decode_packet(log['tail'], LIMIT)
    if len(raw) != log['bytes'] or len(raw) > LIMIT or sha(raw) != log['sha256']:
        raise ValueError('Full log digest mismatch')
    if set(payload['files']) - set(FILES):
        raise ValueError('Unexpected auxiliary path')
    return {LOG: raw, **{n: decode_packet(v, 8 * 1024**2) for n, v in payload['files'].items()}}


def metadata(platform, c, payload, files):
    parsed = {n: json.loads(raw) for n, raw in files.items() if n.endswith('.json')}
    retained = parsed.get('driver-config.json')
    if retained is not None and retained != c:
        raise ValueError('Retained driver configuration differs from the supplied configuration')
    prep, launch = parsed.get('preparation.json', {}), parsed.get('train-launch.json', {})
    if prep and prep.get('run_id') != c['run_id']:
        raise ValueError('Preparation run mismatch')
    if launch and (launch.get('git_commit') != c['source_commit'] or
                   launch.get('source_manifest_sha256') != sha(files.get('source-manifest.json', b''))):
        raise ValueError('Launch source provenance mismatch')
    live = payload.get('live') or {}
    watch = live.get('watchdog') or parsed.get('watchdog.json')
    ray = live.get('ray') or parsed.get('ray-job.json')
    for name, value in [('watchdog', watch), ('ray-job', ray)]:
        if value and value.get('run_id') != c['run_id']:
            raise ValueError(name + ' identity mismatch')
    log = files[LOG].decode(errors='replace')
    entries = re.findall(r'^Running entrypoint for job (\S+): (.+)$', log, re.M)
    if len(entries) > 1:
        raise ValueError('Multiple actual entrypoints')
    entry = ray.get('entrypoint') if ray else None
    submission = ray.get('submission_id') if ray else (watch or {}).get('submission_id')
    if entries:
        if submission and entries[0][0] != submission:
            raise ValueError('Entrypoint and live job identities differ')
        if entry and shlex.split(entry) != shlex.split(entries[0][1]):
            raise ValueError('Actual Ray and logged entrypoints differ')
        submission, entry = entries[0]
    if watch and watch.get('submission_id') and submission != watch['submission_id']:
        raise ValueError('Watchdog and Ray submission differ')
    argv = shlex.split(entry) if entry else []
    if any(x.split('=')[0] == '--use-pytorch-profiler' for x in argv):
        raise ValueError('Unexpected profiler in this main run')
    recipe, saves, exclusions = actual_recipe_and_save_events(argv, log) if argv else ({}, {}, [0])
    total = scalar_flag(argv, '--num-rollout', integer=True) if argv else None
    steps = scalar_flag(argv, '--num-steps-per-rollout', integer=True) if argv else None
    if argv:
        def flag(name, default=None): return scalar_flag(argv, name, default=default)
        recipe.update(model=Path(flag('--hf-checkpoint', 'UNKNOWN')).name,
                      dataset_reward=('GSM8K / strict #### scorer' if flag('--custom-rm-path') == 'lab.rubin_two_node.gsm8k_verl_reward.reward_func' else 'GSM8K / ' + str(flag('--custom-rm-path', 'UNKNOWN'))),
                      reward_function=flag('--custom-rm-path'),
                      batch_description=f"{flag('--rollout-batch-size')} prompts × {flag('--n-samples-per-prompt')} responses; {steps} updates",
                      length_description=f"{flag('--rollout-max-response-len')} response / {flag('--rollout-max-prompt-len')} prompt tokens",
                      sampling_description=f"T={flag('--rollout-temperature')}, top-p={flag('--rollout-top-p')}, top-k={flag('--rollout-top-k')}",
                      learning_rate=flag('--lr'), global_batch_size=flag('--global-batch-size'),
                      group_filter=flag('--dynamic-sampling-filter-path'),
                      prompt_data=flag('--prompt-data'), seed=flag('--seed', 'source default'),
                      rollout_seed=flag('--rollout-seed', 'source default'))
    runtime = parsed.get('preflight/runtime-provenance.json', {})
    terminal = {'SUCCEEDED', 'FAILED', 'STOPPED'}
    status = (live.get('ray') or {}).get('status')
    if not status:
        status = (watch or {}).get('terminal_status') or ((ray or {}).get('status') if (ray or {}).get('status') in terminal else None)
    status = status or ('PENDING' if not launch else 'UNKNOWN')
    for name, value in [('watchdog.json', watch), ('ray-job.json', ray)]:
        if value: files[name] = encoded(value)
    return {'run_id': c['run_id'], 'display_name': 'Rubin ES' if platform == 'rubin' else 'GB300',
            'status': status, 'submission_id': submission, 'gpus': 4, 'expected_rollouts': total or 50,
            'optimizer_steps_per_rollout': steps or 4, 'profiling_coverage_known': bool(argv),
            'profiled_rollouts': [], 'exclude_timing_rollouts': exclusions, 'checkpoint_save_evidence': saves,
            'image': c['image'], 'versions': runtime.get('versions', {}), 'hardware': {'name': runtime.get('device', 'UNKNOWN'), 'engineering_sample': platform == 'rubin'},
            'actual_ray_entrypoint_argv': argv, 'actual_recipe_available': bool(argv), 'recipe': recipe,
            'graph': {'decode_requested': bool(argv) and '--sglang-disable-cuda-graph' not in argv,
                      'prefill_requested': bool(argv) and '--sglang-disable-piecewise-cuda-graph' not in argv,
                      'replay_verified': False,
                      'capture_log_samples': [line[-1200:] for line in log.split('\n') if 'Capture target decode CUDA graph' in line][:12],
                      'decode_log_sample_counts': {value: len(re.findall(r'Decode batch[^\n]*cuda graph: ' + value, log)) for value in ('True', 'False')},
                      'sample_count_scope': 'Scheduler logging samples only, not total forwards or a fallback rate.',
                      'verification': 'Actual argv records intent; scheduler samples/trace required for replay proof.'},
            'kernels': {k: scalar_flag(argv, flag) for k, flag in [('sglang_attention', '--sglang-attention-backend'), ('sglang_moe', '--sglang-moe-runner-backend'), ('sglang_bf16_gemm', '--sglang-bf16-gemm-backend'), ('megatron_attention_policy', '--attention-backend')]},
            'git_commit_at_launch': launch.get('git_commit'), 'source_sha256_at_launch': launch.get('source_sha256'),
            'started_utc': launch.get('started_utc'), 'source_remote_root': c['run_dir'], 'log_timestamps_are_utc': True,
            'driver_exits': {name: parsed.get(name) for name in ('train_exit.json', 'train-driver-exit.json')},
            'startup_gc_error_samples': [line[-1000:] for line in log.split('\n') if 'freeze_gc' in line and any(t in line for t in ('Error', 'error', 'failed'))][:6],
            'input_preparation': {k: prep.get(k) for k in ('prepared_at', 'inputs_staged', 'background_bulk_io', 'data')},
            'status_evidence': {'observed_at': payload['observed_at'], 'live_ray_observed': bool(live.get('ray')), 'allocation': payload['allocation'],
                                'watchdog_host_path': live.get('watchdog_host_path'), 'ray_error': live.get('ray_error'), 'ray_observed_at': (ray or {}).get('observed_at'),
                                'limitation': 'RUNNING is lifecycle status, not a claim that training is healthy; retained evidence may be stale.'}}


def collect_one(platform, c, output):
    old = output / platform / LOG
    previous = old.read_bytes() if old.exists() else b''
    if len(previous) > LIMIT: raise ValueError('Local log exceeds limit')
    source = inspect.getsource(allocation_guard) + '\n' + inspect.getsource(remote_read)
    code = 'import json\n' + source + '\nprint(json.dumps(remote_read(' + repr(c) + ',' + repr({'bytes': len(previous), 'sha256': sha(previous)}) + ',' + repr(FILES) + ',' + repr(inspect.getsource(node_read)) + ')))'
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'dl3',
                                   shlex.join(['python3', '-c', code])], timeout=90)
    payload = json.loads(raw); files = decode(c, payload, previous)
    for name in ('watchdog.json', 'ray-job.json'):
        prior = output / platform / name
        if name not in files and prior.exists(): files[name] = prior.read_bytes()
    meta = metadata(platform, c, payload, files)
    receipt = {k: v for k, v in payload.items() if k not in ('files', 'log', 'live')}
    receipt.update(log={k: v for k, v in payload['log'].items() if k != 'tail'}, transferred_log_bytes=payload['log']['tail']['bytes'],
                   watchdog_source={'host': c['node'], 'path': c['node_run_dir'] + '/watchdog.json', 'method': 'host_ssh_after_allocation_guard', 'sha256': sha(files['watchdog.json']) if 'watchdog.json' in files else None},
                   driver_config_sha256=sha(encoded(c)), collector_sha256=sha(Path(__file__).read_bytes()),
                   helper_sha256={name: sha(Path(__file__).with_name(name).read_bytes()) for name in ('collect_qwen3_snapshot.py', 'summarize_qwen3_runs.py')})
    files['collection-receipt.json'] = encoded(receipt)
    return platform, meta, files


def completion_validation(run, raw):
    """Observed completion checks, not independent reward regrading or weight audit."""
    m=run['metadata']; total=m['expected_rollouts']; steps=m['optimizer_steps_per_rollout']
    events=[e for row in run['rows'] for e in row['train_steps']]
    positive=lambda v: type(v) in (int,float) and math.isfinite(v) and v>0
    finite=lambda v: type(v) in (int,float) and math.isfinite(v)
    outcomes=re.findall(r'actor_cell0_rank(\d+)\].*?op=train_step rollout=(\d+) step=(\d+) attempt=(\d+) outcome=(\w+) valid_step=(\w+)', raw.decode(errors='replace'))
    normal={(int(rank),int(r),int(s)) for rank,r,s,a,o,v in outcomes if o=='NORMAL' and v=='true'}
    expected={(rank,r,s) for rank in range(4) for r in range(total) for s in range(steps)}
    checks={'ray_succeeded':m['status']=='SUCCEEDED', 'all_rollouts_complete':run['completed_training_rollouts']==list(range(total)),
            'all_optimizer_ids':{e['logged_id'] for e in events}==set(range(total*steps)) and len(events)==total*steps,
            'all_gradients_finite_positive':bool(events) and all(positive(e['metrics'].get('train/grad_norm')) for e in events),
            'all_losses_finite':bool(events) and all(finite(e['metrics'].get('train/loss')) for e in events),
            'all_rank_steps_normal':normal==expected and all(o=='NORMAL' and v=='true' for _,_,_,_,o,v in outcomes),
            'driver_exits_zero':all((m.get('driver_exits',{}).get(n) or {}).get('exit_code')==0 for n in ('train_exit.json','train-driver-exit.json')),
            'no_metric_errors':not run['parse_errors'] and not run['conflicts']}
    return {'status':'PASS' if all(checks.values()) else 'PENDING' if m['status'] not in ('SUCCEEDED','FAILED','STOPPED') else 'NOT_PROVEN',
            'checks':checks,'optimizer_events_observed':len(events),'normal_rank_steps_observed':len(normal),
            'expected_optimizer_events':total*steps,'expected_normal_rank_steps':len(expected),
            'scope':'Logged completion, finite positive gradients, finite losses and NORMAL rank outcomes; no independent reward regrading or weight-change proof.'}


def collect(config_path, output, fetch=collect_one):
    output = Path(output).resolve()
    protected = [WORKSPACE / 'outputs/rubin-gb300-qwen3', WORKSPACE / 'reports/rubin-gb300-qwen3']
    if any(output == old or old in output.parents or output in old.parents for old in protected):
        raise ValueError('Refusing an old or broad output root')
    mapping = json.loads(Path(config_path).read_bytes())
    if not isinstance(mapping, dict) or set(mapping) - {'rubin', 'gb300'}:
        raise ValueError('Config must map platform to driver-config path/null')
    configs = {}
    for platform, path in mapping.items():
        if path is not None:
            path = Path(path); path = path if path.is_absolute() else Path(config_path).parent / path
            configs[platform] = json.loads(path.read_bytes()); validate_config(platform, configs[platform])
    output.mkdir(parents=True, exist_ok=True)
    with collection_lock(output):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: fetch(*item, output), configs.items()))
        with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=output) as temporary:
            stage = Path(temporary); paths, runs, specs, health = [], [], [], []
            def write(name, raw):
                final = output / name
                if any(p.is_symlink() for p in [final, *final.parents] if p != output.parent):
                    raise ValueError('Symlink in local output path')
                target = stage / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw); paths.append(name)
            for platform, meta, files in results:
                for name, raw in files.items(): write(platform + '/' + name, raw)
                write(platform + '/metadata.json', encoded(meta))
                run = summarize_run(platform, stage / platform / LOG, meta)
                run['source_log'] = str(output / platform / LOG); runs.append(run)
                specs.append({'label': platform, 'log': run['source_log'], 'metadata': meta})
                health.append({'label': platform, 'run_id': meta['run_id'], 'status': meta['status'],
                               'run_health': 'terminal' if meta['status'] in ('SUCCEEDED', 'FAILED', 'STOPPED') else 'lifecycle_observed',
                               'issue': None if meta['status'] not in ('FAILED', 'STOPPED') else 'Ray ' + meta['status'] + '; inspect retained raw evidence.',
                               'completed_rollouts': len(run['completed_training_rollouts']),
                               'optimizer_updates_observed': sum(r['optimizer_steps_observed'] for r in run['rows']),
                               'completion_validation': completion_validation(run, files[LOG])})
            comparison = report(runs); comparison['pending_platforms'] = sorted({'rubin', 'gb300'} - set(configs))
            comparison['experiment_id'] = 'qwen3-cudagraph-nightly-v1'
            raw = encoded(comparison)
            write('runs.json', encoded(specs)); write('health.json', encoded({'schema': 'miles-run-health-v1', 'comparison_sha256': sha(raw), 'snapshot_at': comparison['collected_at'], 'runs': health}))
            write('comparison.json', raw)  # Commit last; no previous evidence changes before every source validates.
            commit_files(output, stage, paths)
    return {'comparison_sha256': sha(raw), 'collected_at': comparison['collected_at'], 'runs': health, 'pending_platforms': comparison['pending_platforms']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, default=DEFAULT)
    a = p.parse_args(); print(json.dumps(collect(a.config, a.output), indent=2))


if __name__ == '__main__':
    main()
