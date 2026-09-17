#!/usr/bin/env python3
"""TRTLLM ON-only host capture after verified 50-rollout/200-update completion.

Run on dl3. Reuses the existing allocation, container identity, evidence
retention and deadline guards. Never resumes or restarts main training.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rubin_two_node'))
import run_cudagraph_diagnostic_operator as base


def docker_command(c):
    argv = base.docker_command(c)
    return argv[:-3] + ['--mount', f'type=bind,src={c["node_repo"]},dst=/opt/miles,readonly',
                       '--mount', f'type=bind,src={c["node_base"]}/source/frozen-token-input.json,dst=/profile-inputs/frozen-token-input.json,readonly',
                       *argv[-3:]]


def identity(c, info, image_id, diagnostic=False, require_running=True):
    result = base.container_identity(c, info, image_id, diagnostic, require_running)
    if diagnostic:
        mounts = {m['Destination']: m for m in info['Mounts']}
        for target, source in [('/opt/miles', c['node_repo']),
                               ('/profile-inputs/frozen-token-input.json', c['node_base'] + '/source/frozen-token-input.json')]:
            m = mounts.get(target, {})
            if m.get('Source') != source or m.get('RW') is not False or m.get('Type') != 'bind':
                raise ValueError('Unverified source/frozen-input mount: ' + target)
        requests = info['HostConfig'].get('DeviceRequests', [])
        if not any(r.get('DeviceIDs') == ['0'] for r in requests):
            raise ValueError('Require one explicit GPU0 Docker request')
    return result


def execute(c, port, frozen, frozen_sha):
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError('Run on dl3 as UID28644:GID30')
    root = Path(c['durable'])
    if root.exists() or any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError('Fresh diagnostic destination required')
    if frozen.is_symlink() or base.sha(frozen) != frozen_sha:
        raise ValueError('Frozen shared input changed')
    source_manifest = json.loads(Path(c['source_manifest']).read_text())['source_sha256']
    executed_sources = ['lab/rubin_trtllm_full/profiling/run_profile.py',
                        'lab/rubin_two_node/run_sglang_graph_diagnostic.py',
                        'lab/rubin_two_node/sglang_graph_capture.py']
    source_hashes = {name: base.sha(Path(c['repo']) / name) for name in executed_sources}
    if any(source_hashes[name] != source_manifest.get(name) for name in executed_sources):
        raise ValueError('Executed profiling sources differ from frozen manifest')
    def alloc():
        return base.allocation(c, base.run(['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o', str(c['job_id'])]), time.time())
    def inspected(name):
        return json.loads(base.remote(c, ['docker', 'inspect', name]))[0]
    api = base.JobsAPI(c['dashboard'])
    driver = json.loads(Path(c['run_dir'], 'train-driver-exit.json').read_text())
    train = json.loads(Path(c['run_dir'], 'train_exit.json').read_text())
    first = alloc(); actual = base.node(c, 'read'); jobs = api.list_jobs()
    verify_code = "from pathlib import Path;import hashlib;root=Path(" + repr(c['node_repo']) + ");expected=" + repr(source_hashes) + ";assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in expected.items())"
    base.remote(c, ['python3', '-B', '-c', verify_code])
    main_job = base.completion(c, actual['watchdog'], jobs, driver, train)
    main_info = inspected(c['container'])
    image_id = json.loads(base.remote(c, ['docker', 'image', 'inspect', c['image']]))[0]['Id']
    c['image_id'] = image_id
    selected = identity(c, main_info, image_id)
    plan = json.loads(Path(c['run_dir'], 'cudagraph-main-plan.json').read_text())
    if (plan.get('run_id') != c['run_id'] or plan.get('git_commit') != c['source_commit']
            or plan.get('preflight', {}).get('container_id') != main_info['Id']
            or plan['preflight'].get('image_id') != image_id
            or plan['recipe'].get('sglang_moe_runner_backend') != 'flashinfer_trtllm'):
        raise ValueError('Main provenance/backend differs')
    root.mkdir(parents=True)
    probe = root / '.writer-probe'; probe.write_text(c['diagnostic_run'])
    assert probe.stat().st_uid == 28644 and probe.read_text() == c['diagnostic_run']; probe.unlink()
    base.write(root / 'operator-plan.json', {'config': c, 'allocation': first, 'main_container': selected,
               'ray_job': main_job, 'order': ['on'], 'frozen_input_sha256': frozen_sha})
    base.write(root / 'main-driver-evidence.json', {'driver_exit': driver, 'train_exit': train, 'original_main_plan': plan})
    login_files = base.retain_login_main(c, plan, root / 'main-login')
    summary = base.metrics.summarize_run(c['platform'], root / 'main-login/logs/qwen3_train.log', {'status': 'SUCCEEDED'})
    base.completion(c, actual['watchdog'], jobs, driver, train, summary)
    base.write(root / 'main-login-retention.json', {'at': base.utc(), 'files': login_files,
                'stable_source_and_destination_sha256_verified': True})
    base.node(c, 'prepare')
    manifest = base.node(c, 'snapshot', phase='main-before')
    base.retain(c, 'main-before', manifest, root / 'main-before')
    base.write(root / 'main-retention.json', {'at': base.utc(), 'files': manifest,
               'source_prefix_and_destination_sha256_verified': True, 'login_files': login_files,
               'completed_rollouts': summary['completed_training_rollouts'], 'optimizer_updates': 200})
    alloc(); repeated = base.node(c, 'read'); base.verify_login_sources(login_files)
    base.completion(c, repeated['watchdog'], api.list_jobs(), driver, train, summary)
    fresh = inspected(c['container']); identity(c, fresh, image_id)
    if fresh['Id'] != main_info['Id']:
        raise ValueError('Main container replaced')
    base.remote(c, ['docker', 'stop', '--time', '15', fresh['Id']], 40)
    base.write(root / 'main-stopped.json', {'at': base.utc(), 'container_id': fresh['Id'], 'retention_verified_before_stop': True})
    final = base.node(c, 'snapshot', phase='main-final')
    base.retain(c, 'main-final', final, root / 'main-final'); base.write(root / 'main-final-retention.json', final)
    paths = [Path(base.__file__).with_name(name) for name in base.SOURCES] + [frozen]
    names = [p.name for p in paths[:-1]] + ['frozen-token-input.json']
    hashes = dict(zip(names, [base.sha(p) for p in paths]))
    for p, name in zip(paths, names):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(p), c['node'] + ':' + c['node_base'] + '/source/' + name], check=True, timeout=30)
    alloc()
    diagnostic_id = base.remote(c, docker_command(c)).strip()
    if not re.fullmatch(r'[0-9a-f]{64}', diagnostic_id):
        raise ValueError('Docker did not return a new exact container ID')
    result = None
    try:
        info = inspected(diagnostic_id); identity(c, info, image_id, True)
        base.write(root / 'docker-inspect.json', info)
        deadline = min(time.time() + 1800, c['lease_timestamp'] - 120)
        ident = {'run_id': c['diagnostic_run'], 'main_run_id': c['run_id'],
            'container_name': c['diagnostic_name'], 'container_id': diagnostic_id,
            'image_reference': c['image'], 'image_id': image_id, 'uid': 28644, 'gid': 30,
            'labels': {'miles.graph_diagnostic': c['diagnostic_run']}, 'docker_inspect_verified_at': base.utc(),
            'main_terminal_verified': True, 'source_commits': {'miles_main': c['source_commit'], 'sglang': base.capture.SGLANG_COMMIT},
            'diagnostic_source_sha256': hashes, 'executed_source_sha256': source_hashes,
            'main_submission_id': main_job['submission_id']}
        ident = base.node(c, 'identity', hashes=hashes, identity=ident)
        base.write(root / 'diagnostic-identity.json', ident)
        probe = "from pathlib import Path;import os;assert(os.getuid(),os.getgid())==(28644,30);p=Path('/run-output/.host-writer-probe');assert p.read_text()==" + repr(c['diagnostic_run']) + ";p.unlink();q=Path('/run-output/.container-writer-probe');q.write_text('container28644');assert q.stat().st_uid==28644"
        base.remote(c, ['docker', 'exec', diagnostic_id, 'python3', '-c', probe])
        base.remote(c, ['python3', '-c', "from pathlib import Path;p=Path(" + repr(c['node_base'] + '/run/.container-writer-probe') + ");assert p.read_text()=='container28644' and p.stat().st_uid==28644;p.unlink()"])
        guard = base.node(c, 'guard', guard_source=base.guard_source(c, diagnostic_id, deadline), deadline=deadline)
        for _ in range(20):
            try:
                armed = json.loads(base.remote(c, ['cat', c['node_base'] + '/guard-state.json'], 10)); break
            except subprocess.CalledProcessError:
                time.sleep(.2)
        else:
            raise RuntimeError('Host deadline guard failed to arm')
        if armed != {'state': 'armed', 'container_id': diagnostic_id, 'deadline': deadline}:
            raise ValueError('Guard identity mismatch')
        guard_check = "import os;from pathlib import Path;p=" + repr(guard['pid']) + ";os.kill(p,0);assert int(Path('/proc',str(p),'stat').read_text().rpartition(') ')[2].split()[19])==" + repr(guard['start_ticks'])
        base.remote(c, ['python3', '-c', guard_check])
        base.write(root / 'guard-armed.json', {**guard, **armed})
        base.prepare_gb_editable_sources(c, diagnostic_id, {**guard, **armed})
        argv = ['docker', 'exec', diagnostic_id, 'python3', '-u', '/opt/miles/lab/rubin_trtllm_full/profiling/run_profile.py', 'run',
                '--identity-file', '/run-output/diagnostic-identity.json', '--image', c['image'], '--main-run-id', c['run_id'],
                '--model-path', '/models/Qwen3-30B-A3B', '--dataset', '/inputs/train.jsonl',
                '--frozen-token-input', '/profile-inputs/frozen-token-input.json', '--frozen-input-sha256', frozen_sha,
                '--output-dir', '/run-output/profile', '--cache-dir', '/cache/sglang-trtllm-profile',
                '--host', c['node_ip'], '--port', str(port), '--deadline-utc', base.utc(deadline), '--execute']
        with (root / 'wrapper-console.log').open('x') as log:
            try:
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', c['node'], shlex.join(argv)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time() + 10))
            except BaseException as error:
                base.write(root / 'wrapper-exit.json', {'at': base.utc(), 'exit_code': None, 'error': repr(error), 'original_deadline': deadline})
                raise
        base.write(root / 'wrapper-exit.json', {'at': base.utc(), 'exit_code': result.returncode})
    finally:
        try:
            before = base.node(c, 'manifest'); base.write(root / 'diagnostic-source-before.json', {'at': base.utc(), 'files': before})
            base.retain(c, 'run', before, root / 'node-output', timeout=min(300, max(1, c['lease_timestamp'] - time.time() - 45)))
            after = base.node(c, 'manifest')
            if before != after:
                raise ValueError('Diagnostic output changed during retention')
            base.write(root / 'diagnostic-retention.json', {'at': base.utc(), 'files': before, 'source_before_after_and_destination_verified': True})
        except BaseException as error:
            base.write(root / 'diagnostic-retention-failure.json', {'at': base.utc(), 'state': 'FAILED', 'error': repr(error)})
            raise
        finally:
            final = inspected(diagnostic_id); identity(c, final, image_id, True, require_running=False)
            if final['Id'] != diagnostic_id:
                raise ValueError('Diagnostic container replaced')
            if final['State']['Running']:
                base.remote(c, ['docker', 'stop', '--time', '10', diagnostic_id], 30)
            base.write(root / 'diagnostic-stopped.json', {'at': base.utc(), 'container_id': diagnostic_id})
    if result is None or result.returncode != 0:
        raise RuntimeError('TRTLLM profile wrapper did not succeed; inspect retained partial evidence')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path); p.add_argument('--run-id', required=True)
    p.add_argument('--frozen-input', required=True, type=Path); p.add_argument('--frozen-sha256', required=True)
    p.add_argument('--port', type=int, default=31081); p.add_argument('--execute', action='store_true')
    a = p.parse_args(); c = base.main_driver._config(json.loads(a.config.read_text()))
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{5,79}', a.run_id)
            or not re.fullmatch(r'[0-9a-f]{64}', a.frozen_sha256) or not 1024 <= a.port <= 65535
            or c['sglang_moe_runner_backend'] != 'flashinfer_trtllm'):
        raise ValueError('Exact unique identity, frozen input SHA and TRTLLM main required')
    c.update(diagnostic_run=a.run_id, diagnostic_name='miles-graph-diagnostic-' + a.run_id,
             node_base=str(Path(c['node_run_dir']).parent / ('trtllm-profile-' + a.run_id)),
             durable=str(Path(c['run_dir']) / 'diagnostics' / a.run_id))
    print(json.dumps({'mode': 'EXECUTE' if a.execute else 'PLAN_ONLY', 'config': c,
                      'docker_argv': docker_command(c), 'frozen_input_sha256': a.frozen_sha256}, indent=2), flush=True)
    if a.execute:
        execute(c, a.port, a.frozen_input.resolve(), a.frozen_sha256)


if __name__ == '__main__':
    main()
