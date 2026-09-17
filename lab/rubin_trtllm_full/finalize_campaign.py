#!/usr/bin/env python3
"""Finalize retained TRTLLM evidence locally; plan-only unless --execute.

No training, container control, allocation changes, Git operations or publishing.
The existing collector may perform its guarded read-only allocation/node checks.
"""
import argparse
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / 'outputs/rubin-gb300-qwen3-trtllm-full'
REPORT = ROOT / 'reports/rubin-gb300-qwen3-trtllm-full'
REMOTE = Path('/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full')
JOBS = {'rubin': '2213753', 'gb300': '2212644'}
RUN_IDS = {'rubin': '20260917-rubin-j2213753-trtllm-r2', 'gb300': '20260917-gb300-j2212644-trtllm'}
CAMPAIGNS = {'rubin': '20260917-campaign-rubin-j2213753-r2', 'gb300': '20260917-campaign'}
IMAGES = {'rubin': '405315ba3add773be16cfe176a4dcd15a1071cbf17f3be255ab2c47324c65f34',
          'gb300': 'c8b88b345c450e6c40a45122dbedbfc992a2b1f834e2a960d7a340bf1cd89680'}
NODE = Path('/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def bound_configs(config_path, experiment):
    mapping = json.loads(config_path.read_text())
    require(set(mapping) == set(JOBS) and all(mapping.values()), 'Both actual driver configs are required')
    entries = json.loads(experiment.read_text()).get('run_bindings')
    require(isinstance(entries, list) and len(entries) == 2
            and all(isinstance(r, dict) for r in entries), 'Require exactly two explicit experiment bindings')
    bindings = {r.get('label'): r for r in entries}
    require(set(bindings) == set(JOBS), 'Require one explicit binding per platform')
    for platform, binding in bindings.items():
        require(binding.get('run_id') == RUN_IDS[platform]
                and isinstance(binding.get('source_commit'), str)
                and re.fullmatch(r'[0-9a-f]{40}', binding['source_commit']),
                'Exact run and per-platform source_commit required in experiment binding: ' + platform)
    configs = {}
    for platform, job in JOBS.items():
        path = Path(mapping[platform])
        c = json.loads((path if path.is_absolute() else config_path.parent / path).read_text())
        run_id = RUN_IDS[platform]
        require(c.get('run_id') == run_id and str(c.get('job_id')) == job
                and c.get('run_dir') == str(REMOTE / run_id) and c.get('platform') == platform
                and c.get('source_commit') == bindings[platform]['source_commit']
                and c.get('image') == f'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-{platform}@sha256:{IMAGES[platform]}',
                'Wrong/stale run, image or source binding: ' + platform)
        configs[platform] = c
    return configs


def paths(platform, c):
    require(platform in RUN_IDS and c.get('run_id') == RUN_IDS[platform]
            and c.get('run_dir') == str(REMOTE / RUN_IDS[platform]), 'Wrong exact run binding for receipt paths')
    profile = c['run_id'].removeprefix('20260917-') + '-profile-v1'
    diagnostic = Path(c['run_dir']) / 'diagnostics' / profile
    worker = REMOTE / CAMPAIGNS[platform] / 'workers' / platform
    return diagnostic, {'state': worker / 'state.json', 'worker_exit': worker / 'worker-exit.json',
        'checkpoint': Path(c['run_dir']) / 'final-checkpoint-retention.json',
        'retention': diagnostic / 'diagnostic-retention.json', 'operator': diagnostic / 'operator-plan.json',
        'plan': diagnostic / 'node-output/profile/plan.json', 'terminal': diagnostic / 'node-output/profile/terminal.json'}


def remote_read(paths):
    """Read only these small durable receipts on dl3; no subprocess/node access."""
    import hashlib, pathlib, signal
    signal.alarm(45)
    result = {}
    for name, value in paths.items():
        path = pathlib.Path(value)
        assert path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]), str(path)
        before = path.stat()
        assert before.st_uid == 28644 and before.st_size <= 8 * 1024**2, str(path)
        raw = path.read_bytes()
        after = path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        result[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'raw': raw.decode()}
    return result


def snapshot(platform, c):
    _, selected = paths(platform, c)
    code = 'import json\n' + inspect.getsource(remote_read) + '\nprint(json.dumps(remote_read(' + repr({k: str(v) for k, v in selected.items()}) + ')))'
    raw = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'dl3',
                                   shlex.join(['python3', '-B', '-c', code])], timeout=60)
    return json.loads(raw)


def validate_ready(platform, c, records):
    diagnostic, selected = paths(platform, c)
    require(isinstance(c.get('source_commit'), str) and re.fullmatch(r'[0-9a-f]{40}', c['source_commit']),
            'Profile requires the bound exact source commit')
    require(set(records) == set(selected), 'Incomplete readiness receipts')
    data = {}
    for name, expected in selected.items():
        packet = records[name]; raw = packet['raw'].encode()
        require(packet['path'] == str(expected) and len(raw) == packet['bytes']
                and digest(raw) == packet['sha256'], 'Readiness packet identity/hash mismatch')
        data[name] = json.loads(raw)
    state, exited = data['state'], data['worker_exit']
    require(state.get('run_id') == c['run_id'] and state.get('platform') == platform
            and state.get('stage') == 'COMPLETE' and state.get('profile_root') == str(diagnostic)
            and exited.get('state') == 'COMPLETE' and exited.get('exit_code') == 0, 'Controller is not COMPLETE: ' + platform)
    checkpoint = data['checkpoint']
    require(checkpoint.get('status') == 'PASS' and checkpoint.get('run_id') == c['run_id']
            and checkpoint.get('iteration') == 49, 'Final checkpoint retention not complete')
    plan, terminal, operator, retention = [data[x] for x in ('plan', 'terminal', 'operator', 'retention')]
    identity = plan.get('identity', {})
    require(plan.get('schema') == 'trtllm-prefill-decode-profile-v1' and plan.get('main_run_id') == c['run_id']
            and identity.get('run_id') == diagnostic.name and identity.get('main_run_id') == c['run_id']
            and identity.get('image_reference') == c['image']
            and identity.get('source_commits', {}).get('miles_main') == c['source_commit']
            and plan.get('order') == ['on'] and terminal.get('status') == 'COMPLETED', 'Stale or incomplete profile')
    require(operator.get('config', {}).get('run_id') == c['run_id'] and operator['config'].get('image') == c['image']
            and operator['config'].get('source_commit') == c['source_commit']
            and operator.get('ray_job', {}).get('status') == 'SUCCEEDED'
            and operator.get('frozen_input_sha256') == plan.get('frozen_input_sha256'), 'Operator identity/completion differs')
    require(retention.get('source_before_after_and_destination_verified') is True, 'Profile retention not verified')
    files = retention.get('files', {})
    require(files and sum(r['bytes'] for r in files.values()) <= 1100 * 1024**2, 'Invalid/oversized profile manifest')
    for name, rec in files.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and name == str(rel)
                and not any(part in ('checkpoints', 'cache', 'source') for part in rel.parts)
                and re.fullmatch(r'[A-Za-z0-9_./-]+', name)
                and type(rec.get('bytes')) is int and rec['bytes'] >= 0
                and re.fullmatch(r'[0-9a-f]{64}', rec.get('sha256', '')), 'Unsafe profile manifest entry')
    for name in ('plan', 'terminal'):
        rec = files.get('profile/' + name + '.json', {})
        require(rec.get('sha256') == records[name]['sha256'] and rec.get('bytes') == records[name]['bytes'],
                'Retained profile ' + name + ' hash differs')
    return files


def execute(args, configs):
    for directory in (args.work, args.site, args.qa):
        require(not directory.exists(), 'Fresh destination required: ' + str(directory))
    require(args.node.is_file(), 'Local bundled Node.js is required for rendering')
    ready = {p: snapshot(p, c) for p, c in configs.items()}
    manifests = {p: validate_ready(p, configs[p], ready[p]) for p in configs}
    hashes = {json.loads(r['plan']['raw'])['frozen_input_sha256'] for r in ready.values()}
    require(len(hashes) == 1, 'Platforms did not use the same frozen input')
    args.work.mkdir(parents=True)
    write(args.work / 'readiness.json', ready)
    def command(name, argv, timeout):
        write(args.work / (name + '-command.json'), {'argv': argv})
        with (args.work / (name + '.log')).open('x') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    profiles = {}
    for platform, c in configs.items():
        diagnostic, _ = paths(platform, c)
        local = args.work / platform; local.mkdir()
        names = sorted(manifests[platform])
        file_list = args.work / (platform + '-files.txt'); file_list.write_text('\n'.join(names) + '\n')
        command(platform + '-fetch', ['rsync', '-rt', '--no-links', '--no-perms', '--omit-dir-times',
                '--files-from=' + str(file_list), 'dl3:' + str(diagnostic / 'node-output') + '/', str(local) + '/'], 600)
        for name, expected in manifests[platform].items():
            path = local / name
            require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents])
                    and path.stat().st_size == expected['bytes'] and file_sha(path) == expected['sha256'],
                    'Fetched artifact differs from retained manifest: ' + name)
        profiles[platform] = args.work / (platform + '-analysis')
        command(platform + '-analyze', [sys.executable, str(ROOT / 'lab/rubin_trtllm_full/profiling/analyze_profile.py'),
                '--directory', str(local / 'profile'), '--platform', platform, '--main-run-id', c['run_id'],
                '--plan-sha256', ready[platform]['plan']['sha256'], '--terminal-sha256', ready[platform]['terminal']['sha256'],
                '--output', str(profiles[platform]), '--png'], 600)
    command('collect', [sys.executable, str(ROOT / 'lab/rubin_two_node/collect_cudagraph_snapshot.py'),
            '--config', str(args.config), '--output', str(INPUTS)], 300)
    command('build', [sys.executable, str(ROOT / 'lab/rubin_trtllm_full/build_report.py'), '--inputs', str(INPUTS),
            '--experiment', str(args.experiment), '--profile-rubin', str(profiles['rubin'] / 'profile-evidence.json'),
            '--profile-gb300', str(profiles['gb300'] / 'profile-evidence.json'), '--output', str(args.site), '--final'], 180)
    command('browser-qa', [str(args.node), str(REPORT / 'check_slides.mjs'), str(args.site), str(args.qa)], 240)
    result = {'status': 'FINAL_AND_BROWSER_QA_PASSED', 'site': str(args.site / 'index.html'), 'qa': str(args.qa),
              'run_ids': {p: c['run_id'] for p, c in configs.items()}, 'published': False,
              'finalizer_sha256': file_sha(Path(__file__)), 'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(args.work / 'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=INPUTS / 'collector-config.json')
    parser.add_argument('--experiment', type=Path, default=REPORT / 'experiment.json')
    parser.add_argument('--work', type=Path, default=INPUTS / 'finalization')
    parser.add_argument('--site', type=Path, default=REPORT / 'site-final')
    parser.add_argument('--qa', type=Path, default=REPORT / 'qa-final')
    parser.add_argument('--node', type=Path, default=NODE)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for name in ('config', 'experiment', 'work', 'site', 'qa', 'node'):
        setattr(args, name, getattr(args, name).resolve())
    require(INPUTS in args.work.parents and REPORT in args.site.parents and REPORT in args.qa.parents,
            'Keep output under this new campaign/report only')
    configs = bound_configs(args.config, args.experiment)
    if not args.execute:
        result = {'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'run_ids': {p: c['run_id'] for p, c in configs.items()},
                  'requires': 'Both durable controllers COMPLETE; verified retained profiles',
                  'work': str(args.work), 'site': str(args.site), 'qa': str(args.qa),
                  'steps': ['read dl3 receipts', 'fetch manifest-listed diagnostic files only', 'analyze --png',
                            'existing collector', 'build_report --final', 'browser QA'], 'publishes': False}
    else:
        work_existed = args.work.exists()
        try:
            result = execute(args, configs)
        except BaseException as error:
            if not work_existed and args.work.is_dir():
                write(args.work / 'failure.json', {'status': 'FAILED', 'error': repr(error), 'automatic_retry': False})
            raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
