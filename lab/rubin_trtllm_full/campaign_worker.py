#!/usr/bin/env python3
"""Durable one-platform controller for the fixed full TRTLLM campaign.

Plan-only by default. Execute on dl3 as UID28644:GID30. Rubin waits for its
existing allocation, prepares the image/storage, freezes shared CPU-tokenized
profile inputs once, submits main training once, validates completion, retains
checkpoint49, then performs the independent profile. --prepared-only requires
an already verified READY preparation and skips image/preparation work.

An existing controller claim or any main-launch evidence is never resumed or
resubmitted automatically. Failures and partial evidence remain for inspection.
This helper never allocates/releases nodes or resets an experiment deadline.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

BASE = Path('/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full')
CAMPAIGN = BASE / '20260917-campaign'
SOURCE = CAMPAIGN / 'source'
MANIFEST = CAMPAIGN / 'source-manifest.json'
JOBS = {'rubin': '2212643', 'gb300': '2212644'}
WAIT_UNTIL = '2026-09-17T08:00:00Z'
RETENTION = Path('/home/scratch.kaixih_ent/repo/miles-rubin-cu134/lab/rubin_trtllm_full/retain_final_checkpoint.py')
MAIN_MARKERS = ('main-worker-launch.json', 'train-launch.json', 'train_exit.json',
                'train-driver-exit.json', 'cudagraph-main.claim', 'cudagraph-main-plan.json',
                'logs/qwen3_train.log', 'logs/train-driver.log')


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timezone required')
    return parsed.timestamp()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024**2,
            'Missing/oversized/symlinked JSON: ' + str(path))
    return json.loads(path.read_text())


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
        with temporary.open('x') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def no_main_evidence(root):
    found = [name for name in MAIN_MARKERS if (root / name).exists()]
    require(not found, 'Main evidence already exists; refusing automatic resubmission: ' + ', '.join(found))


def frozen_record(path, receipt, capture):
    require(path.exists() == receipt.exists(), 'Incomplete shared frozen-input publication; inspect, do not overwrite')
    if not path.exists():
        return None
    record = read(receipt)
    require(record.get('status') == 'PASS' and record.get('bytes') == path.stat().st_size
            and record.get('sha256') == sha(path), 'Shared frozen-input receipt/hash mismatch')
    data = read(path)
    request = capture.generation_request(data, 'campaign-shared-frozen-input-audit')
    require(len(data['input_ids']) == 128 and record.get('input_ids_sha256') == request['input_ids_sha256']
            and record.get('workload_sha256') == request['workload_sha256'], 'Shared frozen-input workload mismatch')
    return record


class Worker:
    def __init__(self, args):
        self.args = args
        self.platform, self.job = args.platform, JOBS[args.platform]
        self.run_id = f'20260917-{self.platform}-j{self.job}-trtllm'
        self.root = BASE / self.run_id
        self.work = CAMPAIGN / 'workers' / self.platform
        self.config_path = self.root / 'driver-config.json'
        self.original = None
        self.c = None
        self.source, self.commit, self.manifest_sha = SOURCE, None, None

    def state(self, stage, **extra):
        write(self.work / 'state.json', {'at': utc(), 'platform': self.platform, 'run_id': self.run_id,
              'stage': stage, 'original_allocation': self.original, **extra})

    def verify_source(self):
        manifest = read(MANIFEST)
        commit = manifest.get('git_commit', '')
        source = Path(manifest.get('source_root', str(SOURCE)))
        require(re.fullmatch(r'[0-9a-f]{40}', commit), 'Exact frozen source commit required')
        require(source.is_absolute() and source.parent == CAMPAIGN and source.name.startswith('source')
                and '..' not in source.parts and source.is_dir() and not source.is_symlink()
                and source.stat().st_uid == 28644, 'Frozen source must be a campaign-owned source directory')
        digest = sha(MANIFEST)
        require(self.manifest_sha is None or (self.manifest_sha == digest and self.source == source and self.commit == commit),
                'Frozen source manifest changed after controller start')
        self.source, self.commit, self.manifest_sha = source, commit, digest
        hashes = manifest.get('source_sha256', {})
        required = ('lab/rubin_trtllm_full/prepare_platform.py',
                    'lab/rubin_trtllm_full/profile_operator.py',
                    'lab/rubin_trtllm_full/profiling/run_profile.py',
                    'lab/rubin_two_node/run_cudagraph_main.py')
        require(all(name in hashes for name in required), 'Frozen manifest omits a required executed helper')
        for name, expected in hashes.items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe manifest path')
            path = self.source / rel
            require(not any(p.is_symlink() for p in [path, *path.parents]) and sha(path) == expected,
                    'Frozen campaign source changed: ' + name)
        self.prepare = module('_frozen_campaign_preparation', self.source / required[0])
        sys.path.insert(0, str(self.source / 'lab/rubin_two_node'))
        self.driver = module('_frozen_campaign_main', self.source / required[3])
        self.capture = module('_frozen_campaign_capture', self.source / 'lab/rubin_two_node/sglang_graph_capture.py')
        self.collector = module('_frozen_campaign_collector', self.source / 'lab/rubin_two_node/collect_cudagraph_snapshot.py')
        self.diag = module('_frozen_campaign_diagnostic_checks', self.source / 'lab/rubin_two_node/run_cudagraph_diagnostic_operator.py')
        return {'git_commit': self.commit, 'manifest_sha256': sha(MANIFEST), 'files_checked': len(hashes)}

    def allocation(self):
        raw = subprocess.check_output(['env', 'TZ=UTC', 'scontrol', 'show', 'job', '-o', self.job],
                                      text=True, timeout=25)
        values = dict(re.findall(r'(\w+)=([^\s]+)', raw))
        decision = self.prepare.allocation_guard({'job_id': self.job}, values, time.time(), self.original)
        return decision, values

    def wait_allocation(self):
        while True:
            decision, values = self.allocation()
            if decision == 'ready':
                self.original = values
                write(self.work / 'allocation.json', values, exclusive=True)
                return
            require(time.time() < stamp(WAIT_UNTIL), 'Fixed queue deadline expired; no allocation was changed')
            self.state('WAITING_FOR_OWN_ALLOCATION', job_id=self.job, wait_until=WAIT_UNTIL)
            time.sleep(min(60, max(1, stamp(WAIT_UNTIL) - time.time())))

    def remaining(self, reserve=120):
        self.allocation()
        seconds = stamp(self.original['EndTime'] + 'Z') - time.time() - reserve
        require(seconds > 0, 'Original lease budget exhausted')
        return seconds

    def command(self, stage, argv, maximum, *, main=False):
        self.state(stage, command=argv)
        write(self.work / (stage + '-launch.json'), {'at': utc(), 'command': argv,
              'remaining_lease_seconds': self.remaining(), 'source_commit': self.commit}, exclusive=True)
        log = self.work / (stage + '.log')
        try:
            with log.open('x') as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=max(1, min(maximum, self.remaining())))
            record = {'at': utc(), 'exit_code': result.returncode, 'log': str(log)}
        except BaseException as error:
            record = {'at': utc(), 'exit_code': None, 'error': repr(error), 'log': str(log)}
            write(self.work / (stage + '-exit.json'), record, exclusive=True)
            if main:
                write(self.root / 'main-worker-exit.json', record, exclusive=True)
            raise
        write(self.work / (stage + '-exit.json'), record, exclusive=True)
        if main:
            write(self.root / 'main-worker-exit.json', record, exclusive=True)
        require(result.returncode == 0, stage + ' failed; preserving state without automatic retry')
        return record

    def ssh(self, argv, maximum=120):
        seconds = int(min(maximum, self.remaining()))
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                   '-o', 'StrictHostKeyChecking=accept-new', self.original['NodeList'],
                   shlex.join(['timeout', '--signal=TERM', '--kill-after=10s', str(seconds) + 's', *argv])]
        return subprocess.check_output(command, timeout=seconds + 20)

    def image_ready(self):
        path = CAMPAIGN / (self.platform + '-image/ready.json')
        if not path.exists():
            require(not self.args.prepared_only, 'Prepared-only requires the existing image READY receipt')
            helper = CAMPAIGN / 'ensure_image.py'
            write(self.work / 'image-helper.json', {'path': str(helper), 'sha256': sha(helper)}, exclusive=True)
            self.command('image', ['python3', '-u', str(helper), self.platform, '--execute'], 3600)
        receipt = read(path)
        require(receipt.get('status') == 'READY' and receipt.get('platform') == self.platform
                and str(receipt.get('job')) == self.job and receipt.get('architecture') == 'arm64',
                'Image READY receipt identity mismatch')
        require(all(receipt.get('allocation', {}).get(k) == self.original[k] for k in ('NodeList', 'StartTime', 'EndTime')),
                'Image was prepared under a different allocation/lease')
        require(re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', receipt.get('image', '')), 'Image is not immutable')
        return receipt

    def prepare_main(self):
        image = self.image_ready()
        if not self.args.prepared_only:
            no_main_evidence(self.root)
            command = ['python3', '-u', str(self.source / 'lab/rubin_trtllm_full/prepare_platform.py'),
                       '--platform', self.platform, '--source', str(self.source), '--source-commit', self.commit,
                       '--source-manifest', str(MANIFEST), '--image', image['image'],
                       '--wait-until', WAIT_UNTIL, '--execute']
            self.command('preparation', command, max(1, self.remaining() - 4 * 3600))
        ready = read(self.root / 'prep-state.json')
        require(ready.get('state') == 'READY_NOT_TRAINING' and ready.get('run_id') == self.run_id
                and ready.get('config') == str(self.config_path), 'Preparation is not verified READY_NOT_TRAINING')
        raw = read(self.config_path)
        require(raw.get('run_id') == self.run_id and raw.get('platform') == self.platform
                and str(raw.get('job_id')) == self.job and raw.get('repo') == str(self.source)
                and raw.get('source_commit') == self.commit and raw.get('image') == image['image']
                and raw.get('node') == self.original['NodeList']
                and stamp(raw['lease_deadline']) == stamp(self.original['EndTime'] + 'Z'), 'Prepared driver configuration changed')
        self.c = self.driver._config(raw)
        no_main_evidence(self.root)
        inspected = json.loads(self.ssh(['docker', 'inspect', self.c['container']]))[0]
        require(inspected.get('Name') == '/' + self.c['container'] and inspected['State']['Running']
                and inspected['Config']['Image'] == self.c['image'] and inspected['Image'] == image['image_id']
                and inspected['Config']['User'] == '28644:30', 'Prepared container identity/running state differs')
        mounts = {m['Destination']: m for m in inspected['Mounts']}
        for target, source, rw in [('/opt/miles', self.c['node_repo'], False),
                                   (self.c['models'], self.c['node_models'], False),
                                   ('/run-output', self.c['node_run_dir'], True)]:
            require(mounts.get(target, {}).get('Source') == source and mounts[target]['RW'] is rw,
                    'Prepared container mount differs: ' + target)
        write(self.work / 'prepared-validation.json', {'at': utc(), 'state': ready,
              'driver_config_sha256': sha(self.config_path), 'container_id': inspected['Id'],
              'image_id': image['image_id']}, exclusive=True)

    def freeze_inputs(self):
        target, receipt = CAMPAIGN / 'frozen-token-input.json', CAMPAIGN / 'frozen-token-input-receipt.json'
        self.state('FREEZING_SHARED_PROFILE_INPUT')
        with (CAMPAIGN / 'frozen-token-input.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    require(self.remaining() > 4 * 3600, 'Insufficient lease while awaiting the shared input lock')
                    time.sleep(10)
            existing = frozen_record(target, receipt, self.capture)
            if existing:
                return existing
            argv = ['docker', 'exec', '--env', 'CUDA_VISIBLE_DEVICES=', self.c['container'], 'python3', '-B',
                    '/opt/miles/lab/rubin_trtllm_full/profiling/run_profile.py', 'freeze',
                    '--model-path', self.c['models'] + '/Qwen3-30B-A3B',
                    '--dataset', '/run-output/inputs/train.jsonl',
                    '--output', '/run-output/shared-frozen-token-input.json']
            output = self.ssh(argv, 300)
            write(self.work / 'frozen-tokenizer-result.json', json.loads(output), exclusive=True)
            source_path = self.c['node_run_dir'] + '/shared-frozen-token-input.json'
            probe = "from pathlib import Path;import sys;p=Path(" + repr(source_path) + ");assert p.is_file() and not p.is_symlink() and p.stat().st_uid==28644 and p.stat().st_size<2*1024**2;sys.stdout.buffer.write(p.read_bytes())"
            raw = self.ssh(['python3', '-B', '-c', probe])
            require(len(raw) <= 2 * 1024**2, 'Frozen token file exceeds its bound')
            data = json.loads(raw)
            req = self.capture.generation_request(data, 'campaign-shared-frozen-input-audit')
            require(len(data['input_ids']) == 128, 'Expected 128 shared requests')
            with target.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            record = {'status': 'PASS', 'at': utc(), 'source_run_id': self.run_id,
                      'source_node': self.c['node'], 'source_path': source_path,
                      'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'input_ids_sha256': req['input_ids_sha256'], 'workload_sha256': req['workload_sha256'],
                      'requests': 128, 'input_tokens': sum(req['prompt_token_counts'])}
            write(receipt, record, exclusive=True)
            return frozen_record(target, receipt, self.capture)

    def run_main(self):
        no_main_evidence(self.root)
        self.verify_source()
        require(sha(self.config_path) == read(self.work / 'prepared-validation.json')['driver_config_sha256'], 'Driver config changed after READY')
        command = ['python3', '-u', str(self.source / 'lab/rubin_two_node/run_cudagraph_main.py'),
                   '--config', str(self.config_path), '--execute']
        write(self.root / 'main-worker-launch.json', {'at': utc(), 'pid': os.getpid(), 'command': command,
              'controller_sha256': sha(Path(__file__)), 'source_commit': self.commit}, exclusive=True)
        self.command('main', command, self.c['hard_timestamp'] - time.time() + 180, main=True)

    def validate_completion(self):
        self.state('VALIDATING_MAIN')
        driver, train = read(self.root / 'train-driver-exit.json'), read(self.root / 'train_exit.json')
        code = "from pathlib import Path;import sys;p=Path(" + repr(self.c['node_run_dir'] + '/watchdog.json') + ");assert p.stat().st_uid==28644 and not p.is_symlink();sys.stdout.write(p.read_text())"
        until = time.time() + min(120, self.remaining())
        while True:
            watchdog = json.loads(self.ssh(['python3', '-B', '-c', code]))
            if watchdog.get('state') == 'finished':
                break
            require(time.time() < until, 'Main exited but watchdog terminal observation did not arrive within120s')
            time.sleep(5)
        jobs = self.diag.JobsAPI(self.c['dashboard']).list_jobs()
        job = self.diag.completion(self.c, watchdog, jobs, driver, train)
        metadata = {'run_id': self.run_id, 'status': 'SUCCEEDED', 'gpus': 4, 'expected_rollouts': 50,
                    'optimizer_steps_per_rollout': 4, 'profiling_coverage_known': True, 'profiled_rollouts': [],
                    'driver_exits': {'train_exit.json': train, 'train-driver-exit.json': driver}}
        log = self.root / 'logs/qwen3_train.log'
        require(log.is_file() and not log.is_symlink() and log.stat().st_size <= 1024**3, 'Main log missing/oversized')
        summary = self.collector.summarize_run(self.platform, log, metadata)
        self.diag.completion(self.c, watchdog, jobs, driver, train, summary)
        validation = self.collector.completion_validation(summary, log.read_bytes())
        require(sha(log) == summary['source_log_sha256'], 'Main log changed during validation')
        write(self.root / 'campaign-main-completion.json', {'at': utc(), 'run_id': self.run_id,
              'source_log_sha256': summary['source_log_sha256'], 'completion_validation': validation,
              'ray_job': {k: job.get(k) for k in ('submission_id', 'status', 'driver_exit_code', 'start_time', 'end_time')},
              'watchdog': watchdog}, exclusive=True)
        require(validation['status'] == 'PASS' and validation['optimizer_events_observed'] == 200
                and validation['normal_rank_steps_observed'] == 800, 'Main did not pass all 50/200/800 checks')

    def retain_and_profile(self, frozen):
        helper = RETENTION
        while not helper.exists():
            require(self.remaining() > 45 * 60, 'Retention helper not deployed before bounded profile reserve')
            self.state('WAITING_FOR_CHECKPOINT_RETENTION_HELPER', expected_path=str(helper))
            time.sleep(30)
        require(not helper.is_symlink(), 'Retention helper must be a regular reviewed file')
        write(self.work / 'retention-helper.json', {'at': utc(), 'path': str(helper), 'sha256': sha(helper)}, exclusive=True)
        self.command('checkpoint-retention', ['python3', '-u', str(helper), '--config', str(self.config_path), '--execute'], 1900)
        retained = read(self.root / 'final-checkpoint-retention.json')
        require(retained.get('status') == 'PASS' and retained.get('run_id') == self.run_id
                and retained.get('iteration') == 49 and retained.get('source_stable_before_after') is True
                and retained.get('destination_root') == str(self.root / 'final-checkpoint')
                and retained.get('source_checkpoint_id')
                and retained.get('source_checkpoint_id') == retained.get('destination_checkpoint_id')
                and not (self.root / 'final-checkpoint/_INCOMPLETE.json').exists()
                and retained.get('rsync_exit_code') == 0, 'Final checkpoint retention was not verified')
        self.verify_source()
        frozen_record(CAMPAIGN / 'frozen-token-input.json', CAMPAIGN / 'frozen-token-input-receipt.json', self.capture)
        profile_id = f'{self.platform}-j{self.job}-trtllm-profile-v1'
        self.command('profile', ['python3', '-u', str(self.source / 'lab/rubin_trtllm_full/profile_operator.py'),
                     '--config', str(self.config_path), '--run-id', profile_id,
                     '--frozen-input', str(CAMPAIGN / 'frozen-token-input.json'),
                     '--frozen-sha256', frozen['sha256'], '--execute'], 2400)
        directory = self.root / 'diagnostics' / profile_id
        retention = read(directory / 'diagnostic-retention.json')
        terminal = read(directory / 'node-output/profile/terminal.json')
        require(retention.get('source_before_after_and_destination_verified') is True
                and terminal.get('status') == 'COMPLETED', 'Profile workload/retention not complete')
        return {'profile_root': str(directory), 'checkpoint_receipt': str(self.root / 'final-checkpoint-retention.json')}

    def execute(self):
        require((os.getuid(), os.getgid()) == (28644, 30), 'Execute only as UID28644:GID30 on dl3')
        require(CAMPAIGN.is_dir() and CAMPAIGN.stat().st_uid == 28644, 'Existing campaign root required')
        self.work.mkdir(parents=True, exist_ok=True)
        require(not any(p.is_symlink() for p in [self.work, *self.work.parents]), 'Symlink in controller output path')
        with (self.work / 'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(self.work / 'worker-claim.json', {'at': utc(), 'pid': os.getpid(), 'platform': self.platform,
                  'prepared_only': self.args.prepared_only, 'source_manifest': str(MANIFEST),
                  'controller_sha256': sha(Path(__file__)), 'wait_until': WAIT_UNTIL}, exclusive=True)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            try:
                proof = self.verify_source()
                write(self.work / 'source-verification.json', proof, exclusive=True)
                self.wait_allocation()
                self.prepare_main()
                frozen = self.freeze_inputs()
                write(self.work / 'shared-frozen-input.json', frozen, exclusive=True)
                self.run_main()
                self.validate_completion()
                outputs = self.retain_and_profile(frozen)
                self.state('COMPLETE', **outputs, allocation_released=False)
                write(self.work / 'worker-exit.json', {'at': utc(), 'exit_code': 0, 'state': 'COMPLETE'}, exclusive=True)
            except BaseException as error:
                self.state('FAILED', error=repr(error), partial_evidence_preserved=True, automatic_retry=False)
                write(self.work / 'worker-exit.json', {'at': utc(), 'exit_code': 1, 'state': 'FAILED', 'error': repr(error)}, exclusive=True)
                raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=JOBS, default='rubin')
    p.add_argument('--prepared-only', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'mode': 'PLAN_ONLY_NO_REMOTE_CALLS', 'platform': args.platform, 'job_id': JOBS[args.platform],
              'campaign_root': str(CAMPAIGN), 'source_commit': 'resolved_and_pinned_from_manifest', 'source_manifest': str(MANIFEST),
              'prepared_only': args.prepared_only, 'queue_wait_until': WAIT_UNTIL,
              'stages': ['verify_source', 'wait_existing_allocation', 'verify_ready' if args.prepared_only else 'image_and_prepare',
                         'freeze_shared_inputs_once', 'main_once', 'validate_50_200_800', 'retain_checkpoint49', 'profile'],
              'automatic_release': False, 'automatic_resume_or_resubmit': False}, indent=2))
        return
    Worker(args).execute()


if __name__ == '__main__':
    main()
