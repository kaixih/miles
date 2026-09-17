"""CPU-only terminal retention tests; no node, container or remote operations."""
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rubin_two_node'))
import collect_cudagraph_snapshot as C
from test_collect_cudagraph_snapshot import trtllm_config, evidence, payload, packet

T = C.terminal_evidence


def retained():
    c = trtllm_config()
    c.update(source_commit='1' * 40, container_prefix='miles-main-fixture')
    files, p = evidence(c)
    ray = p['live']['ray']
    ray.update(status='SUCCEEDED', driver_exit_code=0, start_time=100, end_time=200,
               runtime_env={'env_vars': {'RUBIN_RUN_ID': c['run_id']}})
    lease = dt.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00'))
    watchdog = {'run_id': c['run_id'], 'submission_id': ray['submission_id'], 'state': 'finished',
                'terminal_status': 'SUCCEEDED', 'stop_request_count': 0, 'expected_rollouts': 50,
                'lease_deadline_at': lease.isoformat(), 'deadline_at': (lease - dt.timedelta(minutes=60)).isoformat(),
                'soft_deadline_at': (lease - dt.timedelta(minutes=90)).isoformat()}
    source = {'git_commit': c['source_commit'], 'source_sha256': {'launcher.py': '2' * 64}}
    files['source-manifest.json'] = C.encoded(source)
    launch = {'git_commit': c['source_commit'], 'source_sha256': source['source_sha256'],
              'source_manifest_sha256': C.sha(files['source-manifest.json'])}
    files['train-launch.json'] = C.encoded(launch)
    files['train-driver-exit.json'] = C.encoded({'run_id': c['run_id'], 'exit_code': 0})
    files['train_exit.json'] = C.encoded({'exit_code': 0})
    image_id, container_id = 'sha256:' + '3' * 64, '4' * 64
    main = {**launch, 'run_id': c['run_id'], 'config': c,
            'preflight': {'container_id': container_id, 'image_id': image_id}}
    files['cudagraph-main-plan.json'] = C.encoded(main)
    root = T.profile_root(c)
    files[root + '/operator-plan.json'] = C.encoded({
        'config': {**c, 'image_id': image_id}, 'ray_job': ray,
        'main_container': {'Id': container_id, 'Image': image_id, 'Name': '/' + c['container_prefix'] + '-0'}})
    files[root + '/main-driver-evidence.json'] = C.encoded({
        'driver_exit': json.loads(files['train-driver-exit.json']),
        'train_exit': json.loads(files['train_exit.json']), 'original_main_plan': main})
    login_files = {name: {'bytes': len(files[name]), 'sha256': C.sha(files[name]), 'source': c['run_dir'] + '/' + name}
                   for name in ['logs/qwen3_train.log', 'train-launch.json', 'cudagraph-main-plan.json',
                                'train_exit.json', 'train-driver-exit.json', 'source-manifest.json']}
    files[root + '/main-login-retention.json'] = C.encoded({
        'stable_source_and_destination_sha256_verified': True, 'files': login_files})
    files[root + '/main-before/watchdog.json'] = C.encoded(watchdog)
    watch_record = {'bytes': len(files[root + '/main-before/watchdog.json']),
                    'sha256': C.sha(files[root + '/main-before/watchdog.json'])}
    files[root + '/main-retention.json'] = C.encoded({
        'source_prefix_and_destination_sha256_verified': True, 'files': {'watchdog.json': watch_record},
        'login_files': login_files})
    files[root + '/main-final/watchdog.json'] = files[root + '/main-before/watchdog.json']
    files[root + '/main-final-retention.json'] = C.encoded({'watchdog.json': watch_record})
    files['campaign-main-completion.json'] = C.encoded({
        'run_id': c['run_id'], 'at': '2026-09-17T05:00:00Z', 'source_log_sha256': C.sha(files[C.LOG]),
        'completion_validation': {'status': 'PASS'}, 'watchdog': watchdog,
        'ray_job': {k: ray.get(k) for k in ('submission_id', 'status', 'driver_exit_code', 'start_time', 'end_time')}})
    p['live'] = None
    p['allocation'] = {'allowed': False, 'reason': 'allocation_JobState_mismatch'}
    return c, files, p


def mutate(files, name, change):
    value = json.loads(files[name]); change(value); files[name] = C.encoded(value)


class TerminalEvidence(unittest.TestCase):
    def test_after_release_actual_ray_success_and_provenance_import(self):
        c, files, p = retained()
        result = C.metadata(c['platform'], c, p, files)
        self.assertEqual(result['status'], 'SUCCEEDED')
        self.assertFalse(result['status_evidence']['live_ray_observed'])
        receipt = json.loads(files['terminal-evidence-import.json'])
        self.assertEqual(receipt['status'], 'VERIFIED')
        self.assertTrue(receipt['watchdog_path'].endswith('/main-final/watchdog.json'))
        self.assertEqual(receipt['source_log_sha256'], C.sha(files[C.LOG]))
        for path, record in receipt['source_artifacts'].items():
            self.assertEqual(record['sha256'], C.sha(files[path]))
        self.assertEqual(json.loads(files['ray-job.json'])['submission_id'], 'job123')

    def test_pre_stop_retention_is_sufficient_if_profiling_failed_after_stop(self):
        c, files, _ = retained(); root = T.profile_root(c)
        for name in ('main-final/watchdog.json', 'main-final-retention.json'):
            del files[root + '/' + name]
        self.assertEqual(T.validate(c, files)['receipt']['watchdog_path'], root + '/main-before/watchdog.json')

    def test_incomplete_curves_or_campaign_claim_cannot_replace_ray_observation(self):
        c, files, p = retained(); root = T.profile_root(c)
        del files[root + '/operator-plan.json']
        self.assertIsNone(T.validate(c, files))
        files['watchdog.json'] = files[root + '/main-before/watchdog.json']
        files['ray-job.json'] = C.encoded({'run_id': c['run_id'], 'submission_id': 'job123', 'status': 'RUNNING'})
        self.assertEqual(C.metadata(c['platform'], c, p, files)['status'], 'UNKNOWN')

    def test_stale_local_running_snapshot_is_replaced_but_live_conflict_rejected(self):
        c, files, p = retained()
        files['ray-job.json'] = C.encoded({'run_id': c['run_id'], 'submission_id': 'job123', 'status': 'RUNNING'})
        self.assertEqual(C.metadata(c['platform'], c, p, files)['status'], 'SUCCEEDED')
        p['live'] = {'ray': {'run_id': c['run_id'], 'submission_id': 'job123', 'status': 'RUNNING'}}
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            C.metadata(c['platform'], c, p, files)

    def test_raw_hashes_and_terminal_identity_cannot_be_changed(self):
        cases = [
            ('watchdog bytes', lambda f, r: f.__setitem__(r + '/main-before/watchdog.json', f[r + '/main-before/watchdog.json'] + b' ')),
            ('log bytes', lambda f, r: f.__setitem__(C.LOG, f[C.LOG] + b'changed\n')),
            ('source bytes', lambda f, r: f.__setitem__('source-manifest.json', f['source-manifest.json'] + b' ')),
            ('Ray status', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['ray_job'].update(status='FAILED'))),
            ('Ray submission', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['ray_job'].update(submission_id='other'))),
            ('Ray run', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['ray_job']['runtime_env']['env_vars'].update(RUBIN_RUN_ID='other'))),
            ('Ray entrypoint', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['ray_job'].update(entrypoint='python3 other.py'))),
            ('image reference', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['config'].update(image='other@sha256:' + '5' * 64))),
            ('image ID', lambda f, r: mutate(f, r + '/operator-plan.json', lambda v: v['main_container'].update(Image='sha256:' + '5' * 64))),
            ('completion run', lambda f, r: mutate(f, 'campaign-main-completion.json', lambda v: v.update(run_id='other'))),
            ('retention incomplete', lambda f, r: mutate(f, r + '/main-login-retention.json', lambda v: v.update(stable_source_and_destination_sha256_verified=False))),
            ('driver exit', lambda f, r: mutate(f, 'train-driver-exit.json', lambda v: v.update(exit_code=1))),
        ]
        for name, change in cases:
            with self.subTest(name=name):
                c, files, _ = retained(); change(files, T.profile_root(c))
                with self.assertRaises(ValueError): T.validate(c, files)

    def test_nested_transport_paths_and_packet_hashes_are_validated(self):
        c, files, p = retained()
        p = payload(c, tail=files[C.LOG]); p['files'] = {n: packet(raw) for n, raw in files.items() if n != C.LOG}
        self.assertEqual(C.decode(c, p, b''), files)
        p['files']['diagnostics/unrelated/operator-plan.json'] = packet(b'{}')
        with self.assertRaisesRegex(ValueError, 'Unexpected auxiliary path'): C.decode(c, p, b'')

    def test_released_allocation_fetch_reads_only_durable_login_files(self):
        c, files, _ = retained()
        with tempfile.TemporaryDirectory() as directory:
            for name, raw in files.items():
                path = Path(directory, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            remote_c = {**c, 'run_dir': str(Path(directory).resolve())}
            calls = []
            def command(argv, **kwargs):
                calls.append(argv)
                self.assertEqual(argv[:5], ['env', 'TZ=UTC', 'scontrol', 'show', 'job'])
                return b'JobId=2212644 JobState=COMPLETED'
            try:
                with patch('subprocess.check_output', side_effect=command):
                    value = C.remote_read(remote_c, {'bytes': 0, 'sha256': C.sha(b'')}, C.evidence_files(c), 'raise AssertionError("node accessed")')
            finally:
                signal.alarm(0)
            self.assertIsNone(value['live']); self.assertFalse(value['allocation']['allowed'])
            self.assertEqual(len(calls), 1)
            self.assertIn(T.profile_root(c) + '/operator-plan.json', value['files'])
            self.assertEqual(C.decode(remote_c, value, b''), files)


if __name__ == '__main__':
    unittest.main()
