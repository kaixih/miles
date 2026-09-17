"""Synthetic receipt guards only; no actual collection, SSH or rendering."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import finalize_campaign as final


def fixture():
    platform = 'rubin'; run_id = '20260917-rubin-j2213753-trtllm-r2'
    c = {'run_id': run_id, 'run_dir': str(final.REMOTE / run_id), 'image': 'fixture-image',
         'source_commit': 'b' * 40}
    diagnostic, selected = final.paths(platform, c)
    data = {
        'state': {'run_id': run_id, 'platform': platform, 'stage': 'COMPLETE', 'profile_root': str(diagnostic)},
        'worker_exit': {'state': 'COMPLETE', 'exit_code': 0},
        'checkpoint': {'status': 'PASS', 'run_id': run_id, 'iteration': 49},
        'plan': {'schema': 'trtllm-prefill-decode-profile-v1', 'main_run_id': run_id, 'order': ['on'],
            'identity': {'run_id': diagnostic.name, 'main_run_id': run_id, 'image_reference': c['image'],
                         'source_commits': {'miles_main': c['source_commit']}}, 'frozen_input_sha256': 'a' * 64},
        'terminal': {'status': 'COMPLETED'},
        'operator': {'config': {'run_id': run_id, 'image': c['image'], 'source_commit': c['source_commit']},
                     'ray_job': {'status': 'SUCCEEDED'},
                     'frozen_input_sha256': 'a' * 64},
        'retention': {'source_before_after_and_destination_verified': True, 'files': {}}}
    def packet(name):
        raw = json.dumps(data[name], indent=2)
        return {'path': str(selected[name]), 'raw': raw, 'bytes': len(raw.encode()), 'sha256': final.digest(raw.encode())}
    records = {name: packet(name) for name in data}
    for name in ('plan', 'terminal'):
        data['retention']['files']['profile/' + name + '.json'] = {k: records[name][k] for k in ('bytes', 'sha256')}
    records['retention'] = packet('retention')
    return c, data, records, packet


class FinalizationGuards(unittest.TestCase):
    def test_binding_accepts_distinct_exact_platform_sources_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config_path = root / 'collector.json'; experiment = root / 'experiment.json'
            mapping = {p: p + '.json' for p in final.JOBS}; configs = {}; bindings = []
            for platform, job in final.JOBS.items():
                run_id = final.RUN_IDS[platform]; commit = ('a' if platform == 'gb300' else 'b') * 40
                configs[platform] = {'platform': platform, 'job_id': job, 'run_id': run_id,
                    'run_dir': str(final.REMOTE / run_id), 'source_commit': commit,
                    'image': f'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-{platform}@sha256:{final.IMAGES[platform]}'}
                (root / mapping[platform]).write_text(json.dumps(configs[platform]))
                bindings.append({'label': platform, 'run_id': run_id, 'source_commit': commit})
            config_path.write_text(json.dumps(mapping)); experiment.write_text(json.dumps({'run_bindings': bindings}))
            self.assertEqual(final.bound_configs(config_path, experiment), configs)
            # Reusing GB's old source for the newly bound Rubin retry is not allowed.
            changed = {**configs['rubin'], 'source_commit': configs['gb300']['source_commit']}
            (root / mapping['rubin']).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Wrong/stale'):
                final.bound_configs(config_path, experiment)
            (root / mapping['rubin']).write_text(json.dumps(configs['rubin']))
            for commit in (None, '', 'old-main', 'c' * 39, 'c' * 41):
                bad = [dict(r) for r in bindings]; bad[0]['source_commit'] = commit
                experiment.write_text(json.dumps({'run_bindings': bad}))
                with self.subTest(commit=commit), self.assertRaisesRegex(ValueError, 'source_commit'):
                    final.bound_configs(config_path, experiment)
            # Even mutually matching stale config/manifest cannot select the failed attempt.
            bad = [dict(r) for r in bindings]; bad[0]['run_id'] = bad[0]['run_id'].removesuffix('-r2')
            changed = {**configs['rubin'], 'run_id': bad[0]['run_id'], 'run_dir': str(final.REMOTE / bad[0]['run_id'])}
            (root / mapping['rubin']).write_text(json.dumps(changed)); experiment.write_text(json.dumps({'run_bindings': bad}))
            with self.assertRaisesRegex(ValueError, 'Exact run'):
                final.bound_configs(config_path, experiment)

    def test_retry_paths_are_derived_from_exact_bound_run(self):
        c, _, _, _ = fixture()
        diagnostic, paths = final.paths('rubin', c)
        self.assertEqual(diagnostic.name, 'rubin-j2213753-trtllm-r2-profile-v1')
        self.assertEqual(paths['state'], final.REMOTE / '20260917-campaign-rubin-j2213753-r2/workers/rubin/state.json')
        old = c['run_id'].removesuffix('-r2')
        with self.assertRaisesRegex(ValueError, 'exact run binding'):
            final.paths('rubin', {**c, 'run_id': old, 'run_dir': str(final.REMOTE / old)})

    def test_actual_controller_schema_and_retention_hashes_pass(self):
        c, data, records, _ = fixture()
        self.assertEqual(final.validate_ready('rubin', c, records), data['retention']['files'])

    def test_pending_old_run_and_wrong_image_never_pass(self):
        for name, key, value in [('state', 'stage', 'RUNNING'), ('plan', 'main_run_id', '20260917-rubin-j2212643-trtllm'),
                                 ('terminal', 'status', 'FAILED')]:
            c, data, records, packet = fixture(); data[name][key] = value; records[name] = packet(name)
            with self.subTest(name=name), self.assertRaises(ValueError): final.validate_ready('rubin', c, records)
        c, data, records, packet = fixture()
        data['plan']['identity']['image_reference'] = 'different-image'; records['plan'] = packet('plan')
        with self.assertRaisesRegex(ValueError, 'Stale'): final.validate_ready('rubin', c, records)

    def test_old_attempt_profile_and_source_mismatch_are_rejected(self):
        for key, value in [('run_id', 'rubin-j2213753-trtllm-profile-v1'),
                           ('source_commits', {'miles_main': 'a' * 40})]:
            c, data, records, packet = fixture(); data['plan']['identity'][key] = value
            records['plan'] = packet('plan')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Stale'):
                final.validate_ready('rubin', c, records)
        c, data, records, packet = fixture(); data['operator']['config']['source_commit'] = 'a' * 40
        records['operator'] = packet('operator')
        with self.assertRaisesRegex(ValueError, 'Operator identity'):
            final.validate_ready('rubin', c, records)

    def test_receipt_hash_and_unsafe_manifest_are_rejected_before_fetch(self):
        c, data, records, packet = fixture()
        data['retention']['files']['profile/plan.json']['sha256'] = '0' * 64
        records['retention'] = packet('retention')
        with self.assertRaisesRegex(ValueError, 'hash differs'): final.validate_ready('rubin', c, records)
        for name in ('../private', '/absolute', 'checkpoints/iter49.distcp'):
            c, data, records, packet = fixture()
            data['retention']['files'][name] = {'bytes': 1, 'sha256': 'b' * 64}; records['retention'] = packet('retention')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe'):
                final.validate_ready('rubin', c, records)
        c, _, records, _ = fixture(); records['terminal']['raw'] += ' '
        with self.assertRaisesRegex(ValueError, 'packet identity/hash'): final.validate_ready('rubin', c, records)

    def test_plan_only_makes_no_remote_or_render_calls(self):
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['finalize_campaign.py']), \
                patch.object(final, 'bound_configs', return_value={p: {'run_id': 'fixture-' + p} for p in final.JOBS}), \
                patch.object(final.subprocess, 'run', side_effect=AssertionError('execute')), \
                patch.object(final.subprocess, 'check_output', side_effect=AssertionError('read')), \
                contextlib.redirect_stdout(stream):
            final.main()
        self.assertEqual(json.loads(stream.getvalue())['mode'], 'PLAN_ONLY_NO_REMOTE_CALLS')


if __name__ == '__main__':
    unittest.main()
