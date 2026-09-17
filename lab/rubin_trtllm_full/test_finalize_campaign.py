"""Synthetic receipt guards only; no actual collection, SSH or rendering."""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import finalize_campaign as final


def fixture():
    platform = 'rubin'; run_id = '20260917-rubin-j2213753-trtllm'
    c = {'run_id': run_id, 'run_dir': str(final.REMOTE / run_id), 'image': 'fixture-image'}
    diagnostic, selected = final.paths(platform, c)
    data = {
        'state': {'run_id': run_id, 'platform': platform, 'stage': 'COMPLETE', 'profile_root': str(diagnostic)},
        'worker_exit': {'state': 'COMPLETE', 'exit_code': 0},
        'checkpoint': {'status': 'PASS', 'run_id': run_id, 'iteration': 49},
        'plan': {'schema': 'trtllm-prefill-decode-profile-v1', 'main_run_id': run_id, 'order': ['on'],
            'identity': {'run_id': diagnostic.name, 'main_run_id': run_id, 'image_reference': c['image'],
                         'source_commits': {'miles_main': final.COMMIT}}, 'frozen_input_sha256': 'a' * 64},
        'terminal': {'status': 'COMPLETED'},
        'operator': {'config': {'run_id': run_id, 'image': c['image']}, 'ray_job': {'status': 'SUCCEEDED'},
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
