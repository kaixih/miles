"""CPU-only guards for durable claims, shared-input publication and plan mode."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import campaign_worker as worker

spec = importlib.util.spec_from_file_location('campaign_capture_fixture', Path(__file__).parents[1] / 'rubin_two_node/sglang_graph_capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def inputs():
    return {'input_ids': [[5, 7] for _ in range(128)], 'source': {
        'dataset_sha256': 'a' * 64, 'tokenizer_model': '/models/Qwen3-30B-A3B',
        'selected_row_indices': list(range(128)), 'chat_template_kwargs': {}}}


class DurableController(unittest.TestCase):
    def test_plan_makes_no_subprocess_calls(self):
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['campaign_worker.py', '--platform', 'gb300', '--prepared-only']), \
                patch.object(worker.subprocess, 'run', side_effect=AssertionError('unexpected execution')), \
                patch.object(worker.subprocess, 'check_output', side_effect=AssertionError('unexpected remote read')), \
                contextlib.redirect_stdout(stream):
            worker.main()
        result = json.loads(stream.getvalue())
        self.assertEqual(result['job_id'], '2212644')
        self.assertTrue(result['prepared_only'])
        self.assertFalse(result['automatic_resume_or_resubmit'])

    def test_existing_main_evidence_blocks_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train-launch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'refusing automatic resubmission'):
                worker.no_main_evidence(root)

    def test_persistent_claim_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / 'worker-claim.json'
            worker.write(claim, {'owner': 'first'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                worker.write(claim, {'owner': 'second'}, exclusive=True)
            self.assertEqual(worker.read(claim), {'owner': 'first'})

    def test_orphaned_frozen_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root / 'frozen.json'; receipt = root / 'receipt.json'
            target.write_text(json.dumps(inputs()))
            with self.assertRaisesRegex(ValueError, 'Incomplete shared'):
                worker.frozen_record(target, receipt, capture)

    def test_shared_input_reuse_checks_file_and_workload_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root / 'frozen.json'; receipt = root / 'receipt.json'
            value = inputs(); target.write_text(json.dumps(value))
            request = capture.generation_request(value, 'fixture')
            record = {'status': 'PASS', 'bytes': target.stat().st_size, 'sha256': worker.sha(target),
                      'input_ids_sha256': request['input_ids_sha256'], 'workload_sha256': request['workload_sha256']}
            worker.write(receipt, record)
            self.assertEqual(worker.frozen_record(target, receipt, capture), record)
            record['workload_sha256'] = '0' * 64
            worker.write(receipt, record)
            with self.assertRaisesRegex(ValueError, 'workload mismatch'):
                worker.frozen_record(target, receipt, capture)
            target.write_text(target.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'receipt/hash mismatch'):
                worker.frozen_record(target, receipt, capture)

    def test_main_marker_is_checked_before_any_launch_or_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = worker.Worker(argparse.Namespace(platform='rubin', prepared_only=False))
            instance.root = Path(directory)
            (instance.root / 'main-worker-launch.json').write_text('{}')
            with patch.object(instance, 'verify_source', side_effect=AssertionError('source read')), \
                    patch.object(instance, 'command', side_effect=AssertionError('main resubmit')):
                with self.assertRaisesRegex(ValueError, 'refusing automatic resubmission'):
                    instance.run_main()


if __name__ == '__main__':
    unittest.main()
