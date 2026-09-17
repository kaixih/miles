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
import types
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


def replacement_options(**changes):
    return argparse.Namespace(platform='rubin', prepared_only=False, job_id='2213753',
        campaign_root=worker.BASE / '20260917-campaign-rubin-rerun-j2213753',
        source_manifest=worker.MANIFEST,
        prepare_helper=Path('/home/scratch.kaixih_ent/repo/miles-rubin-cu134/lab/rubin_trtllm_full/prepare_platform.py'),
        reuse_frozen_input_from=worker.CAMPAIGN, wait_until='2026-09-17T08:00:00Z', **changes)


class DurableController(unittest.TestCase):
    def test_replacement_allocation_uses_host_guard_and_actual_rounded_deadline(self):
        import prepare_platform
        instance = worker.Worker(replacement_options())
        instance.host_prepare = prepare_platform
        instance.prepare = types.SimpleNamespace(allocation_guard=lambda *a: self.fail('Frozen guard selected'))
        text = ('JobId=2213753 UserId=kaixih(28644) GroupId=dip(30) JobState=RUNNING '
                'NumNodes=1 AllocTRES=cpu=352,gres/gpu=4 NodeList=vr-nvl72-ts2-l11-038-c15 '
                'TimeLimit=08:00:00 StartTime=2026-09-17T04:23:10 EndTime=2026-09-17T12:23:11')
        with patch.object(worker.subprocess, 'check_output', return_value=text) as command, \
                patch.object(worker.time, 'time', return_value=worker.stamp('2026-09-17T04:30:00Z')):
            decision, original = instance.allocation()
            self.assertEqual(decision, 'ready')
            self.assertEqual(original['EndTime'], '2026-09-17T12:23:11')
            self.assertEqual(command.call_args.args[0][-1], '2213753')
            instance.original = original
            command.return_value = text.replace('12:23:11', '12:23:12')
            with self.assertRaisesRegex(ValueError, 'lease changed'): instance.allocation()

    def test_replacement_identity_is_separate_and_original_defaults_survive(self):
        new = worker.Worker(replacement_options())
        old = worker.Worker(argparse.Namespace(platform='gb300', prepared_only=True))
        self.assertEqual(new.job, '2213753')
        self.assertEqual(new.run_id, '20260917-rubin-j2213753-trtllm')
        self.assertEqual(new.manifest, worker.MANIFEST)
        self.assertNotEqual(new.work.parent.parent, old.work.parent.parent)
        self.assertEqual(old.job, '2212644')
        self.assertEqual(old.campaign, worker.CAMPAIGN)
        self.assertIsNone(old.prepare_helper)
        self.assertIsNone(old.reuse_inputs)
        for field, value in [('job_id', '../2213753'), ('campaign_root', worker.CAMPAIGN),
                             ('prepare_helper', None), ('reuse_frozen_input_from', None),
                             ('source_manifest', worker.BASE / 'unrelated/source-manifest.json')]:
            args = replacement_options(); setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError): worker.Worker(args)

    def test_frozen_input_copy_preserves_original_receipt_bytes_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory) / 'old'; target = Path(directory) / 'new'
            origin.mkdir(); target.mkdir()
            input_path = origin / 'frozen-token-input.json'
            input_path.write_text(json.dumps(inputs(), indent=1) + '\n')
            request = capture.generation_request(inputs(), 'fixture')
            record = {'status': 'PASS', 'source_run_id': 'original-gb300', 'bytes': input_path.stat().st_size,
                      'sha256': worker.sha(input_path), 'input_ids_sha256': request['input_ids_sha256'],
                      'workload_sha256': request['workload_sha256']}
            receipt = origin / 'frozen-token-input-receipt.json'
            receipt.write_text(json.dumps(record, indent=3) + '\n\n')
            result, proof = worker.copy_frozen_inputs(origin, target, capture)
            self.assertEqual(result, record)
            self.assertTrue(proof['byte_exact'])
            for name in proof['files']: self.assertEqual((origin / name).read_bytes(), (target / name).read_bytes())
            self.assertEqual(worker.copy_frozen_inputs(origin, target, capture)[0], record)
            (target / receipt.name).write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'differs from original bytes'):
                worker.copy_frozen_inputs(origin, target, capture)

    def test_replacement_plan_makes_no_remote_calls(self):
        stream = io.StringIO(); args = replacement_options()
        argv = ['campaign_worker.py', '--job-id', args.job_id, '--campaign-root', str(args.campaign_root),
                '--source-manifest', str(args.source_manifest), '--prepare-helper', str(args.prepare_helper),
                '--reuse-frozen-input-from', str(args.reuse_frozen_input_from)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stream), \
                patch.object(worker.subprocess, 'run', side_effect=AssertionError('execution')), \
                patch.object(worker.subprocess, 'check_output', side_effect=AssertionError('remote')):
            worker.main()
        self.assertEqual(json.loads(stream.getvalue())['job_id'], '2213753')

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
