"""CPU-only guards for durable claims, shared-input publication and plan mode."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
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
    def retry_options(self):
        args = replacement_options()
        args.attempt_tag = 'r2'
        args.campaign_root = worker.BASE / '20260917-campaign-rubin-j2213753-r2'
        args.source_manifest = args.campaign_root / 'source-manifest.json'
        args.source_manifest_sha256 = 'b' * 64
        args.reuse_node_models = Path('/tmp/miles-kaixih-j2213753-trtllm/models')
        return args

    def test_tagged_retry_is_fresh_and_requires_explicit_bound_manifest(self):
        args = self.retry_options(); new = worker.Worker(args)
        self.assertEqual(new.run_id, '20260917-rubin-j2213753-trtllm-r2')
        self.assertEqual(new.profile_id, 'rubin-j2213753-trtllm-r2-profile-v1')
        old = worker.Worker(replacement_options())
        self.assertNotEqual(new.root, old.root)
        self.assertNotEqual(new.work, old.work)
        self.assertEqual(new.reuse_inputs, worker.CAMPAIGN)
        for key, value in [('attempt_tag', '../r2'), ('job_id', None), ('source_manifest', worker.MANIFEST),
                           ('source_manifest_sha256', None), ('reuse_frozen_input_from', None),
                           ('reuse_node_models', Path('/tmp/miles-kaixih-j999-trtllm/models'))]:
            bad = self.retry_options(); setattr(bad, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): worker.Worker(bad)

    def test_retry_allows_new_commit_but_preserves_original_recipe_sources(self):
        original = {'git_commit': worker.REPLACEMENT_SOURCE_COMMIT,
                    'source_sha256': {name: 'a' * 64 for name in worker.RECIPE_FILES}}
        new = {'git_commit': 'b' * 40, 'source_sha256': dict(original['source_sha256'])}
        self.assertEqual(worker.retry_source_contract(new, original)['unchanged_recipe_sources'], original['source_sha256'])
        new['source_sha256'][worker.RECIPE_FILES[0]] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'original launcher'): worker.retry_source_contract(new, original)

    def test_retry_preparation_forwards_tag_manifest_hash_and_model_reuse(self):
        instance = worker.Worker(self.retry_options())
        instance.commit = 'c' * 40; instance.prepare_helper_sha = 'd' * 64
        with tempfile.TemporaryDirectory() as directory:
            instance.root = Path(directory)
            with patch.object(instance, 'image_ready', return_value={'image': 'registry/image@sha256:' + 'e' * 64}), \
                    patch.object(worker, 'sha', return_value='d' * 64), patch.object(instance, 'remaining', return_value=5*3600), \
                    patch.object(instance, 'command', side_effect=RuntimeError('stop before preparation executes')) as command:
                with self.assertRaisesRegex(RuntimeError, 'stop before'): instance.prepare_main()
            argv = command.call_args.args[1]
            for flag, value in [('--job-id', '2213753'), ('--attempt-tag', 'r2'),
                                ('--source-manifest-sha256', 'b' * 64),
                                ('--reuse-node-models', '/tmp/miles-kaixih-j2213753-trtllm/models')]:
                self.assertEqual(argv[argv.index(flag)+1], value)

    def test_retry_retention_forwards_same_job_and_attempt_tag(self):
        instance = worker.Worker(self.retry_options())
        with tempfile.TemporaryDirectory() as directory:
            instance.work = Path(directory) / 'worker'; instance.work.mkdir()
            helper = Path(directory) / 'retainer.py'; helper.write_text('# fixture\n')
            with patch.object(worker, 'RETENTION', helper), \
                    patch.object(instance, 'command', side_effect=RuntimeError('stop before retention executes')) as command:
                with self.assertRaisesRegex(RuntimeError, 'stop before'): instance.retain_and_profile({})
            argv = command.call_args.args[1]
            self.assertEqual(argv[argv.index('--job-id')+1], '2213753')
            self.assertEqual(argv[argv.index('--attempt-tag')+1], 'r2')

    def test_retry_source_verification_records_new_commit_and_rejects_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = root / 'source-r2'; source.mkdir()
            names = set(worker.RECIPE_FILES) | {'lab/rubin_trtllm_full/prepare_platform.py',
                    'lab/rubin_trtllm_full/profile_operator.py', 'lab/rubin_trtllm_full/profiling/run_profile.py'}
            hashes = {}
            for name in names:
                path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# fixture\n')
                hashes[name] = worker.sha(path)
            manifest = root / 'source-manifest.json'
            manifest.write_text(json.dumps({'git_commit': 'c' * 40, 'source_root': str(source), 'source_sha256': hashes}))
            original = root / 'original-manifest.json'
            original.write_text(json.dumps({'git_commit': worker.REPLACEMENT_SOURCE_COMMIT, 'source_sha256': hashes}))
            args = self.retry_options(); args.source_manifest_sha256 = worker.sha(manifest)
            instance = worker.Worker(args); instance.manifest = manifest; instance.prepare_helper = None
            stat = Path.stat
            def owned(path, *args, **kwargs):
                values = list(stat(path, *args, **kwargs)); values[4] = 28644
                return os.stat_result(values)
            with patch.object(Path, 'stat', owned), patch.object(worker, 'MANIFEST', original), \
                    patch.object(worker, 'module', return_value=types.SimpleNamespace()):
                proof = instance.verify_source()
                self.assertEqual(proof['git_commit'], 'c' * 40)
                self.assertEqual(proof['manifest_sha256'], args.source_manifest_sha256)
                self.assertEqual(proof['retry_recipe_contract']['baseline_commit'], worker.REPLACEMENT_SOURCE_COMMIT)
                manifest.write_text(manifest.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'): instance.verify_source()

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
