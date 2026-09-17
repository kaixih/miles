"""Checkpoint scope, stability, no-overwrite and bounded-copy tests; no SSH."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('trtllm_retain', Path(__file__).with_name('retain_final_checkpoint.py'))
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)


def config():
    run = '20260917-gb300-j2212644-trtllm'
    return {'platform': 'gb300', 'job_id': '2212644', 'run_id': run, 'node': 'gb300-nvl-012-compute04',
            'run_dir': str(R.BASE / run), 'node_run_dir': '/raid/tmp/miles-kaixih-j2212644-trtllm/run',
            'lease_deadline': '2026-09-17T09:30:21Z', 'sglang_moe_runner_backend': 'flashinfer_trtllm',
            'save_optimizer': False}


def checkpoint(root):
    (root / 'iter_0000049').mkdir(parents=True)
    (root / 'rollout').mkdir()
    files = {'latest_checkpointed_iteration.txt': b'49\n', 'iter_0000049/common.pt': b'common',
             'iter_0000049/.metadata': b'dcp-index', 'iter_0000049/__0_0.distcp': b'tensor-data',
             'rollout/global_dataset_state_dict_49.pt': b'rollout-state'}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return files


class RetentionTests(unittest.TestCase):
    def test_exact_job_campaign_model_only_and_node_path(self):
        c = config(); self.assertEqual(R.validate_config(c), c)
        changes = [{'job_id': '2212643'}, {'run_id': c['run_id'].replace('20260917', '20260918')},
                   {'run_dir': c['run_dir'] + '/other'}, {'node': 'node;touch bad'},
                   {'node_run_dir': '/mnt/cifs/home/scratch.kaixih_ent/run'},
                   {'node_run_dir': '/raid/tmp/j2212643/run'},
                   {'node_run_dir': '/raid/tmp/j22126440/run'},
                   {'node_run_dir': '/raid/tmp/notj2212644/run'},
                   {'node_run_dir': '/raid/tmp/j2212644/../run'},
                   {'save_optimizer': True}, {'save_optimizer': 0}, {'uid': 0},
                   {'sglang_moe_runner_backend': 'triton'}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                R.validate_config({**c, **change})

    def test_replacement_requires_explicit_job_and_exact_campaign_paths(self):
        c = config()
        replacement = json.loads(json.dumps(c).replace('gb300', 'rubin').replace('2212644', '2212999'))
        replacement['node'] = 'vr-nvl72-ts2-l11-038-c05'
        self.assertEqual(R.validate_config(replacement, job_id='2212999'), replacement)
        self.assertEqual(R.validate_config(replacement, job_id=2212999), replacement)
        with self.assertRaises(ValueError): R.validate_config(replacement)
        for job in ('', '0', '-1', '02212999', '2212999_0', '2212999;false', ' 2212999', '2212643'):
            with self.subTest(job=job), self.assertRaises(ValueError):
                R.validate_config(replacement, job_id=job)
        for field in ('job_id', 'run_id', 'run_dir', 'node_run_dir'):
            changed = {**replacement, field: replacement[field].replace('2212999', '2212643')}
            with self.subTest(field=field), self.assertRaises(ValueError):
                R.validate_config(changed, job_id='2212999')

    def test_default_original_campaigns_and_load_config_remain_compatible(self):
        for platform, job in R.JOBS.items():
            c = json.loads(json.dumps(config()).replace('gb300', platform).replace('2212644', job))
            with self.subTest(platform=platform):
                self.assertEqual(R.validate_config(c), c)
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / 'driver-config.json'; path.write_text(json.dumps(c))
                    self.assertEqual(R.load_config(path), c)
                    self.assertEqual(R.load_config(path, job_id=job), c)

    def test_cli_passes_explicit_job_to_config_and_preserves_no_job_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'driver-config.json'
            for job in (None, '2212999'):
                c = config()
                if job:
                    c = json.loads(json.dumps(c).replace('gb300', 'rubin').replace('2212644', job))
                path.write_text(json.dumps(c))
                argv = ['retain_final_checkpoint.py', '--config', str(path), '--execute']
                if job: argv.extend(['--job-id', job])
                with self.subTest(job=job), patch.object(sys, 'argv', argv), \
                        patch.object(R, 'execute', return_value=0) as execute:
                    self.assertEqual(R.main(), 0)
                    execute.assert_called_once_with(c)

    def test_allocation_owner_node_state_and_original_end_are_required(self):
        c = config(); now = R.timestamp('2026-09-17T05:00:00Z')
        raw = 'JobId=2212644 JobState=RUNNING NodeList=gb300-nvl-012-compute04 NumNodes=1 UserId=kaixih(28644) GroupId=dip(30) EndTime=2026-09-17T09:30:21'
        self.assertEqual(R.allocation_guard(c, raw, now)['lease_deadline'], c['lease_deadline'])
        for old, new in [('2212644', '123'), ('RUNNING', 'COMPLETED'), ('compute04', 'compute05'),
                         ('28644', '123'), ('dip(30)', 'dip(31)'), ('09:30:21', '10:30:21')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                R.allocation_guard(c, raw.replace(old, new), now)

    def test_main_completion_requires_both_zero_exits_and_model_only_plan(self):
        with tempfile.TemporaryDirectory() as d, patch.object(R.helpers, 'UID', os.getuid()):
            root = Path(d).resolve(); c = {**config(), 'run_dir': str(root)}
            original = {
                'train-driver-exit.json': {'run_id': c['run_id'], 'exit_code': 0},
                'train_exit.json': {'exit_code': 0},
                'cudagraph-main-plan.json': {'run_id': c['run_id'], 'recipe': {
                    'save_optimizer': False, 'rollouts': 50, 'optimizer_updates': 200}},
            }
            for name, data in original.items(): (root / name).write_text(json.dumps(data))
            self.assertEqual(R.main_completion(c)['driver_exit']['exit_code'], 0)
            for name, change in [
                ('train_exit.json', {'exit_code': 1}),
                ('train-driver-exit.json', {'run_id': 'another-run', 'exit_code': 0}),
                ('cudagraph-main-plan.json', {'run_id': c['run_id'], 'recipe': {
                    'save_optimizer': True, 'rollouts': 50, 'optimizer_updates': 200}}),
            ]:
                (root / name).write_text(json.dumps(change))
                with self.subTest(name=name), self.assertRaises(ValueError): R.main_completion(c)
                (root / name).write_text(json.dumps(original[name]))

    def test_transfer_deadline_leaves_original_lease_margin_and_final_verification(self):
        c = config(); lease = R.timestamp(c['lease_deadline'])
        self.assertEqual(R.operation_deadline(c, lease - 3600), lease - 1800)
        self.assertEqual(R.operation_deadline(c, lease - 600), lease - 120)
        with self.assertRaises(ValueError): R.operation_deadline(c, lease - 150)
        with self.assertRaises(TimeoutError): R.remaining(100, now=101)
        command = R.rsync_command(c, '/nfs/new.partial', '/nfs/files.txt', 400, now=100)
        self.assertIn('--rsync-path=timeout --signal=TERM --kill-after=5s 300s rsync', command)
        self.assertIn('gb300-nvl-012-compute04:/raid/tmp/miles-kaixih-j2212644-trtllm/run/checkpoints/', command)
        self.assertNotIn('--delete', command)
        self.assertFalse(any('/mnt/cifs' in arg for arg in command))

    def test_inventory_copies_rollout_metadata_and_rejects_same_size_source_mutation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(R.helpers, 'UID', os.getuid()):
            root = Path(d).resolve() / 'checkpoint'; files = checkpoint(root)
            manifest = R.helpers.inventory(root, iteration=49)
            before = {'root': str(root), 'inventory': manifest, 'file_state': R.helpers.file_state(root, manifest)}
            self.assertEqual(R.validate_snapshot(before), sum(map(len, files.values())))
            self.assertIn('rollout/global_dataset_state_dict_49.pt', manifest['files'])
            self.assertIn('sha256', manifest['files']['iter_0000049/.metadata'])
            self.assertNotIn('sha256', manifest['files']['iter_0000049/__0_0.distcp'])
            shard = root / 'iter_0000049/__0_0.distcp'; old = shard.stat()
            shard.write_bytes(b'x' * old.st_size); os.utime(shard, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000))
            after = {'root': str(root), 'inventory': R.helpers.inventory(root, iteration=49),
                     'file_state': R.helpers.file_state(root, manifest)}
            self.assertEqual(before['inventory'], after['inventory'])
            with self.assertRaisesRegex(ValueError, 'changed'): R.stable_source(before, after)
            (root / 'latest_checkpointed_iteration.txt').write_text('48\n')
            with self.assertRaisesRegex(ValueError, 'Tracker'): R.helpers.inventory(root, iteration=49)

    def test_inventory_rejects_symlink_and_missing_rollout_state(self):
        with tempfile.TemporaryDirectory() as d, patch.object(R.helpers, 'UID', os.getuid()):
            root = Path(d).resolve() / 'checkpoint'; checkpoint(root)
            state = root / 'rollout/global_dataset_state_dict_49.pt'; state.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)): R.helpers.inventory(root, iteration=49)
            state.write_text('restored'); shard = root / 'iter_0000049/__0_0.distcp'
            shard.unlink(); shard.symlink_to(state)
            with self.assertRaises(ValueError): R.helpers.inventory(root, iteration=49)

    def test_local_destination_matches_json_transported_remote_inventory(self):
        with tempfile.TemporaryDirectory() as d, patch.object(R.helpers, 'UID', os.getuid()):
            root = Path(d).resolve() / 'checkpoint'; checkpoint(root)
            remote = json.loads(json.dumps(R.helpers.inventory(root, iteration=49)))
            self.assertEqual(R.verify_destination(root, remote), remote)
            (root / 'rollout/global_dataset_state_dict_49.pt').write_text('different-state')
            with self.assertRaisesRegex(ValueError, 'differ'): R.verify_destination(root, remote)

    def test_bounded_copy_terminates_only_its_new_process_and_preserves_partial(self):
        with tempfile.TemporaryDirectory() as d:
            partial = Path(d) / 'partial'; partial.mkdir(); (partial / 'kept').write_text('partial data')
            with self.assertRaisesRegex(TimeoutError, 'partial destination retained'):
                R.helpers.copy_process([sys.executable, '-c', 'import time;time.sleep(30)'],
                                       time.time() + .15, Path(d) / 'copy.log')
            self.assertEqual((partial / 'kept').read_text(), 'partial data')

    def test_nfs_destination_reservation_never_reuses_or_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d).resolve() / 'final-checkpoint'
            marker = R.reserve_destination(destination, 'my-run')
            (destination / 'partial-shard').write_text('keep')
            with self.assertRaises(FileExistsError): R.reserve_destination(destination, 'another-run')
            self.assertEqual(json.loads((destination / R.INCOMPLETE_MARKER).read_text()), marker)
            self.assertEqual((destination / 'partial-shard').read_text(), 'keep')

    def test_only_own_marker_is_removed_after_inventory_validation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(R.helpers, 'UID', os.getuid()):
            root = Path(d).resolve() / 'final-checkpoint'
            marker = R.reserve_destination(root, 'my-run'); checkpoint(root)
            expected = json.loads(json.dumps(R.helpers.inventory(root, iteration=49)))
            self.assertEqual(R.verify_destination(root, expected, marker), expected)
            altered = {**marker, 'invocation_id': 'changed'}
            with self.assertRaisesRegex(ValueError, 'changed'): R.finish_destination(root, altered)
            self.assertTrue((root / R.INCOMPLETE_MARKER).exists())
            R.finish_destination(root, marker)
            self.assertFalse((root / R.INCOMPLETE_MARKER).exists())
            self.assertEqual(R.verify_destination(root, expected), expected)

    def test_remote_program_uses_inert_metadata_helper_and_compiles(self):
        source = R.source_program(config()); compile(source, 'remote-checkpoint-inventory', 'exec')
        self.assertIn('pickletools.genops', source)
        self.assertNotIn('pickle.loads(', source)
        self.assertNotIn('torch.load(', source)
        self.assertIn('iteration=49', source)


if __name__ == '__main__':
    unittest.main()
