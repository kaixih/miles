"""CPU-only handoff, one-writer, deadline, and publication regressions."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tempfile
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab.rubin_two_node.test_retain_profile_checkpoint import expected_ownership

spec = importlib.util.spec_from_file_location('fallback_test', Path(__file__).with_name('copy_gb_checkpoint_parallel.py'))
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)


def identity(pid, worker=False):
    argv = ['python3', 'retain_profile_checkpoint.py', '--execute-retention'] if worker else ['rsync', '--server', '.', F.GB_PARTIAL]
    return {'pid': pid, 'start_ticks': 100 + pid, 'uid': F.UID, 'argv': argv,
            'cmdline_sha256': hashlib.sha256(b'\0'.join(x.encode() for x in argv) + b'\0').hexdigest()}


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.raid = self.root / 'raid'
        self.partial = self.raid / 'profile-checkpoint-9.partial'
        self.final = self.raid / 'profile-checkpoint-9'
        self.partial.mkdir(parents=True)
        self.files = {'latest_checkpointed_iteration.txt': b'9\n',
                      'iter_0000009/common.pt': b'common state',
                      'iter_0000009/.metadata': b'index',
                      'rollout/global_dataset_state_dict_9.pt': b'cursor'}
        self.files.update({f'iter_0000009/__{i}_0.distcp': b'tensor' * (i + 1) for i in range(4)})
        for name, data in self.files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with expected_ownership():
            self.manifest = F.inventory(self.source)
        for key, value in {'ROOT': str(self.root), 'DEST': str(self.source), 'GB_RAID': str(self.raid),
                           'GB_PARTIAL': str(self.partial), 'GB_DEST': str(self.final),
                           'EXPECTED_ID': self.manifest['id']}.items():
            patcher = patch.object(F, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.budget = F.Budget(time.time() + 600)
        state = {'pid': 111, 'phase': 'staging_durable_checkpoint_to_gb_raid'}
        raw = json.dumps(state).encode()
        (self.root / F.STATE_NAME).write_bytes(raw)
        self.handoff = {'schema_version': 1, 'authorization': F.AUTHORIZATION, 'reviewed': True,
                        'reviewed_at': F.utc(), 'reviewer': 'CPU fixture reviewer',
                        'source_root': F.DEST, 'destination_root': F.GB_PARTIAL,
                        'checkpoint_id': F.EXPECTED_ID, 'deadline_utc': '2026-09-16T05:40:00+00:00',
                        'original_worker_host': F.LOCAL_HOST, 'gb_host': F.GB_NODE,
                        'original_worker_terminal': True, 'original_gb_copy_terminal': True,
                        'original_worker': identity(111, True),
                        'original_local_copy_processes': [identity(222)],
                        'original_gb_copy_processes': [identity(333)],
                        'original_state_sha256': hashlib.sha256(raw).hexdigest()}

    def record(self, value=None):
        raw = json.dumps(value or self.handoff).encode()
        path = self.root / 'handoff.json'
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_default_plan_performs_no_inspection_or_subprocess(self):
        with patch.object(F, 'execute', side_effect=AssertionError('execution forbidden')), \
                patch.object(F.subprocess, 'run', side_effect=AssertionError('no subprocess')), \
                patch.object(F, 'proc_snapshot', side_effect=AssertionError('no process inspection')), \
                redirect_stdout(io.StringIO()) as out:
            F.main([])
        self.assertEqual(json.loads(out.getvalue())['mode'], 'plan_only')

    def test_exact_reviewed_handoff_and_state_are_required(self):
        with expected_ownership():
            self.assertEqual(F.load_handoff(*self.record(), self.budget)['original_worker']['pid'], 111)
            for key, value in [('reviewed', False), ('original_gb_copy_terminal', False),
                               ('original_worker_terminal', False), ('original_gb_copy_processes', []),
                               ('checkpoint_id', '0' * 64), ('deadline_utc', '2026-09-16T06:00:00+00:00')]:
                bad = {**self.handoff, key: value}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    F.load_handoff(*self.record(bad), self.budget)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                F.load_handoff(self.record()[0], '0' * 64, self.budget)
            path, digest = self.record()
            (self.root / F.STATE_NAME).write_text('{"pid":111,"phase":"complete"}')
            with self.assertRaisesRegex(ValueError, 'state differs'):
                F.load_handoff(path, digest, self.budget)

    def test_exact_live_worker_and_idle_receiver_rejected(self):
        observed = {**identity(111, True), 'state': 'S', 'writers': []}
        with self.assertRaisesRegex(ValueError, 'still live'):
            F.check_terminal(identity(111, True), {111: observed})
        observed['start_ticks'] += 1
        self.assertTrue(F.check_terminal(identity(111, True), {111: observed})['terminal'])
        receiver = {**identity(333), 'state': 'S', 'writers': []}
        with self.assertRaisesRegex(ValueError, 'transfer/writer'):
            F.no_writers({333: receiver})
        receiver.update(argv=['unrelated'], writers=[str(self.partial / 'iter_0000009/__0_0.distcp')])
        with self.assertRaises(ValueError):
            F.no_writers({333: receiver})

    def test_two_fresh_writer_scans_are_required(self):
        receiver = {**identity(333), 'state': 'S', 'writers': []}
        with patch.object(F, 'proc_snapshot', side_effect=[{}, {333: receiver}]), \
                patch.object(F.time, 'sleep'), self.assertRaises(ValueError):
            F.guarded_processes([identity(222)], self.budget)

    def test_owned_partial_bytes_preserved_completed_or_symlink_refused(self):
        target = self.partial / 'iter_0000009/__0_0.distcp'
        target.parent.mkdir()
        target.write_bytes(b'ten')
        with expected_ownership():
            summary = F.inspect_partial(self.manifest)
            self.assertEqual(summary['partial_bytes'], 3)
            self.assertEqual(target.read_bytes(), b'ten')
            (self.partial / 'link').symlink_to(self.source)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                F.inspect_partial(self.manifest)
            (self.partial / 'link').unlink()
            target.write_bytes(b'x' * 100)
            with self.assertRaisesRegex(ValueError, 'oversized'):
                F.inspect_partial(self.manifest)
            self.final.mkdir()
            with self.assertRaises(FileExistsError):
                F.inspect_partial(self.manifest)

    def test_durable_record_and_four_shard_fingerprint_are_required(self):
        record = {'destination_root': F.DEST, 'destination_checkpoint_id': F.EXPECTED_ID,
                  'source_checkpoint_id': F.EXPECTED_ID, 'uid': F.UID, 'rsync_exit_code': 0,
                  'files': self.manifest['files']}
        path = self.root / 'checkpoint-retention-profile-9.json'
        path.write_text(json.dumps(record))
        with expected_ownership():
            self.assertEqual(F.verify_source(self.budget)[0], self.manifest)
            record['rsync_exit_code'] = 1
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'has not verified'):
                F.verify_source(self.budget)

    def test_four_parallel_one_file_one_writer_no_retry(self):
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        seen = []
        def transfer(names, log, budget):
            with lock:
                seen.extend(names)
            barrier.wait(timeout=3)
            return {'paths': names, 'exit_code': 0}
        with patch.object(F, 'transfer', side_effect=transfer):
            result = F.copy_shards(self.manifest, self.root, self.budget)
        self.assertEqual(len(result), 4)
        self.assertEqual(sorted(seen), sorted(name for name, _ in self.manifest['shards']))
        self.assertEqual(len(set(seen)), 4)
        with patch.object(F, 'transfer', side_effect=RuntimeError('failure')) as transfer, self.assertRaises(RuntimeError):
            F.copy_shards(self.manifest, self.root, self.budget)
        self.assertLessEqual(transfer.call_count, 4)

    def test_rsync_resume_no_delete_or_new_tensor_hash(self):
        argv = F.transfer_argv([self.manifest['shards'][0][0]], self.budget)
        for required in ('--partial', '--no-whole-file', '--relative'):
            self.assertIn(required, argv)
        for forbidden in ('--delete', '--inplace', '--append', '--checksum'):
            self.assertNotIn(forbidden, argv)
        code = shlex.split(F.receiver_command())[2]
        self.assertIn(f'{F.DEADLINE!r}-time.time()-5', code)
        self.assertIn("'--server'", code)
        self.assertIn(F.GB_PARTIAL, code)
        for name in ('../foreign', '/foreign', '-arg'):
            with self.assertRaises(ValueError):
                F.transfer_argv([name], self.budget)

    def test_receiver_uses_absolute_deadline_after_ssh_delay(self):
        code = shlex.split(F.receiver_command())[2]
        with patch.object(F.os, 'getuid', return_value=F.UID), patch.object(F.os, 'getgid', return_value=30), \
                patch.object(F.sys, 'argv', ['receiver', '--server', '.', F.GB_PARTIAL]), \
                patch.object(F.time, 'time', return_value=F.DEADLINE + 1), \
                patch.object(F.os, 'execvp', side_effect=AssertionError('must not launch')) as execute, \
                self.assertRaisesRegex(AssertionError, 'deadline expired'):
            exec(compile(code, '<receiver fixture>', 'exec'), {})
        execute.assert_not_called()

    def test_slow_validation_cannot_reset_overall_deadline(self):
        with patch.object(F.time, 'time', return_value=100), patch.object(F.time, 'monotonic', return_value=10):
            budget = F.Budget(200)
        with patch.object(F.time, 'time', return_value=170), patch.object(F.time, 'monotonic', return_value=80):
            self.assertEqual(budget.remaining(), 30)
            self.assertEqual(F.transfer_argv(['iter_0000009/__0_0.distcp'], budget)[3], '25')
        with patch.object(F.time, 'time', return_value=199), patch.object(F.time, 'monotonic', return_value=111), \
                self.assertRaises(TimeoutError):
            budget.remaining()

    def test_expired_deadline_prevents_writes_and_commands(self):
        expired = F.Budget(time.time() - 1)
        with patch.object(F, 'Budget', return_value=expired), \
                patch.object(F.subprocess, 'run', side_effect=AssertionError('must not launch')), \
                self.assertRaises(TimeoutError):
            F.execute('absent', '0' * 64)
        self.assertFalse((self.root / F.AUDIT_NAME).exists())
        with self.assertRaises(TimeoutError):
            F.write_json(self.root / 'no-write.json', {}, expired)
        self.assertFalse((self.root / 'no-write.json').exists())

    def test_copy_failure_keeps_partial_and_never_publishes(self):
        marker = self.partial / 'marker'
        marker.write_text('preserved failure evidence')
        remote_calls = []
        def remote(operation, *_):
            remote_calls.append(operation)
            return {'fixture': True}
        with expected_ownership(), patch.object(F, 'Budget', return_value=self.budget), \
                patch.object(F.os, 'getuid', return_value=F.UID), patch.object(F.os, 'getgid', return_value=30), \
                patch.object(F.socket, 'gethostname', return_value=F.LOCAL_HOST), \
                patch.object(F, 'load_handoff', return_value=copy.deepcopy(self.handoff)), \
                patch.object(F, 'guarded_processes', return_value=[]), \
                patch.object(F, 'verify_source', return_value=(self.manifest, {}, 'hash')), \
                patch.object(F, 'remote', side_effect=remote), \
                patch.object(F, 'transfer', return_value={'exit_code': 0}), \
                patch.object(F, 'copy_shards', side_effect=RuntimeError('copy failed')), \
                redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'copy failed'):
            F.execute('reviewed', 'a' * 64)
        self.assertEqual(remote_calls, ['inspect', 'prepare'])
        self.assertEqual(marker.read_text(), 'preserved failure evidence')
        self.assertFalse(self.final.exists())
        self.assertFalse((self.root / F.STAGE_NAME).exists())
        self.assertEqual(json.loads((self.root / F.AUDIT_NAME / 'state.json').read_text())['status'], 'failed')

    def test_near_deadline_cannot_rename_or_claim_completion(self):
        for name, data in self.files.items():
            path = self.partial / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        handoff = {**self.handoff, '_sha256': 'a' * 64}
        (self.raid / '.checkpoint9-parallel-fallback.lock').write_text(json.dumps({'handoff_sha256': 'a' * 64}))
        source = F.remote_source('publish', handoff, self.manifest)
        prefix, body = source.split('\nif os.getuid() != UID', 1)
        ns = {}
        exec(compile(prefix, '<receiver definitions>', 'exec'), ns)
        ns['proc_snapshot'] = lambda: {}
        near_deadline = F.Budget(time.time() + 8)
        ns['Budget'] = lambda: near_deadline
        with expected_ownership(), patch.object(F.os, 'getuid', return_value=F.UID), \
                patch.object(F.os, 'getgid', return_value=30), patch.object(F.socket, 'gethostname', return_value=F.GB_NODE), \
                patch.object(F.time, 'sleep'), self.assertRaisesRegex(TimeoutError, 'rename and completion'):
            exec(compile('\nif os.getuid() != UID' + body, '<receiver deadline fixture>', 'exec'), ns)
        self.assertTrue(self.partial.is_dir())
        self.assertFalse(self.final.exists())

    def test_standalone_receiver_publish_checks_complete_inventory(self):
        for name, data in self.files.items():
            path = self.partial / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        handoff = {**self.handoff, '_sha256': 'a' * 64}
        lock = self.raid / '.checkpoint9-parallel-fallback.lock'
        lock.write_text(json.dumps({'handoff_sha256': 'a' * 64}))
        source = F.remote_source('publish', handoff, self.manifest)
        prefix, body = source.split('\nif os.getuid() != UID', 1)
        ns = {}
        exec(compile(prefix, '<receiver definitions>', 'exec'), ns)
        ns['proc_snapshot'] = lambda: {}
        ns['Budget'] = lambda: self.budget
        with expected_ownership(), patch.object(F.os, 'getuid', return_value=F.UID), \
                patch.object(F.os, 'getgid', return_value=30), patch.object(F.socket, 'gethostname', return_value=F.GB_NODE), \
                patch.object(F.time, 'sleep'), redirect_stdout(io.StringIO()) as output:
            exec(compile('\nif os.getuid() != UID' + body, '<receiver publish fixture>', 'exec'), ns)
        self.assertEqual(json.loads(output.getvalue())['checkpoint_id'], self.manifest['id'])
        self.assertTrue(self.final.is_dir())
        self.assertFalse(self.partial.exists())
        self.assertEqual((self.final / 'iter_0000009/__0_0.distcp').read_bytes(), self.files['iter_0000009/__0_0.distcp'])


if __name__ == '__main__':
    unittest.main()
