"""CPU-only plan and retention guards; no SSH, Ray, containers, or GPU work."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import datetime as dt
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("profile_operator_test_target", Path(__file__).with_name("profile_operator.py"))
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_both_default_plans_are_offline_and_use_4096(self):
        for platform in ("rubin", "gb300"):
            with self.subTest(platform=platform), patch.object(operator, "run", side_effect=AssertionError("No remote calls")), \
                    patch("sys.argv", ["operator", platform, "--run-id", "plan-test"]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                operator.main()
                plan = json.loads(out.getvalue())
                self.assertEqual(plan["mode"], "PLAN_ONLY_NO_REMOTE_CALLS")
                command = plan["replay_plan_argv"]
                self.assertEqual(command[command.index("--profile-max-tokens-per-gpu") + 1], "4096")
                self.assertEqual(command[command.index("--checkpoint-iteration") + 1], "9")
                self.assertIn("--print-only", command)
                self.assertEqual(plan["replay_workload"]["rollout_ids"], [10, 11])
                self.assertEqual(plan["replay_workload"]["rollouts"], 2)
                self.assertEqual(plan["replay_workload"]["optimizer_updates"], 8)
                self.assertEqual(plan["replay_workload"]["profile_step_start"], 1)
                self.assertEqual(plan["replay_workload"]["profile_step_end"], 2)
                self.assertEqual(plan["retention_margin_seconds"], 1200)
                self.assertIn("50rollouts/200updates", plan["completion_gates"])
                self.assertEqual(plan["config"]["input"], plan["config"]["runtime_output"] + "/inputs")
                self.assertEqual(plan["config"]["profile_checkpoint"], plan["config"]["raid_root"] + "/profile-checkpoint-9")
                mounts = plan["docker_argv"]
                self.assertIn(f'type=bind,src={plan["config"]["models"]},dst={operator.BASE}/models,readonly', mounts)
                self.assertIn(f'type=bind,src={plan["config"]["profile_checkpoint"]},dst=/profile-checkpoint,readonly', mounts)
        self.assertEqual(operator.RUNS["rubin"]["main_id"], "raysubmit_qCmmzUzbwHrBKAiU")
        self.assertEqual(operator.RUNS["gb300"]["main_id"], "raysubmit_W7k1mh2Y9Uh9wmmk")

    def test_both_main_runs_require_retained_local_evidence(self):
        for name in ("train_exit.json", "train-driver-exit.json"):
            (self.root / name).write_text('{"exit_code": 0}')
        for platform in ("rubin", "gb300"):
            c = {**operator.RUNS[platform], "platform": platform, "root": str(self.root)}
            with self.subTest(platform=platform), self.assertRaises(FileNotFoundError) as raised:
                operator.check_main(c)
            self.assertIn("artifact-retention-exit.json", str(raised.exception))

    def test_healthy_running_gb_job_still_blocks_preparation(self):
        for name in ("train_exit.json", "train-driver-exit.json", "artifact-retention-exit.json"):
            (self.root / name).write_text('{"exit_code": 0, "uid": 28644}')
        summary = SimpleNamespace(summarize_run=lambda *a: {"partial": False,
            "completed_training_rollouts": list(range(50)), "source_log_sha256": "0" * 64})
        loader = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
        c = {**operator.RUNS["gb300"], "platform": "gb300", "root": str(self.root)}
        with patch.object(operator.importlib.util, "spec_from_file_location", return_value=loader), \
                patch.object(operator.importlib.util, "module_from_spec", return_value=summary), \
                patch.object(operator, "jobs", return_value=[{"type": "SUBMISSION", "status": "RUNNING",
                    "submission_id": c["main_id"]}]), \
                self.assertRaisesRegex(RuntimeError, "active/unknown job"):
            operator.check_main(c)

    def manifest(self):
        checkpoint = self.root / "profile-checkpoint-9"
        files = {}
        for name, data in {
            "latest_checkpointed_iteration.txt": b"9",
            "iter_0000009/common.pt": b"common",
            "iter_0000009/.metadata": b"index",
            "rollout/global_dataset_state_dict_9.pt": b"cursor",
            "iter_0000009/__0_0.distcp": b"tensor shard",
        }.items():
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = {"bytes": len(data)}
            if not name.endswith(".distcp"):
                files[name]["sha256"] = hashlib.sha256(data).hexdigest()
        metadata = ["iter_0000009/common.pt", "rollout/global_dataset_state_dict_9.pt", "iter_0000009/.metadata"]
        identity = {"iteration": 9, "metadata": [{"path": n, "sha256": files[n]["sha256"]} for n in metadata],
                    "shards": [("iter_0000009/__0_0.distcp", len(b"tensor shard"))]}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        record = {"exit_code": 0, "rsync_exit_code": 0, "uid": 28644,
            "run_id": operator.RUNS["rubin"]["main_run"], "source_node": operator.RUNS["rubin"]["node"],
            "source_root": operator.RUNS["rubin"]["runtime_output"] + "/checkpoints",
            "destination_root": str(checkpoint), "iteration": 9,
            "verification": "rsync_transfer_plus_sizes_and_metadata_sha256",
            "source_checkpoint_id": fingerprint, "destination_checkpoint_id": fingerprint, "files": files}
        (self.root / operator.CHECKPOINT_RECORD).write_text(json.dumps(record))
        return checkpoint, record

    @contextlib.contextmanager
    def owned_files(self):
        original_stat = Path.stat
        def owned_stat(path, *args, **kwargs):
            values = list(original_stat(path, *args, **kwargs))
            values[4] = 28644
            return os.stat_result(values)
        with patch.object(Path, "stat", new=owned_stat):
            yield

    def verify(self, checkpoint):
        with patch.object(operator, "RUBIN_ROOT", str(self.root)), \
                patch.object(operator, "CHECKPOINT", str(checkpoint)), self.owned_files():
            return operator.check_checkpoint_retention()

    def test_verified_metadata_and_shard_size_manifest_is_accepted_without_tensor_hash(self):
        checkpoint, record = self.manifest()
        self.assertNotIn("sha256", record["files"]["iter_0000009/__0_0.distcp"])
        verified = self.verify(checkpoint)
        self.assertEqual(verified["checkpoint_id"], record["source_checkpoint_id"])
        self.assertEqual(verified["iteration"], 9)

    def test_metadata_change_and_unrecorded_shard_are_rejected(self):
        checkpoint, _ = self.manifest()
        (checkpoint / "iter_0000009/common.pt").write_bytes(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "metadata changed"):
            self.verify(checkpoint)
        (checkpoint / "iter_0000009/common.pt").write_bytes(b"common")
        (checkpoint / "iter_0000009/extra.distcp").write_bytes(b"unrecorded")
        with self.assertRaisesRegex(RuntimeError, "shard inventory"):
            self.verify(checkpoint)

    def test_wrong_source_and_symlink_shard_are_rejected(self):
        checkpoint, record = self.manifest()
        record["run_id"] = "old-run"
        (self.root / operator.CHECKPOINT_RECORD).write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "matching verified"):
            self.verify(checkpoint)
        record["run_id"] = operator.RUNS["rubin"]["main_run"]
        (self.root / operator.CHECKPOINT_RECORD).write_text(json.dumps(record))
        shard = checkpoint / "iter_0000009/__0_0.distcp"
        shard.unlink()
        shard.symlink_to(checkpoint / "iter_0000009/common.pt")
        with self.assertRaisesRegex(RuntimeError, "symlinked"):
            self.verify(checkpoint)

    def test_selected_fingerprint_matches_replay_helper_with_both_indexes(self):
        from lab.rubin_two_node.test_profile_qwen3_replay import P
        checkpoint, _ = self.manifest()
        (checkpoint / "iter_0000009/metadata.json").write_bytes(b"second index")
        with self.owned_files():
            inventory = operator._checkpoint_inventory(str(checkpoint), 9)
        self.assertEqual(inventory["id"], P._checkpoint_manifest(str(checkpoint), 9)["id"])
        self.assertEqual([x["path"] for x in inventory["metadata"]],
                         ["iter_0000009/common.pt", "rollout/global_dataset_state_dict_9.pt",
                          "iter_0000009/.metadata", "iter_0000009/metadata.json"])
        with self.owned_files(), self.assertRaises(FileNotFoundError):
            operator._checkpoint_inventory(str(checkpoint), 49)

    def staged_models(self):
        raid = self.root / "raid"
        node = operator.RUNS["rubin"]["node"]
        sources = {name: operator.BASE + "/models/" + name
                   for name in ("Qwen3-30B-A3B", "Qwen3-30B-A3B_torch_dist")}
        models = {}
        for name, source in sources.items():
            destination = raid / "models" / name
            payload = {"config.json": b"config", "weights.safetensors": b"HF shard"} if not name.endswith("_torch_dist") else {
                "latest_checkpointed_iteration.txt": b"release", "release/common.pt": b"common",
                "release/.metadata": b"index", "release/__0_0.distcp": b"reference shard"}
            files = {}
            for relative, data in payload.items():
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files[relative] = {"bytes": len(data)}
                if path.suffix not in (".safetensors", ".distcp"):
                    files[relative]["sha256"] = hashlib.sha256(data).hexdigest()
            models[name] = {"source_root": source, "destination_root": str(destination),
                            "files": files, "file_count": len(files), "total_bytes": sum(x["bytes"] for x in files.values()),
                            "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
        manifest = raid / "manifests/input-staging.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"schema_version": 1, "status": "complete", "uid": 28644, "gid": 30,
            "node": node, "raid_root": str(raid), "models": models,
            "verification": "shard_inventory_sizes_and_metadata_sha256"}))
        return raid, node, sources

    def test_staged_inputs_require_exact_inventory_and_metadata(self):
        raid, node, sources = self.staged_models()
        with self.owned_files():
            valid = operator._input_staging_summary(str(raid), node, sources)
        self.assertEqual(set(valid["models"]), set(sources))
        metadata = raid / "models/Qwen3-30B-A3B/config.json"
        metadata.write_bytes(b"tamper")
        with self.owned_files(), self.assertRaisesRegex(RuntimeError, "inventory or metadata"):
            operator._input_staging_summary(str(raid), node, sources)
        metadata.write_bytes(b"config")
        with self.owned_files(), self.assertRaisesRegex(RuntimeError, "mismatched completed"):
            operator._input_staging_summary(str(raid), "wrong-node", sources)

    def test_local_snapshot_must_match_durable_and_native_main_must_be_final49(self):
        raid, node, _ = self.staged_models()
        checkpoint, record = self.manifest()
        c = {**operator.config(SimpleNamespace(platform="rubin", run_id="test")),
             "raid_root": str(raid), "profile_checkpoint": str(checkpoint),
             "runtime_output": str(self.root / "native")}
        final = Path(c["runtime_output"]) / "checkpoints"
        for relative, data in {"latest_checkpointed_iteration.txt": b"49", "iter_0000049/common.pt": b"final",
                "iter_0000049/.metadata": b"finalindex", "iter_0000049/__0_0.distcp": b"finalshard",
                "rollout/global_dataset_state_dict_49.pt": b"finalcursor"}.items():
            path = final / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        def local_only(_config, code, _timeout):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                exec(compile(code, "<CPU-only node prerequisite fixture>", "exec"), {})
            return json.loads(out.getvalue())
        with self.owned_files(), patch.object(operator, "node_python", side_effect=local_only):
            result = operator.check_node_prerequisites(c, {"checkpoint_id": record["source_checkpoint_id"]})
            self.assertEqual(result["main_final_iteration"], 49)
            self.assertEqual(result["profile_iteration"], 9)
            with self.assertRaisesRegex(RuntimeError, "Local selected checkpoint differs"):
                operator.check_node_prerequisites(c, {"checkpoint_id": "0" * 64})
            (final / "latest_checkpointed_iteration.txt").write_text("9")
            with self.assertRaisesRegex(RuntimeError, "tracker does not equal"):
                operator.check_node_prerequisites(c, {"checkpoint_id": record["source_checkpoint_id"]})


class RetentionDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.clock = 1000.0
        self.c = {**operator.config(SimpleNamespace(platform='rubin', run_id='deadline-test')),
                  'durable': str(self.root), 'local': str(self.root / 'node'),
                  'lease': dt.datetime.fromtimestamp(5000, dt.timezone.utc).isoformat()}
        self.args = SimpleNamespace(retention_margin_seconds=600)
        self.job = {'submission_id': self.c['submission_id'], 'status': 'SUCCEEDED'}
        self.payload = b'actual fixture trace'
        self.files = {'replay/test.trace.gz': {'bytes': len(self.payload),
                      'sha256': hashlib.sha256(self.payload).hexdigest()}}
        self.timeouts = []

    @contextlib.contextmanager
    def clocked(self):
        with patch.object(operator.time, 'time', side_effect=lambda: self.clock), \
             patch.object(operator.time, 'monotonic', side_effect=lambda: self.clock), \
             patch.object(operator.signal, 'getitimer', return_value=(0, 0)), \
             patch.object(operator.signal, 'setitimer'), patch.object(operator.signal, 'signal'), \
             contextlib.redirect_stdout(io.StringIO()):
            yield

    def remote(self, _c, argv, timeout):
        self.timeouts.append((argv, timeout))
        if argv[:3] == ['docker', 'exec', self.c['name']]:
            if '-c' in argv:
                compile(argv[-1], '<capture script>', 'exec')
        self.clock += 20
        return '{}'

    def copy(self, argv, *, check, timeout):
        self.assertEqual(argv[0], 'rsync')
        self.timeouts.append(('rsync', timeout))
        target = self.root / 'node-output/replay/test.trace.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)
        self.clock += 100

    def test_one_budget_across_capture_both_hashes_copy_and_final_record(self):
        starts = []
        def manifest(_c, deadline):
            starts.append(deadline.remaining())
            self.clock += 150 if len(starts) == 1 else 100
            return self.files
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'remote', side_effect=self.remote), \
             patch.object(operator, 'manifest', side_effect=manifest), \
             patch.object(operator.subprocess, 'run', side_effect=self.copy), \
             OperatorTests.owned_files(self), patch.object(operator.os, 'getuid', return_value=28644):
            operator.retain(self.c, self.args)
        self.assertEqual(starts, [580, 330])
        self.assertIn(('rsync', 430), self.timeouts)
        record = json.loads((self.root / 'retention.json').read_text())
        audit = json.loads((self.root / 'retain-attempt.json').read_text())
        self.assertEqual(record['bytes'], len(self.payload))
        self.assertEqual(audit['actual_total_bytes'], len(self.payload))
        self.assertEqual(audit['deadline_timestamp'], 1600)
        self.assertEqual(audit['elapsed_seconds'], 370)
        self.assertEqual(audit['state'], 'verified')
        self.assertEqual(record['attempt_id'], audit['attempt_id'])
        self.assertEqual([x['name'] for x in audit['stages']], ['terminal_identity', 'capture_driver_evidence',
            'source_manifest_before', 'rsync', 'source_manifest_after', 'destination_hashes', 'verified_retention_record'])
        self.assertTrue(all(x['status'] == 'complete' for x in audit['stages']))

    def test_slow_source_manifest_exhausts_budget_before_copy_and_never_stops(self):
        target = self.root / 'node-output/previous.partial'
        target.parent.mkdir(); target.write_bytes(b'preserve')
        def manifest(_c, deadline):
            self.assertEqual(deadline.remaining(), 580)
            self.clock += 581
            return self.files
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'remote', side_effect=self.remote), patch.object(operator, 'manifest', side_effect=manifest), \
             patch.object(operator.subprocess, 'run') as copy, patch.object(operator, 'stop') as stop:
            with self.assertRaises(TimeoutError): operator.retain(self.c, self.args)
        copy.assert_not_called(); stop.assert_not_called()
        self.assertEqual(target.read_bytes(), b'preserve')
        self.assertFalse((self.root / 'retention.json').exists())
        self.assertNotEqual(json.loads((self.root / 'retain-attempt.json').read_text())['state'], 'verified')

    def test_slow_copy_preserves_partial_and_cannot_renew_budget_for_second_hash(self):
        def copy(argv, *, check, timeout):
            self.assertEqual(timeout, 580)
            self.copy(argv, check=check, timeout=timeout)
            self.clock += 500
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'remote', side_effect=self.remote), \
             patch.object(operator, 'manifest', return_value=self.files) as manifest, \
             patch.object(operator.subprocess, 'run', side_effect=copy), patch.object(operator, 'stop') as stop:
            with self.assertRaises(TimeoutError): operator.retain(self.c, self.args)
        self.assertEqual(manifest.call_count, 1); stop.assert_not_called()
        self.assertEqual((self.root / 'node-output/replay/test.trace.gz').read_bytes(), self.payload)
        self.assertFalse((self.root / 'retention.json').exists())

    def test_expired_lease_prevents_reads_or_mutations_for_retain_and_stop(self):
        self.c['lease'] = dt.datetime.fromtimestamp(1000, dt.timezone.utc).isoformat()
        for action in (operator.retain, operator.stop):
            with self.clocked(), patch.object(operator, 'profile_terminal') as check, \
                 patch.object(operator, 'remote') as remote:
                with self.assertRaises(TimeoutError): action(self.c, self.args)
            check.assert_not_called(); remote.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_recorded_lease_caps_requested_budget_and_clock_rollback_cannot_extend(self):
        self.c['lease'] = dt.datetime.fromtimestamp(1420, dt.timezone.utc).isoformat()
        with self.clocked():
            deadline = operator.OperationDeadline(self.c, 600, 'retain')
            self.assertEqual(deadline.deadline, 1300)
            self.clock += 250
            self.assertEqual(deadline.remaining(), 50)
            with patch.object(operator.time, 'time', return_value=1100):
                self.assertEqual(deadline.remaining(), 50)

    def test_expired_context_does_not_install_alarm_handler(self):
        with self.clocked():
            deadline = operator.OperationDeadline(self.c, 600, 'retain')
            self.clock += 601
            with patch.object(operator.signal, 'signal') as handler:
                with self.assertRaises(TimeoutError):
                    with deadline.enforce(): self.fail('expired context entered')
                handler.assert_not_called()

    def test_stop_can_use_shutdown_reserve_but_never_extend_lease(self):
        self.c['lease'] = dt.datetime.fromtimestamp(1100, dt.timezone.utc).isoformat()
        with self.clocked():
            with self.assertRaises(TimeoutError): operator.OperationDeadline(self.c, 600, 'retain')
            deadline = operator.OperationDeadline(self.c, 600, 'stop', lease_reserve_seconds=0)
            self.assertEqual(deadline.remaining(), 100)
            self.assertEqual(deadline.deadline, 1100)
            self.clock += 101
            with self.assertRaises(TimeoutError): deadline.remaining()

    def verified_records(self):
        record = {'state': 'verified', 'attempt_id': 'same', 'run_id': self.c['run_id'],
                  'terminal_ray_job': self.job, 'bytes': len(self.payload), 'files': self.files}
        (self.root / 'retention.json').write_text(json.dumps(record))
        (self.root / 'retain-attempt.json').write_text(json.dumps({'state': 'verified', 'attempt_id': 'same'}))

    def test_stop_hash_timeout_prevents_both_scoped_stop_commands(self):
        self.verified_records()
        def manifest(_c, deadline):
            self.clock += 601
            return self.files
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'manifest', side_effect=manifest), patch.object(operator, 'remote') as remote:
            with self.assertRaises(TimeoutError): operator.stop(self.c, self.args)
        remote.assert_not_called()
        self.assertFalse((self.root / 'operator-stopped.json').exists())

    def test_stop_requires_completed_retention_and_only_targets_exact_container(self):
        self.verified_records()
        (self.root / 'retain-attempt.json').write_text('{"state":"running","attempt_id":"same"}')
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'remote') as remote:
            with self.assertRaisesRegex(RuntimeError, 'completed verification'):
                operator.stop(self.c, self.args)
        remote.assert_not_called()
        self.verified_records()
        with self.clocked(), patch.object(operator, 'profile_terminal', return_value=self.job), \
             patch.object(operator, 'manifest', return_value=self.files), \
             patch.object(operator, 'remote', side_effect=self.remote):
            operator.stop(self.c, self.args)
        self.assertEqual([argv for argv, _ in self.timeouts], [
            ['docker', 'exec', self.c['name'], 'ray', 'stop', '--force'],
            ['docker', 'stop', '--time', '30', self.c['name']]])
        self.assertEqual(json.loads((self.root / 'stop-attempt.json').read_text())['state'], 'verified')

    def test_explicit600_allowed_default1200_preserved_below600_rejected(self):
        for value in (600, 1200):
            with patch('sys.argv', ['operator', 'rubin', '--run-id', 'test', '--retention-margin-seconds', str(value)]), \
                 contextlib.redirect_stdout(io.StringIO()) as output, patch.object(operator, 'run') as run:
                operator.main()
            run.assert_not_called()
            self.assertEqual(json.loads(output.getvalue())['retention_margin_seconds'], value)
        with patch('sys.argv', ['operator', 'rubin', '--run-id', 'test', '--retention-margin-seconds', '599']), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            operator.main()




class SavedStoppedAndInitialPolicyTests(unittest.TestCase):
    """Synthetic terminal evidence only; never query or control a real cluster."""
    owned_files = OperatorTests.owned_files

    def setUp(self):
        OperatorTests.setUp(self)
        self.c = {**operator.RUNS['rubin'], 'platform': 'rubin', 'root': str(self.root),
                  'initial_policy': False, 'run_id': 'fixture-profile'}
        self.expected = dict(operator.SAVED_STOP_PROVENANCE['rubin'])
        self.pins = {**operator.SAVED_STOP_PROVENANCE, 'rubin': self.expected}
        patcher = patch.object(operator, 'SAVED_STOP_PROVENANCE', self.pins)
        patcher.start(); self.addCleanup(patcher.stop)
        (self.root / 'node-output/logs').mkdir(parents=True)
        (self.root / 'logs').mkdir()
        self.training = {'started_utc': '2026-09-16T01:40:00+00:00',
                         'source_sha256': {'lab/rubin_two_node/watch_qwen3_run.py': operator.WATCHDOG_SOURCE_SHA256}}
        self.launch = {'pid': self.expected['pid'], 'command': ['fixture original watchdog']}
        self.write_record('train-launch.json', self.training)
        self.write_record('watchdog-launch.json', self.launch)
        for filename, key in [('train-launch.json', 'train_launch_sha256'), ('watchdog-launch.json', 'watchdog_launch_sha256')]:
            self.expected[key] = hashlib.sha256((self.root / filename).read_bytes()).hexdigest()
        (self.root / self.expected['driver_file']).write_text('fixture audited stage source')
        self.expected['driver_sha256'] = hashlib.sha256((self.root / self.expected['driver_file']).read_bytes()).hexdigest()
        self.main = {'type': 'SUBMISSION', 'submission_id': self.c['main_id'], 'status': 'STOPPED',
                     'entrypoint': 'python3 train.py --fixture', 'driver_exit_code': None,
                     'end_time': operator._timestamp('2026-09-16T05:22:11+00:00') * 1000,
                     'runtime_env': {'env_vars': {'RUBIN_RUN_ID': self.c['main_run']}}}
        self.expected['entrypoint_sha256'] = hashlib.sha256(self.main['entrypoint'].encode()).hexdigest()
        self.write_record('original-ray-job.json', self.main)
        self.watch = {'state': 'finished', 'terminal_status': 'STOPPED', 'ray_status': 'STOPPED',
            'run_id': self.c['main_run'], 'submission_id': self.c['main_id'], 'pid': self.expected['pid'],
            'armed_at': self.expected['armed_at'], 'ray_address': 'http://' + self.c['ip'] + ':28265',
            'soft_deadline_at': self.expected['soft'], 'deadline_at': self.expected['hard'],
            'lease_deadline_at': self.c['lease'], 'expected_rollouts': 50,
            'stop_reason': 'soft_deadline_checkpoint_confirmed', 'outcome': 'partial',
            'sentinel': '/run-output/checkpoint-now', 'save_dir': '/run-output/checkpoints',
            'sentinel_created': True, 'sentinel_seen': True, 'last_stop_response': True,
            'training_complete_verified': False, 'stop_request_count': 1, 'driver_exit_code': None,
            'checkpoint_baseline': {'iteration': 29, 'directory': '/run-output/checkpoints/iter_0000029', 'directory_exists': True},
            'latest_checkpoint': {'iteration': 39, 'directory': '/run-output/checkpoints/iter_0000039', 'directory_exists': True},
            'checkpoint_requested_at': '2026-09-16T05:20:02+00:00',
            'checkpoint_confirmed_at': '2026-09-16T05:22:10+00:00', 'completed_at': '2026-09-16T05:22:12+00:00'}
        self.retained = {'exit_code': 0, 'uid': 28644, 'retained_at': '2026-09-16T05:22:20+00:00'}
        self.exits = {'train_exit.json': 0, 'train-driver-exit.json': 0}
        self.metrics = {'partial': True, 'parse_errors': [], 'conflicts': [], 'source_log_sha256': 'a' * 64,
                        'completed_training_rollouts': list(range(40)), 'rows': [
            {'rollout_id': i, 'train_steps': [{'logged_id': 4*i+j, 'metrics': {'train/loss': .01, 'train/grad_norm': .2}}
                                             for j in range(4)]} for i in range(40)]}
        self.write_watch()
        self.write_record('artifact-retention-exit.json', self.retained)
        for filename, code in self.exits.items(): self.write_record(filename, {'exit_code': code})
        (self.root / 'logs/qwen3_train.log').write_text('fixture finite training metrics\n')

    def write_record(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def write_watch(self):
        self.write_record('node-output/watchdog.json', self.watch)
        stopping = {'state': 'stopping', 'submission_id': self.c['main_id'],
                    'stop_reason': self.watch['stop_reason'], 'latest_checkpoint': self.watch['latest_checkpoint']}
        (self.root / 'node-output/logs/watchdog.log').write_text(json.dumps(stopping) + '\n' + json.dumps(self.watch) + '\n')

    def evaluate(self):
        self.write_record('artifact-retention-exit.json', self.retained)
        for name, code in self.exits.items(): self.write_record(name, {'exit_code': code})
        with self.owned_files():
            return operator._saved_stop_evidence(self.c, self.main, self.metrics, self.exits, self.retained)

    def test_saved_stop_preserves_partial_counts_and_exact_checkpoint(self):
        result = self.evaluate()
        self.assertEqual(result['checkpoint_iteration'], 39)
        self.assertEqual(result['observed_optimizer_updates'], 160)
        self.assertEqual(result['saved_optimizer_updates'], 160)
        self.assertEqual(result['main_status'], 'STOPPED')
        self.assertTrue(result['partial'])
        self.assertEqual(len(result['completed_rollouts']), 40)

    def test_manual_hard_stale_and_modified_watchdog_refused(self):
        cases = [('stop_reason', 'manual'), ('stop_reason', 'hard_deadline'), ('pid', 999),
                 ('armed_at', '2026-09-16T05:19:00+00:00'), ('sentinel_created', False),
                 ('sentinel_seen', False), ('last_stop_response', False), ('stop_request_count', 0),
                 ('soft_deadline_at', '2026-09-16T05:30:00+00:00'), ('terminal_status', 'FAILED')]
        for key, bad in cases:
            old = self.watch[key]
            self.watch[key] = bad; self.write_watch()
            with self.subTest(key=key, bad=bad), self.assertRaises(RuntimeError): self.evaluate()
            self.watch[key] = old
        self.watch['latest_checkpoint'] = self.watch['checkpoint_baseline']; self.write_watch()
        with self.assertRaisesRegex(RuntimeError, 'advance'): self.evaluate()

    def test_missing_provenance_sentinel_transcript_and_retention_rejected(self):
        original = (self.root / 'watchdog-launch.json').read_text()
        self.write_record('watchdog-launch.json', {'pid': 999})
        with self.assertRaisesRegex(RuntimeError, 'provenance'): self.evaluate()
        (self.root / 'watchdog-launch.json').write_text(original)
        (self.root / 'node-output/logs/watchdog.log').write_text(json.dumps(self.watch) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'transcript'): self.evaluate()
        self.write_watch()
        self.retained['retained_at'] = '2026-09-16T05:22:00+00:00'
        with self.assertRaisesRegex(RuntimeError, 'chronology'): self.evaluate()

    def test_unexplained_exits_oom_nonfinite_and_missing_steps_refused(self):
        self.exits['train_exit.json'] = 143
        with self.assertRaisesRegex(RuntimeError, 'exit mapping'): self.evaluate()
        self.exits['train_exit.json'] = 0
        self.main['driver_exit_code'] = -15
        with self.assertRaisesRegex(RuntimeError, 'driver_exit_code'): self.evaluate()
        self.main['driver_exit_code'] = None
        log = self.root / 'logs/qwen3_train.log'; log.write_text('torch.OutOfMemoryError: allocation failed\n')
        with self.assertRaisesRegex(RuntimeError, 'OOM'): self.evaluate()
        log.write_text('normal\n')
        self.metrics['rows'][0]['train_steps'][0]['metrics']['train/grad_norm'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'): self.evaluate()
        self.metrics['rows'][0]['train_steps'][0]['metrics']['train/grad_norm'] = .2
        self.metrics['rows'][0]['train_steps'].pop(0)
        with self.assertRaisesRegex(RuntimeError, 'contiguous'): self.evaluate()

    def test_opt_in_is_required_and_initial_mode_skips_only_frozen9_gate(self):
        summary = SimpleNamespace(summarize_run=lambda *a: self.metrics)
        loader = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None))
        self.c['initial_policy'] = True
        with self.owned_files(), patch.object(operator, 'jobs', return_value=[self.main]), \
                patch.object(operator.importlib.util, 'spec_from_file_location', return_value=loader), \
                patch.object(operator.importlib.util, 'module_from_spec', return_value=summary), \
                patch.object(operator, 'check_checkpoint_retention', side_effect=AssertionError('No frozen9 dependency')), \
                patch.object(operator, 'check_node_prerequisites', return_value={'main_final_iteration': 39}) as node:
            with self.assertRaisesRegex(RuntimeError, 'not SUCCEEDED'): operator.check_main(self.c)
            result = operator.check_main(self.c, allow_saved_stopped=True)
            self.assertEqual(result['optimizer_updates'], 160)
            self.assertEqual(result['main_status'], 'STOPPED')
            self.assertIsNone(result['checkpoint_retention_record_sha256'])
            self.assertEqual(node.call_args.args[2], 39)
            self.main['status'] = 'FAILED'
            with self.assertRaisesRegex(RuntimeError, 'not SUCCEEDED'): operator.check_main(self.c, True)
            self.main['status'] = 'RUNNING'
            with self.assertRaisesRegex(RuntimeError, 'active/unknown'): operator.check_main(self.c, True)

    def test_initial_plan_is_offline_has_shared_manifest_and_no_checkpoint_mount(self):
        with patch.object(operator, 'run', side_effect=AssertionError('No remote calls')), \
                patch('sys.argv', ['operator', 'gb300', '--run-id', 'initial-fixture', '--initial-policy', '--allow-saved-stopped-main']), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            operator.main()
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['replay_workload']['rollout_ids'], [0, 1])
        self.assertEqual(plan['replay_workload']['original_training_horizon'], 50)
        self.assertTrue(plan['saved_stopped_opt_in'])
        self.assertIn('--initial-policy', plan['replay_plan_argv'])
        self.assertNotIn('--checkpoint-root', plan['replay_plan_argv'])
        mounts = ' '.join(plan['docker_argv'])
        self.assertIn('dst=/profile-input-manifest.json,readonly', mounts)
        self.assertNotIn('dst=/profile-checkpoint', mounts)
        command = operator.replay_command(plan['config'], 900, True, initial_model_id='b' * 64)
        self.assertEqual(command[command.index('--expected-initial-model-id') + 1], 'b' * 64)
        self.assertNotIn('--expected-checkpoint-id', command)

    def test_initial_input_id_matches_pair_and_native_main_iteration_is_still_required(self):
        self.c.update(raid_root='/raid/fixture', profile_checkpoint='/raid/fixture/profile-checkpoint-9', initial_policy=True)
        inputs = {'record_sha256': 'c' * 64, 'models': {
            'Qwen3-30B-A3B': {'fingerprint': 'a' * 64}, 'Qwen3-30B-A3B_torch_dist': {'fingerprint': 'b' * 64}}}
        with patch.object(operator, 'node_python', return_value={'inputs': inputs, 'main_final_checkpoint': {'id': 'd' * 64}}) as node:
            result = operator.check_node_prerequisites(self.c, None, 39, {'watchdog_sha256': 'e' * 64})
        expected = hashlib.sha256(json.dumps({n: m['fingerprint'] for n, m in inputs['models'].items()}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(result['initial_model_id'], expected)
        self.assertEqual(result['main_final_iteration'], 39)
        code = node.call_args.args[1]
        self.assertIn('Save sentinel still present', code)
        self.assertIn('/checkpoints\',39)', code)
        self.assertNotIn('result["selected_checkpoint"]', code)



    def test_second_main_gate_prevents_container_stop_if_new_active_job_appears(self):
        self.c.update(durable=str(self.root / 'profiles/new'), local='/raid/new', input='/tmp/inputs', name='new-profile')
        args = SimpleNamespace(max_runtime_seconds=900, retention_margin_seconds=600,
                               allow_stop_completed_main=True, allow_saved_stopped_main=True)
        evidence = {'main_status': 'STOPPED', 'log_sha256': 'a' * 64,
                    'node_prerequisites': {'main_final_iteration': 39}, 'completion_mode': 'original_watchdog_saved_STOPPED'}
        commands = []
        def remote(c, argv, *unused):
            commands.append(argv)
            if argv[:2] == ['docker', 'inspect']:
                return json.dumps([{'Config': {'Image': c['image'], 'User': '28644:30'}}])
            if argv[-1] == '--help':
                return '--dashboard-agent-grpc-port --dashboard-agent-listen-port --metrics-export-port'
            raise AssertionError('Unexpected process/container operation: ' + repr(argv))
        with self.owned_files(), patch.object(operator, 'allocation'), patch.object(operator, 'budget', return_value=900), \
                patch.object(operator, 'check_main', side_effect=[evidence, RuntimeError('Main Ray cluster still has an active/unknown job')]) as guard, \
                patch.object(operator, 'node_python', return_value={'uid': 28644}), \
                patch.object(operator, 'remote', side_effect=remote), \
                self.assertRaisesRegex(RuntimeError, 'active/unknown'):
            operator.prepare(self.c, args)
        self.assertEqual(guard.call_count, 2)
        self.assertTrue(all(call.kwargs['allow_saved_stopped'] for call in guard.call_args_list))
        self.assertFalse(any(argv[:2] == ['docker', 'stop'] for argv in commands))


if __name__ == "__main__":
    unittest.main()
