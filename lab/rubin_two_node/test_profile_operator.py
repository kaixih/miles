"""CPU-only plan and retention guards; no SSH, Ray, containers, or GPU work."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
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
                self.assertIn("--print-only", command)
                self.assertEqual(plan["config"]["input"], plan["config"]["runtime_output"] + "/inputs")
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
        checkpoint = self.root / "checkpoints"
        files = {}
        for name, data in {
            "latest_checkpointed_iteration.txt": b"49",
            "iter_0000049/common.pt": b"common",
            "iter_0000049/.metadata": b"index",
            "rollout/global_dataset_state_dict_49.pt": b"cursor",
            "iter_0000049/__0_0.distcp": b"tensor shard",
        }.items():
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = {"bytes": len(data)}
            if not name.endswith(".distcp"):
                files[name]["sha256"] = hashlib.sha256(data).hexdigest()
        metadata = ["iter_0000049/common.pt", "rollout/global_dataset_state_dict_49.pt", "iter_0000049/.metadata"]
        identity = {"iteration": 49, "metadata": [{"path": n, "sha256": files[n]["sha256"]} for n in metadata],
                    "shards": [("iter_0000049/__0_0.distcp", len(b"tensor shard"))]}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        record = {"exit_code": 0, "rsync_exit_code": 0, "uid": 28644,
            "run_id": operator.RUNS["rubin"]["main_run"], "source_node": operator.RUNS["rubin"]["node"],
            "source_root": operator.RUNS["rubin"]["runtime_output"] + "/checkpoints",
            "destination_root": str(checkpoint), "iteration": 49,
            "verification": "rsync_transfer_plus_sizes_and_metadata_sha256",
            "source_checkpoint_id": fingerprint, "destination_checkpoint_id": fingerprint, "files": files}
        (self.root / "checkpoint-retention.json").write_text(json.dumps(record))
        return checkpoint, record

    def verify(self, checkpoint):
        original_stat = Path.stat
        def owned_stat(path, *args, **kwargs):
            values = list(original_stat(path, *args, **kwargs))
            values[4] = 28644
            return os.stat_result(values)
        with patch.object(operator, "RUBIN_ROOT", str(self.root)), \
                patch.object(operator, "CHECKPOINT", str(checkpoint)), patch.object(Path, "stat", new=owned_stat):
            return operator.check_checkpoint_retention()

    def test_verified_metadata_and_shard_size_manifest_is_accepted_without_tensor_hash(self):
        checkpoint, record = self.manifest()
        self.assertNotIn("sha256", record["files"]["iter_0000049/__0_0.distcp"])
        self.assertEqual(len(self.verify(checkpoint)), 64)

    def test_metadata_change_and_unrecorded_shard_are_rejected(self):
        checkpoint, _ = self.manifest()
        (checkpoint / "iter_0000049/common.pt").write_bytes(b"tamper")
        with self.assertRaisesRegex(RuntimeError, "metadata changed"):
            self.verify(checkpoint)
        (checkpoint / "iter_0000049/common.pt").write_bytes(b"common")
        (checkpoint / "iter_0000049/extra.distcp").write_bytes(b"unrecorded")
        with self.assertRaisesRegex(RuntimeError, "shard inventory"):
            self.verify(checkpoint)

    def test_wrong_source_and_symlink_shard_are_rejected(self):
        checkpoint, record = self.manifest()
        record["run_id"] = "old-run"
        (self.root / "checkpoint-retention.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "matching verified"):
            self.verify(checkpoint)
        record["run_id"] = operator.RUNS["rubin"]["main_run"]
        (self.root / "checkpoint-retention.json").write_text(json.dumps(record))
        shard = checkpoint / "iter_0000049/__0_0.distcp"
        shard.unlink()
        shard.symlink_to(checkpoint / "iter_0000049/common.pt")
        with self.assertRaisesRegex(RuntimeError, "symlinked"):
            self.verify(checkpoint)


if __name__ == "__main__":
    unittest.main()
