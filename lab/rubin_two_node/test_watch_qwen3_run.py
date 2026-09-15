"""CPU-only contract tests; no Ray server, GPU, or real watchdog is started."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("watch_qwen3_run", Path(__file__).with_name("watch_qwen3_run.py"))
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def job(run_id="run-test-50", status="RUNNING"):
    return {"type": "SUBMISSION", "submission_id": "exact-job", "status": status,
            "entrypoint": "python3 train.py --num-rollout 50",
            "runtime_env": {"env_vars": {"RUBIN_RUN_ID": run_id}}}


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.save = self.root / "checkpoints"
        self.save.mkdir()
        self.sentinel = self.root / "checkpoint-now"
        self.state = {"run_id": "run-test-50"}

    def checkpoint(self, iteration):
        (self.save / f"iter_{iteration:07d}").mkdir(exist_ok=True)
        (self.save / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    def test_target_requires_exact_run_and_training_entrypoint(self):
        self.assertTrue(watcher.is_target(job(), "run-test-50"))
        for mutation in [{"type": "DRIVER"}, {"entrypoint": "python3 diagnostic.py"},
                         {"entrypoint": "bash -c 'python3 train.py'"}, {"submission_id": ""},
                         {"runtime_env": {"env_vars": {"RUBIN_RUN_ID": "run-test-500"}}}]:
            self.assertFalse(watcher.is_target({**job(), **mutation}, "run-test-50"))

    def test_soft_stop_requires_both_advancing_tracker_and_consumed_sentinel(self):
        self.checkpoint(2)
        self.assertIsNone(watcher.stop_reason(self.state, self.sentinel, self.save, 10, 10, 30))
        self.checkpoint(3)
        self.assertFalse(watcher.checkpoint_ready(self.state, self.sentinel, self.save))
        self.sentinel.unlink()
        self.assertTrue(watcher.checkpoint_ready(self.state, self.sentinel, self.save))
        self.assertEqual(watcher.stop_reason(self.state, self.sentinel, self.save, 20, 10, 30),
                         "soft_deadline_checkpoint_confirmed")

    def test_sentinel_disappearance_alone_is_insufficient(self):
        self.checkpoint(2)
        watcher.request_checkpoint(self.state, self.sentinel, self.save, 10)
        self.sentinel.unlink()
        self.assertFalse(watcher.checkpoint_ready(self.state, self.sentinel, self.save))

    def test_first_checkpoint_zero_is_valid_but_missing_directory_is_not(self):
        watcher.request_checkpoint(self.state, self.sentinel, self.save, 10)
        (self.save / "latest_checkpointed_iteration.txt").write_text("0")
        self.sentinel.unlink()
        self.assertFalse(watcher.checkpoint_ready(self.state, self.sentinel, self.save))
        self.checkpoint(0)
        self.assertTrue(watcher.checkpoint_ready(self.state, self.sentinel, self.save))

    def test_hard_deadline_does_not_wait_for_checkpoint_or_create_sentinel(self):
        self.assertEqual(watcher.stop_reason(self.state, self.sentinel, self.save, 30, 10, 30), "hard_deadline")
        self.assertFalse(self.sentinel.exists())

    def test_existing_foreign_sentinel_is_never_overwritten(self):
        self.sentinel.write_text("other run")
        with self.assertRaises(FileExistsError):
            watcher.request_checkpoint(self.state, self.sentinel, self.save, 10)
        self.assertEqual(self.sentinel.read_text(), "other run")

    def test_completed_save_stops_only_locked_job_and_records_partial(self):
        self.checkpoint(2)
        args = SimpleNamespace(run_id="run-test-50", sentinel=self.sentinel, save_dir=self.save,
                               state_path=self.root / "state.json", soft_deadline=150, hard_deadline=200,
                               lease_deadline=220, num_rollout=50, log=None, poll_seconds=5)
        clock = [100]
        owner = self

        class API:
            address = "http://localhost:28265"
            lists = 0
            gets = 0
            stopped = []

            def list_jobs(self):
                self.lists += 1
                return [] if self.lists == 1 else [job()]

            def get_job(self, target):
                owner.assertEqual(target, "exact-job")
                self.gets += 1
                if self.gets == 2:
                    owner.checkpoint(3)
                    owner.sentinel.unlink()
                return job(status="STOPPED" if self.gets == 3 else "RUNNING")

            def stop_job(self, target):
                self.stopped.append(target)
                return True

        api = API()
        def advance(_):
            clock[0] = 160 if clock[0] == 100 else clock[0] + 10
        with patch.object(watcher.time, "time", side_effect=lambda: clock[0]), \
                patch.object(watcher.time, "sleep", side_effect=advance), patch("builtins.print"):
            self.assertEqual(watcher.watch(api, args), 0)
        result = json.loads(args.state_path.read_text())
        self.assertEqual(api.stopped, ["exact-job"])
        self.assertEqual(result["outcome"], "partial")
        self.assertFalse(result["training_complete_verified"])
        self.assertEqual(result["latest_checkpoint"]["iteration"], 3)
        self.assertEqual(result["expected_rollouts"], 50)


if __name__ == "__main__":
    unittest.main()
