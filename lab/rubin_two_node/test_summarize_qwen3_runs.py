"""Offline collector contracts; no Ray, packages, or GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).with_name("summarize_qwen3_runs.py"))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def event(kind, index, metrics, second=0):
    return f"\x1b[36m(Actor pid=123)\x1b[0m [2026-09-15 20:00:{second:02d}.100 rank0] x.py:1 - {kind} {index}: {metrics}\n"


class CollectorTests(unittest.TestCase):
    def collect(self, text, **metadata):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.log"
            path.write_text(text)
            return summary.summarize_run("rubin", path, metadata)

    def test_four_updates_dedup_and_two_eval_phases_survive(self):
        text = event("eval", 0, {"eval/gsm8k": 0.4}, 0)
        text += event("perf", 0, {"rollout/num_training_samples": 2048, "perf/rollout_time": 30}, 1)
        for step in range(4):
            line = event("step", step, {"train/step": step, "train/grad_norm": 0.1}, step + 2)
            text += line * 2
        text += event("perf", 0, {"perf/train_time": 10, "perf/train_wait_time": 40, "perf/step_time": 50}, 8)
        text += event("eval", 0, {"eval/gsm8k": 0.5}, 9)
        run = self.collect(text, expected_rollouts=1, status="SUCCEEDED")
        row = run["rows"][0]
        self.assertFalse(run["partial"])
        self.assertEqual(run["duplicate_train_lines_removed"], 4)
        self.assertEqual(len(row["train_steps"]), 4)
        self.assertEqual([e["weight_phase"] for e in row["eval"]], ["before_this_update", "after_this_update"])
        self.assertEqual(row["step_timer_identity_error_seconds"], 0)
        report = summary.report([run])
        self.assertEqual(report["shared_metric_observations"]["train/grad_norm"]["rubin"], 4)
        self.assertEqual(report["shared_metric_observations"]["eval/gsm8k"]["rubin"], 2)

    def test_timer_scope_throughput_and_profile_exclusion(self):
        text = event("rollout", 1, {"rollout/response_lengths": 100, "rollout/total_lengths": 120})
        text += event("perf", 1, {"rollout/num_training_samples": 8, "perf/rollout_time": 10,
                                  "perf/tokens_per_gpu_per_sec": 20})
        text += event("step", 1, {"train/step": 1, "train/grad_norm": 1})
        text += event("perf", 1, {"perf/train_wait_time": 11, "perf/train_time": 5,
                                  "perf/actor_train_time": 3, "perf/log_probs_time": 1, "perf/step_time": 16})
        meta = dict(optimizer_steps_per_rollout=1, expected_rollouts=2)
        unknown = self.collect(text, **meta)
        self.assertEqual(unknown["unprofiled_stage_statistics"]["train"]["count"], 0)
        run = self.collect(text, **meta, profiling_coverage_known=True)
        self.assertEqual(run["unprofiled_stage_statistics"]["train"]["mean_seconds"], 5)
        self.assertEqual(run["rows"][0]["common"]["retained_input_output_tokens_per_gpu_miles_step_second"], 15)
        profiled = self.collect(text, **meta, profiling_coverage_known=True, profiled_optimizer_steps=[1])
        self.assertEqual(profiled["unprofiled_stage_statistics"]["train"]["count"], 0)

    def test_running_or_missing_updates_is_partial(self):
        text = event("step", 0, {"train/step": 0}) + event("perf", 0, {"perf/train_time": 1})
        self.assertTrue(self.collect(text, expected_rollouts=1, status="SUCCEEDED")["partial"])
        self.assertTrue(self.collect(text, optimizer_steps_per_rollout=1, expected_rollouts=1, status="RUNNING")["partial"])

    def test_literals_reject_execution_and_preserve_nonfinite(self):
        with self.assertRaises(ValueError):
            summary.literal_metrics("{'train/loss': __import__('os').system('false')}")
        result = summary.literal_metrics("{'train/loss': nan, 'train/grad_norm': -inf}")
        self.assertEqual(result, {"train/loss": "nan", "train/grad_norm": "-inf"})
        json.dumps(result, allow_nan=False)
        run = self.collect(event("step", 0, {"train/loss": 1}) + "x - perf 0: {'broken': fn()}\n")
        self.assertEqual(len(run["parse_errors"]), 1)

    def test_conflicting_duplicate_is_retained_and_flagged(self):
        run = self.collect(event("step", 0, {"train/loss": 1}) + event("step", 0, {"train/loss": 2}))
        self.assertEqual(len(run["rows"][0]["train_steps"]), 2)
        self.assertEqual(len(run["conflicts"]), 1)
        self.assertTrue(run["partial"])


if __name__ == "__main__":
    unittest.main()
