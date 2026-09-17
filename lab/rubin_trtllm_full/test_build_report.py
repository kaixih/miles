"""CPU checks that guard against misleading new-report conclusions."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import build_report as build


def comparison():
    runs = []
    for label, actor in (("rubin", 200), ("gb300", 100)):
        rows = []
        for i, seconds, rate in ((0, 1000, 900), (1, 10, 100), (2, 90, 10), (3, 800, 100)):
            rows.append({"rollout_id": i, "training_stage_complete": True, "profiled": False,
                         "unprofiled_timing_eligible": i in (1, 2),
                         "common": {"step_seconds": actor + seconds, "actor_train_seconds": actor,
                                    "rollout_seconds": seconds, "output_tokens_per_gpu_generation_second": rate,
                                    "log_probs_seconds": 3, "ref_log_probs_seconds": 5, "update_weights_seconds": 2}})
        runs.append({"label": label, "metadata": {"exclude_timing_rollouts": [0, 3]},
                     "completed_training_rollouts": [0, 1, 2, 3], "rows": rows})
    return {"runs": runs}


class ReportBoundary(unittest.TestCase):
    def test_old_comparison_cannot_be_new_results(self):
        with self.assertRaisesRegex(ValueError, "Only the new"):
            build.validate_main({"schema": "miles-qwen3-comparison-v1", "experiment_id": "qwen3-cudagraph-nightly-v1"},
                                None, Path("/tmp"), {})

    def test_final_rejects_no_evidence(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as output, tempfile.TemporaryDirectory() as inputs:
            args = argparse.Namespace(experiment=build.ROOT / "experiment.json", inputs=Path(inputs),
                                      output=Path(output), profile_rubin=None, profile_gb300=None, final=True)
            with self.assertRaisesRegex(ValueError, "Final report requires"):
                build.build(args)
            self.assertFalse((Path(output) / "index.html").exists())

    def test_timing_exclusions_and_weighted_throughput(self):
        value = build.performance(comparison())
        self.assertEqual(value["paired"]["rollout_ids"], [1, 2])
        self.assertEqual(value["weighted_output_tokens_per_gpu_generation_second"]["rubin"], 19)

    def test_actor_regression_remains_visible(self):
        value = build.performance(comparison())
        self.assertEqual(value["ratios"]["actor_train"]["gb300_over_rubin"], .5)
        self.assertEqual(value["ratios"]["actor_train"]["rubin_time_reduction"], -1)
        self.assertLess(value["ratios"]["step"]["rubin_time_reduction"], 0)

    def test_missing_metric_does_not_shorten_one_platform_cohort(self):
        data = comparison()
        data["runs"][0]["rows"][2]["common"]["actor_train_seconds"] = None
        value = build.performance(data)
        self.assertIsNone(value["ratios"]["actor_train"])
        self.assertEqual(value["paired"]["statistics"]["rubin"]["actor_train"]["missing_rollout_ids"], [2])

    def test_changed_profile_attachment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "trace.png"
            p.write_bytes(b"altered image")
            with self.assertRaisesRegex(ValueError, "attachment identity/hash"):
                build.copy_checked(str(p), "0" * 64, root, "copied.png")
            self.assertFalse((root / "evidence").exists())


if __name__ == "__main__":
    unittest.main()
