"""Regression checks for real summary-shaped rows; entirely CPU/offline."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("io_overlap", Path(__file__).with_name("io_overlap.py"))
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def row(index, stamp, seconds=60):
    return {"rollout_id": index, "training_stage_complete": True, "metric_conflict": False,
            "common": {"step_seconds": seconds}, "events": [
                {"kind": "perf", "timestamp": "unrelated-generation-stamp", "metrics": {"perf/rollout_time": 20}},
                {"kind": "perf", "timestamp": stamp,
                 "metrics": {"perf/train_time": 20, "perf/step_time": seconds}}]}


class IoOverlapTests(unittest.TestCase):
    def assess(self, rows, windows, snapshot="2026-09-16T02:10:00Z", **kwargs):
        return io.assess_io_overlap(rows, "rubin", windows, snapshot,
                                    log_timestamps_are_utc=kwargs.pop("utc", True), **kwargs)

    def test_overlap_previous_guard_and_unrelated_row(self):
        rows = [row(i, f"2026-09-16 02:0{i+1}:00.000") for i in range(4)]
        windows = [{"id": "copy", "platform": "rubin", "operation": "staging", "status": "complete",
                    "start_utc": "2026-09-16T02:01:10Z", "end_utc": "2026-09-16T02:01:30Z"}]
        original = copy.deepcopy(rows)
        result = self.assess(rows, windows)
        self.assertEqual(result["excluded_rollout_ids"], [1, 2])
        self.assertEqual(result["rows"][1]["overlapping_window_ids"], ["copy"])
        self.assertEqual(result["rows"][2]["previous_interval_guard_window_ids"], ["copy"])
        self.assertEqual(result["rows"][3]["status"], "clear")
        self.assertEqual(rows, original)

    def test_unknown_timestamp_never_eligible(self):
        result = self.assess([row(1, "not-a-timestamp"), row(2, "2026-09-16 02:03:00")], [])
        self.assertEqual(result["unknown_rollout_ids"], [1, 2])
        self.assertEqual(result["excluded_rollout_ids"], [1, 2])
        self.assertIn("unknown_previous_interval", result["rows"][1]["unknown_reasons"])

    def test_naive_timestamp_needs_explicit_utc_provenance(self):
        self.assertEqual(self.assess([row(1, "2026-09-16 02:01:00")], [], utc=False)["unknown_rollout_ids"], [1])
        self.assertEqual(self.assess([row(1, "2026-09-16T02:01:00Z")], [], utc=False)["excluded_rollout_ids"], [])

    def test_open_window_is_bounded_by_snapshot_and_platform(self):
        windows = [{"id": "active", "platform": "rubin", "status": "running",
                    "start_utc": "2026-09-16T02:01:10Z", "end_utc": None},
                   {"id": "other-platform", "platform": "gb300", "start_utc": "invalid"}]
        result = self.assess([row(i, f"2026-09-16 02:0{i+1}:00") for i in range(3)], windows,
                             snapshot="2026-09-16T02:02:30Z")
        self.assertEqual([r["status"] for r in result["rows"]], ["clear", "overlap", "unknown"])
        self.assertEqual(result["excluded_rollout_ids"], [1, 2])

    def test_bad_window_or_last_perf_is_unknown(self):
        bad = [{"platform": "rubin", "status": "complete", "start_utc": "bad", "end_utc": None}]
        self.assertEqual(self.assess([row(1, "2026-09-16 02:02:00")], bad)["unknown_rollout_ids"], [1])
        item = row(1, "2026-09-16 02:02:00")
        item["events"].append({"kind": "perf", "timestamp": None,
                               "metrics": {"perf/train_time": 20, "perf/step_time": 60}})
        self.assertEqual(self.assess([item], [])["unknown_rollout_ids"], [1])


if __name__ == "__main__":
    unittest.main()
