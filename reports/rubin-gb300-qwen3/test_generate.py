import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("report_generator", ROOT / "generate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceTests(unittest.TestCase):
    def test_pending_does_not_invent_results(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "site"
            result = module.build_report(runs=Path(directory) / "missing.json", output=output)
            self.assertEqual(result["run_count"], 0)
            data = json.loads((output / "report-data.json").read_text())
            self.assertIsNone(data["inputs"]["comparison"])
            self.assertEqual(data["provenance"][0]["status"], "pending")
            self.assertTrue((output / "assets" / module.PLOTLY).is_file())

    def test_embedded_json_is_safe_and_points_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "input.json"
            source = {"schema": "miles-qwen3-comparison-v1", "runs": [
                {"label": "</script><script>alert(1)</script>", "partial": True,
                 "rows": [{"rollout_id": 0, "common": {"training_reward_mean": 0.25}}]}]}
            path.write_text(json.dumps(source))
            output = Path(directory) / "site"
            module.build_report(runs=path, output=output)
            data = json.loads((output / "report-data.json").read_text())
            self.assertEqual(data["inputs"]["comparison"], source)
            self.assertNotIn("</script><script>alert", (output / "index.html").read_text())

    def test_profile_files_are_local_hashed_and_missing_stays_pending(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            (root / "trace.json").write_text('{"traceEvents": []}')
            profiles = root / "profiles.json"
            profiles.write_text(json.dumps({"profiles": [{"run_label": "rubin", "trace": "trace.json", "image": "absent.png"}]}))
            module.build_report(profiles=profiles, output=root / "site")
            data = json.loads((root / "site/report-data.json").read_text())
            files = data["inputs"]["profiles"]["profiles"][0]["attachments"]
            self.assertEqual(files["image"]["status"], "pending")
            self.assertEqual(files["trace"]["sha256"], module.sha256(root / "trace.json"))
            self.assertTrue((root / "site" / files["trace"]["url"]).is_file())
            profiles.write_text(json.dumps({"profiles": [{"image": "https://example.com/a.png"}]}))
            with self.assertRaisesRegex(ValueError, "local file"):
                module.build_report(profiles=profiles, output=root / "rejected")

    def test_invalid_schema_is_not_silently_rendered(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            path = root / "bad.json"
            path.write_text('{"schema": "unknown", "runs": []}')
            with self.assertRaisesRegex(ValueError, "schema"):
                module.build_report(runs=path, output=root / "site")

    def test_health_metadata_cannot_attach_to_a_different_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            source = {"schema": "miles-qwen3-comparison-v1", "runs": [
                {"label": "gb300", "metadata": {"run_id": "run-one", "status": "RUNNING"}, "rows": []}]}
            runs, health = root / "runs.json", root / "health.json"
            runs.write_text(json.dumps(source))
            record = {"schema": "miles-run-health-v1", "runs": [
                {"label": "gb300", "run_id": "run-one", "run_health": "recovery_pending", "issue": "Recovery pending"}]}
            health.write_text(json.dumps(record))
            module.build_report(runs=runs, run_health=health, output=root / "site")
            result = json.loads((root / "site/report-data.json").read_text())
            self.assertEqual(result["inputs"]["comparison"], source)
            self.assertEqual(result["inputs"]["run_health"], record)
            record["comparison_sha256"] = "0" * 64
            health.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "comparison_sha256 must match"):
                module.build_report(runs=runs, run_health=health, output=root / "stale-rejected")
            record["comparison_sha256"] = module.sha256(runs)
            health.write_text(json.dumps(record))
            module.build_report(runs=runs, run_health=health, output=root / "matched")
            record["runs"][0]["run_id"] = "another-run"
            health.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "run label and run_id"):
                module.build_report(runs=runs, run_health=health, output=root / "rejected")

    def test_paired_stage_cohort_uses_intersection_without_changing_raw_statistics(self):
        def row(i, value, **changes):
            return {"rollout_id": i, "training_stage_complete": True, "profiled": False,
                    "unprofiled_timing_eligible": True,
                    "common": {key + "_seconds": value for key in ("step", *module.PAIRED_STAGE_KEYS)}, **changes}
        source = {"schema": "miles-qwen3-comparison-v1", "runs": [
            {"label": "rubin", "completed_training_rollouts": list(range(7)), "metadata": {"exclude_timing_rollouts": [4]},
             "unprofiled_stage_statistics": {"rollout": {"mean_seconds": 999}},
             "rows": [row(0, 9999), row(1, 10), row(2, 20), row(3, 30, profiled=True), row(4, 40), row(5, 50, unprofiled_timing_eligible=False), row(6, 60)]},
            {"label": "gb300", "completed_training_rollouts": [0, 1, 2, 3, 4, 5], "metadata": {},
             "rows": [row(0, 9999), row(1, 20), row(2, 40), row(3, 60), row(4, 80), row(5, 100), row(6, 120)]}]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory); path = root / "runs.json"; path.write_text(json.dumps(source))
            module.build_report(runs=path, output=root / "site")
            data = json.loads((root / "site/report-data.json").read_text())
        paired = data["derived"]["paired_timing"]
        self.assertEqual(data["inputs"]["comparison"], source)
        self.assertEqual(paired["rollout_ids"], [1, 2])
        self.assertEqual(paired["count"], 2)
        self.assertEqual(paired["statistics"]["rubin"]["rollout"]["mean_seconds"], 15)
        self.assertEqual(paired["statistics"]["gb300"]["rollout"]["median_seconds"], 30)
        self.assertEqual(paired["step_difference_seconds_second_minus_first"], 15)
        self.assertIsNotNone(paired["comparison_sha256"])

    def test_empty_or_missing_paired_cohort_stays_pending(self):
        self.assertEqual(module.paired_timing(None)["status"], "pending")
        result = module.paired_timing({"runs": [{"label": "a", "rows": []}, {"label": "b", "rows": []}]})
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["statistics"], {})
        self.assertIsNone(result["largest_observed_stage_gap"])

    def test_missing_stage_does_not_shrink_one_side_of_paired_cohort(self):
        source = {"runs": [{"label": label, "completed_training_rollouts": [1, 2], "rows": [
            {"rollout_id": i, "training_stage_complete": True, "profiled": False,
             "unprofiled_timing_eligible": True,
             "common": {"rollout_seconds": None if label == "b" and i == 2 else i * 10}}
            for i in [1, 2]]} for label in ["a", "b"]]}
        paired = module.paired_timing(source)
        self.assertEqual(paired["status"], "partial")
        self.assertEqual(paired["rollout_ids"], [1, 2])
        self.assertEqual(paired["statistics"]["b"]["rollout"]["missing_rollout_ids"], [2])
        self.assertFalse(paired["statistics"]["a"]["rollout"]["paired_metric_available"])
        self.assertFalse(paired["statistics"]["b"]["rollout"]["paired_metric_available"])
        self.assertIsNone(paired["statistics"]["b"]["rollout"]["mean_seconds"])


if __name__ == "__main__":
    unittest.main()
