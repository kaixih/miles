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


if __name__ == "__main__":
    unittest.main()
