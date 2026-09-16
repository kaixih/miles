"""CPU fixtures only. Synthetic values never enter the canonical pending deck."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("graph_deck", ROOT / "generate.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class EvidenceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment = json.loads((ROOT / "experiment.json").read_text())

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def fixture(self):
        self.experiment["run_bindings"] = [{"label": label, "run_id": f"SYNTHETIC_QA_{label}_NEW"} for label in ["rubin", "gb300"]]
        return {"schema": "miles-qwen3-comparison-v1", "runs": [{"label": b["label"], "metadata": {"run_id": b["run_id"]}, "rows": [], "completed_training_rollouts": []} for b in self.experiment["run_bindings"]]}

    def test_no_inputs_creates_pending_deck_without_old_measurements(self):
        G.build_report(output=self.root / "site")
        report = json.loads((self.root / "site/report-data.json").read_text())
        self.assertIsNone(report["inputs"]["comparison"])
        self.assertEqual(report["derived"]["paired_timing"]["count"], 0)
        self.assertNotIn("historical", report["inputs"])

    def test_rejects_unbound_new_or_old_rows(self):
        run = self.fixture()
        self.experiment["run_bindings"] = []
        with self.assertRaisesRegex(ValueError, "NEW exact"):
            G.build_report(experiment=self.write("experiment.json", self.experiment), runs=self.write("runs.json", run), output=self.root / "site")

    def test_cannot_bind_previous_eager_identity(self):
        self.experiment["run_bindings"] = [{"label": "rubin", "run_id": self.experiment["previous_run_ids"][0]}]
        with self.assertRaisesRegex(ValueError, "Previous eager"):
            G.build_report(experiment=self.write("experiment.json", self.experiment), output=self.root / "site")

    def test_accepts_bound_rows_but_does_not_infer_graph_proof(self):
        run = self.fixture()
        G.build_report(experiment=self.write("experiment.json", self.experiment), runs=self.write("runs.json", run), output=self.root / "site")
        report = json.loads((self.root / "site/report-data.json").read_text())
        self.assertEqual(len(report["inputs"]["comparison"]["runs"]), 2)
        self.assertIsNone(report["inputs"]["profiles"])

    def test_stale_health_hash_rejected(self):
        run = self.fixture()
        health = {"schema": "miles-run-health-v1", "runs": [], "comparison_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            G.build_report(experiment=self.write("experiment.json", self.experiment), runs=self.write("runs.json", run), run_health=self.write("health.json", health), output=self.root / "site")

    def test_counts_cannot_exceed_observed_forwards(self):
        self.fixture()
        proof = {"schema": "qwen3-cudagraph-profiles-v1", "experiment_id": self.experiment["experiment_id"], "graph_evidence": [{"run_label": "rubin", "source_run_id": "SYNTHETIC_QA_rubin_NEW", "decode_forwards": 4, "decode_graph_replays": 5, "decode_fallbacks": 0, "decode_unknown": 0}]}
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            G.build_report(experiment=self.write("experiment.json", self.experiment), profiles=self.write("profiles.json", proof), output=self.root / "site")

    def test_diagnostic_context_mismatch_rejected(self):
        self.fixture()
        common = {"image_digest": "fixture", "initial_model_id": "fixture", "request_sha256": "a"*64, "context_sha256": "b"*64, "cache_policy": "empty", "prefill_graph": False, "generation_seconds": [1,2,3], "timing_scope": "http_request_prefill_decode_queue_response"}
        diagnostic = {"schema": "qwen3-cudagraph-diagnostics-v1", "experiment_id": self.experiment["experiment_id"], "pairs": [{"platform": "rubin", "source_run_id": "SYNTHETIC_QA_rubin_NEW", "verified": True, "off": {**common, "decode_graph": False}, "on": {**common, "decode_graph": True, "context_sha256": "c"*64}}]}
        with self.assertRaisesRegex(ValueError, "context_sha256"):
            G.build_report(experiment=self.write("experiment.json", self.experiment), diagnostics=self.write("diagnostics.json", diagnostic), output=self.root / "site")

    def diagnostic_fixture(self):
        self.fixture()
        receipt=self.write("SYNTHETIC_QA_receipt.json", {"purpose":"CPU schema fixture only"})
        common={"image_digest":"fixture", "initial_model_id":"fixture", "request_sha256":"a"*64,
                "context_sha256":"b"*64, "cache_policy":"empty", "prefill_graph":False,
                "timing_scope":"http_request_prefill_decode_queue_response"}
        return {"schema":"qwen3-cudagraph-diagnostics-v1", "experiment_id":self.experiment["experiment_id"],
                "pairs":[{"platform":"rubin", "source_run_id":"SYNTHETIC_QA_rubin_NEW", "verified":True,
                          "scope":"SYNTHETIC CPU fixture; never production evidence",
                          "evidence_refs":[{"path":str(receipt),"sha256":G.sha256(receipt)}],
                          "off":{**common,"decode_graph":False,"generation_seconds":[8.5,9.0,9.5]},
                          "on":{**common,"decode_graph":True,"generation_seconds":[5.0,5.5,6.0]}}]}

    def test_generation_request_seconds_derive_means_without_decode_label(self):
        diagnostic=self.diagnostic_fixture()
        diagnostic["pairs"][0]["off"]["generation_seconds_mean"]=999
        G.build_report(experiment=self.write("experiment.json",self.experiment), diagnostics=self.write("diagnostics.json",diagnostic), output=self.root/"site")
        pair=json.loads((self.root/"site/report-data.json").read_text())["inputs"]["diagnostics"]["pairs"][0]
        self.assertEqual(pair["off"]["generation_seconds"],[8.5,9.0,9.5])
        self.assertEqual(pair["off"]["generation_seconds_mean"],9.0)
        self.assertEqual(pair["on"]["generation_seconds_median"],5.5)
        self.assertEqual(pair["on"]["sample_count"],3)
        self.assertNotIn("decode_ms_mean",pair["off"])

    def test_decode_or_ambiguous_scope_cannot_be_mislabeled_as_http_time(self):
        original=self.diagnostic_fixture()
        for mutation in ["decode_ms","wrong_scope","missing_scope","invalid_duration"]:
            diagnostic=copy.deepcopy(original); condition=diagnostic["pairs"][0]["off"]
            if mutation=="decode_ms": condition["decode_ms"]=condition.pop("generation_seconds")
            if mutation=="wrong_scope": condition["timing_scope"]="decode_only"
            if mutation=="missing_scope": condition.pop("timing_scope")
            if mutation=="invalid_duration": condition["generation_seconds"]=[True,0,-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                G.build_report(experiment=self.write("experiment.json",self.experiment), diagnostics=self.write("diagnostics.json",diagnostic), output=self.root/"site")

    def test_original_report_output_is_protected(self):
        with self.assertRaisesRegex(ValueError, "preserved original"):
            G.build_report(output=ROOT.parent / "rubin-gb300-qwen3/site")


if __name__ == "__main__":
    unittest.main()
