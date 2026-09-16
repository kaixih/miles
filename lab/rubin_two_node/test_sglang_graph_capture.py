"""CPU-only evidence checks: no API, Docker, Ray, or GPU work."""

import gzip
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("graph_capture", Path(__file__).with_name("sglang_graph_capture.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def event(name, category, ts=10, duration=1, pid=7, tid=8):
    return {"name": name, "cat": category, "ph": "X", "ts": ts, "dur": duration,
            "pid": pid, "tid": tid}


class EvidenceTests(unittest.TestCase):
    def test_request_preserves_tokens_and_template_provenance(self):
        tokens = {"input_ids": [[11, 22], [33, 44, 55]], "source": {
            "dataset_sha256": "a" * 64, "selected_row_indices": [2, 5],
            "tokenizer_model": "Qwen3-30B-A3B", "chat_template_kwargs": {}}}
        original = copy.deepcopy(tokens)
        a = C.generation_request(tokens, "off")
        b = C.generation_request(tokens, "on")
        self.assertEqual(tokens, original)
        self.assertEqual(a["request"]["input_ids"], tokens["input_ids"])
        self.assertEqual(a["prompt_token_counts"], [2, 3])
        self.assertEqual(a["request"]["sampling_params"]["max_new_tokens"], 64)
        self.assertTrue(a["request"]["sampling_params"]["ignore_eos"])
        self.assertEqual(a["source"]["chat_template_kwargs"], {})
        self.assertEqual(a["workload_sha256"], b["workload_sha256"])
        self.assertNotEqual(a["request_sha256"], b["request_sha256"])
        self.assertNotEqual(a["workload_sha256"], C.generation_request(tokens, "off", 32)["workload_sha256"])

    def test_request_rejects_unbounded_or_unattributed_inputs(self):
        valid = {"input_ids": [[1, 2]], "source": {
            "dataset_sha256": "b" * 64, "selected_row_indices": [0],
            "tokenizer_model": "model", "chat_template_kwargs": {}}}
        for prompts in [[], [[1]] * 129, [[]], [[1] * 513], [[True]], [[-1]], [[2**63]]]:
            bad = {**valid, "input_ids": prompts}
            with self.subTest(prompts=str(prompts)[:60]), self.assertRaises(ValueError):
                C.generation_request(bad, "run")
        for field, value in [("dataset_sha256", "unknown"), ("selected_row_indices", []),
                             ("chat_template_kwargs", None), ("tokenizer_model", "")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.generation_request({**valid, "source": {**valid["source"], field: value}}, "run")
        for output, context in [(7, 1536), (65, 1536), (64, 32), (64, 1537)]:
            with self.subTest(output=output, context=context), self.assertRaises(ValueError):
                C.generation_request(valid, "run", output, context)

    def test_stage_payload_is_bounded_without_python_stacks(self):
        payload = C.stage_payload("/run-output/diagnostic/traces", "unique-engine")
        self.assertTrue(payload["profile_by_stage"])
        self.assertTrue(payload["detailed_annotations"])
        self.assertEqual(payload["num_steps"], 4)
        self.assertFalse(payload["with_stack"])
        self.assertNotIn("profile_stages", payload)
        for directory, steps in [("/tmp", 4), ("/run-output/../other", 4), ("/run-output/x", 5)]:
            with self.assertRaises(ValueError):
                C.stage_payload(directory, "name", steps)

    def test_actual_decode_launch_requires_same_scope_and_gpu(self):
        events = [event("step[DECODE bs=128 g_sq=128 g_sk=32768]", "user_annotation", 5, 20),
                  event("cudaGraphLaunch", "cuda_runtime"), event("actual_kernel", "kernel")]
        result = C.analyze_events(events)
        self.assertTrue(result["decode_graph_replay_proven"])
        self.assertTrue(result["cpu_and_gpu_activity_observed"])
        self.assertEqual(result["scheduler_forward_counts"], {"DECODE": 1})
        self.assertFalse(C.analyze_events(events[:-1])["decode_graph_replay_proven"])
        for replacement in [event("cudaGraphLaunch", "cuda_runtime", 50),
                            event("cudaGraphLaunch", "cuda_runtime", tid=99),
                            event("cudaGraphInstantiate", "cuda_runtime"),
                            event("cudaGraphLaunch", "kernel")]:
            with self.subTest(replacement=replacement):
                self.assertFalse(C.analyze_events([events[0], replacement, events[2]])["decode_graph_replay_proven"])

    def test_small_extend_and_mixed_modes_not_misrepresented(self):
        events = [event("step[EXTEND bs=1 c_sq=1 c_sqsq=1 c_sqsk=97 c_sk=97]", "user_annotation"),
                  event("step[DECODE bs=128]", "user_annotation", 15),
                  event("step[DECODE bs=128]", "gpu_user_annotation", 15)]
        result = C.analyze_events(events)
        self.assertTrue(result["prefill_forward_observed"])
        self.assertTrue(result["mixed_forward_modes_observed"])
        self.assertEqual(result["prefill_new_query_token_totals"], [1])
        self.assertEqual(result["scheduler_forward_counts"], {"EXTEND": 1, "DECODE": 1})
        self.assertIn("not inferred", result["prefill_scope"])
        outside_bracket = C.analyze_events([
            event("step[EXTEND bs=1] c_sq=32 c_sk=64", "user_annotation")])
        self.assertEqual(outside_bracket["prefill_new_query_token_totals"], [32])

    def test_enable_or_capture_label_is_not_graph_replay(self):
        data = [event("CUDA Graph capture complete", "user_annotation"),
                event("step[DECODE bs=32]", "user_annotation"),
                event("eager_decode_kernel", "kernel")]
        self.assertFalse(C.analyze_events(data)["decode_graph_replay_proven"])

    def test_partial_json_in_valid_gzip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "partial.gz"
            with gzip.open(p, "wb") as f:
                f.write(b'{"traceEvents": [{"name": "unfinished"')
            with self.assertRaises(json.JSONDecodeError):
                C.inspect_trace(p)

    def test_real_file_hash_and_decoded_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "trace.gz"
            with gzip.open(p, "wt") as f:
                json.dump({"traceEvents": [event("kernel", "kernel")]}, f)
            self.assertEqual(len(C.inspect_trace(p)["sha256"]), 64)
            with patch.object(C, "MAX_DECODED_BYTES", 10), self.assertRaisesRegex(ValueError, "decoded-byte limit"):
                C.inspect_trace(p)
            link = Path(directory) / "link.gz"
            link.symlink_to(p)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                C.inspect_trace(link)


if __name__ == "__main__":
    unittest.main()
