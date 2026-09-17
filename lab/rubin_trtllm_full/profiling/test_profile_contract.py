"""CPU-only checks for the diagnostic's backend and replay-evidence boundary."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_profile as run
import analyze_profile as analysis


def trace():
    name = "step[DECODE bs=128 g_sk=11011]"
    return {"traceEvents": [
        {"ph": "X", "cat": "user_annotation", "name": name, "pid": 123, "tid": 123,
         "ts": 10, "dur": 5, "args": {"External id": 4}},
        {"ph": "X", "cat": "gpu_user_annotation", "name": name, "pid": 0, "tid": 7,
         "ts": 100, "dur": 40, "args": {"External id": 4}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 123, "tid": 123,
         "ts": 11, "dur": 2, "args": {"correlation": 55}},
        {"ph": "X", "cat": "kernel", "name": "actual_backend_kernel", "pid": 0, "tid": 7,
         "ts": 110, "dur": 20, "args": {"device": 0, "correlation": 55, "graph id": 8, "graph node id": 99}},
    ]}


class Contract(unittest.TestCase):
    def test_only_moe_backend_changes_in_engine_recipe(self):
        args = argparse.Namespace(model_path=Path("/models/Qwen3-30B-A3B"), host="127.0.0.1", port=31081)
        old, new = run.ORIGINAL_ENGINE_ARGV(args, "on"), run.engine_argv(args, "on")
        self.assertEqual(len(old), len(new))
        self.assertEqual([(x, y) for x, y in zip(old, new) if x != y], [("triton", "flashinfer_trtllm")])

    def test_runtime_fallback_backend_is_rejected(self):
        with patch.object(run, "ORIGINAL_HTTP", return_value={"body": json.dumps({
                "moe_runner_backend": "triton", "dtype": "bfloat16", "attention_backend": "triton",
                "bf16_gemm_backend": "torch"})}):
            with self.assertRaisesRegex(ValueError, "Actual server backend"):
                run.checked_http("http://127.0.0.1:31081", "/server_info")

    def test_cpu_enqueue_is_not_gpu_elapsed_time(self):
        result = analysis.correlated_analysis(trace())["forwards"][0]
        self.assertTrue(result["decode_replay_proven"])
        self.assertEqual(result["cpu_duration_ms"], .005)
        self.assertEqual(result["gpu_duration_ms"], .040)
        self.assertEqual(result["graph_kernel_correlation"]["kernel_intervals"]["union_ms"], .020)

    def test_temporal_overlap_without_launch_correlation_is_not_proof(self):
        doc = trace()
        doc["traceEvents"][-1]["args"]["correlation"] = 56
        result = analysis.correlated_analysis(doc)["forwards"][0]
        self.assertFalse(result["decode_replay_proven"])

    def test_plain_kernel_with_graph_correlation_is_not_graph_node_proof(self):
        doc = trace()
        doc["traceEvents"][-1]["args"]["graph node id"] = 0
        self.assertFalse(analysis.correlated_analysis(doc)["forwards"][0]["decode_replay_proven"])

    def test_foreign_thread_graph_launch_is_not_proof(self):
        doc = trace()
        doc["traceEvents"][2]["tid"] = 124
        self.assertFalse(analysis.correlated_analysis(doc)["forwards"][0]["decode_replay_proven"])


if __name__ == "__main__":
    unittest.main()
