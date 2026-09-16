"""Synthetic parser fixtures only; no fixtures are written to report outputs."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab.rubin_two_node import actor_trace_analysis as analysis


def event(name, cat, ts, dur, ext=None, corr=None, pid=7, tid=11, **kwargs):
    args = dict(kwargs)
    if ext is not None:
        args["External id"] = ext
    if corr is not None:
        args["correlation"] = corr
    if cat in {"kernel", "gpu_memcpy", "gpu_user_annotation"}:
        args.update(device=0, stream=1)
        pid = 0
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur,
            "pid": pid, "tid": tid, "args": args}


def fixture():
    return {"traceEvents": [
        event("actor.selected_update", "user_annotation", 0, 1000, ext=1),
        event("actor.selected_update", "gpu_user_annotation", 100, 900, ext=1),
        event("actor.forward_step", "user_annotation", 10, 90, ext=2),
        event("actor.moe_forward", "user_annotation", 20, 50, ext=3),
        event("aten::mm", "cpu_op", 30, 20, ext=10),
        event("cudaLaunchKernel", "cuda_runtime", 35, 1, corr=90),
        event("cutlass_gemm", "kernel", 150, 100, ext=10, corr=90),
        event("actor.backward_step", "user_annotation", 300, 300, ext=4),
        event("actor.checkpoint_recompute_and_backward", "user_annotation", 320, 270, ext=5),
        event("flash_bwd", "cpu_op", 330, 10, ext=20),
        event("cudaLaunchKernel", "cuda_runtime", 335, 1, corr=91),
        event("flash_bwd_kernel", "kernel", 360, 100, ext=20, corr=91),
        event("actor.comm_enqueue.all_to_all_single", "user_annotation", 400, 100, ext=6),
        event("c10d::alltoall", "cpu_op", 420, 30, ext=21),
        event("cudaLaunchKernel", "cuda_runtime", 425, 1, corr=93),
        event("nccl_alltoall", "kernel", 450, 100, ext=21, corr=93),
        event("actor.optimizer_step", "user_annotation", 700, 100, ext=7),
        event("optimizer_op", "cpu_op", 720, 10, ext=30),
        event("cudaLaunchKernel", "cuda_runtime", 725, 1, corr=92),
        event("multi_tensor_adam", "kernel", 730, 50, ext=30, corr=92),
        event("unattributed", "kernel", 900, 50),
        event("copy", "gpu_memcpy", 950, 30),
    ]}


def microbatch_fixture():
    document = fixture()
    for event in document["traceEvents"]:
        if event["name"] == "actor.selected_update":
            event["name"] = "actor.selected_microbatch"
            event["dur"] = 650 - event["ts"]
    return document


class ActorTraceAnalysisTests(unittest.TestCase):
    def test_async_correlation_and_overlapping_semantic_ranges(self):
        document = fixture()
        original = copy.deepcopy(document)
        result = analysis.analyze_document(document)
        self.assertEqual(document, original)
        self.assertEqual(result["kernel_activity"]["events"], 5)
        self.assertAlmostEqual(result["kernel_activity"]["cumulative_ms"], 0.4)
        self.assertAlmostEqual(result["kernel_activity"]["union_ms"], 0.39)
        self.assertAlmostEqual(result["all_gpu_activity"]["union_ms"], 0.42)
        ranges = result["semantic_ranges_non_additive"]
        self.assertAlmostEqual(ranges["forward"]["cpu_inclusive"]["union_ms"], 0.09)
        self.assertAlmostEqual(ranges["forward"]["correlated_gpu_kernels"]["union_ms"], 0.1)
        self.assertAlmostEqual(ranges["moe_forward"]["correlated_gpu_kernels"]["union_ms"], 0.1)
        self.assertAlmostEqual(ranges["backward"]["correlated_gpu_kernels"]["union_ms"], 0.19)
        self.assertIsNone(ranges["recompute"])
        self.assertEqual(result["attribution"]["methods"], {"external_and_correlation": 4, "unlinked": 1})
        self.assertEqual(result["window"]["gpu_annotation_matches"], 1)

    def test_conflicting_external_id_and_launch_scope_not_guessed(self):
        document = fixture()
        kernel = document["traceEvents"][6]
        kernel["args"]["External id"] = 20  # Backward op conflicts with forward launch.
        result = analysis.analyze_document(document)
        self.assertEqual(result["attribution"]["ambiguous"]["events"], 1)
        forward = result["semantic_ranges_non_additive"]["forward"]
        self.assertEqual(forward["correlated_gpu_kernels"]["events"], 0)
        self.assertEqual(forward["gpu_attribution_status"], "NO_CORRELATED_KERNELS_OBSERVED")

    def test_cross_thread_time_containment_does_not_invent_parent(self):
        document = fixture()
        for index in (4, 5):
            document["traceEvents"][index]["tid"] = 12
        result = analysis.analyze_document(document)
        self.assertEqual(result["attribution"]["linked"]["events"], 4)
        self.assertEqual(result["attribution"]["linked_to_semantic_range"]["events"], 3)
        self.assertEqual(result["semantic_ranges_non_additive"]["forward"]["correlated_gpu_kernels"]["events"], 0)

    def test_external_only_runtime_only_and_nested_driver_aliases(self):
        document = fixture()
        del document["traceEvents"][6]["args"]["correlation"]
        del document["traceEvents"][11]["args"]["External id"]
        document["traceEvents"].append(event("cuLaunchKernel", "cuda_driver", 335.2, 0.3, corr=91))
        result = analysis.analyze_document(document)
        self.assertEqual(result["attribution"]["methods"]["external_id"], 1)
        self.assertEqual(result["attribution"]["methods"]["runtime_correlation"], 1)
        self.assertEqual(result["attribution"]["ambiguous"]["events"], 0)

    def test_duplicate_ids_across_processes_fail_closed(self):
        document = fixture()
        document["traceEvents"].append(event("other_process", "cpu_op", 30, 20, ext=10, pid=99))
        result = analysis.analyze_document(document)
        self.assertEqual(result["attribution"]["ambiguous"]["events"], 1)

    def test_explicit_recompute_annotation_is_required(self):
        document = fixture()
        document["traceEvents"].append(event("actor.recompute_forward", "user_annotation", 325, 25, ext=40))
        result = analysis.analyze_document(document)
        recompute = result["semantic_ranges_non_additive"]["recompute"]
        self.assertEqual(recompute["correlated_gpu_kernels"]["events"], 1)
        self.assertAlmostEqual(recompute["correlated_gpu_kernels"]["union_ms"], 0.1)

    def test_containment_prefix_keeps_longer_earlier_scope_and_equal_boundaries(self):
        document = fixture()
        # The newest-starting MoE range ends before the operation. The earlier
        # range still contains it, so a last-range-only lookup would be wrong.
        document["traceEvents"] += [
            event("actor.moe_forward", "user_annotation", 25, 4, ext=80),
            event("actor.moe_forward", "user_annotation", 75, 20, ext=81),
            event("actor.recompute_forward", "user_annotation", 30, 20, ext=82),
        ]
        result = analysis.analyze_document(document)
        ranges = result["semantic_ranges_non_additive"]
        self.assertEqual(ranges["moe_forward"]["correlated_gpu_kernels"]["events"], 1)
        self.assertEqual(ranges["recompute"]["correlated_gpu_kernels"]["events"], 1)

    def test_parent_queries_are_cached_for_repeated_correlated_kernels(self):
        document = fixture()
        document["traceEvents"] += [copy.deepcopy(document["traceEvents"][6]) for _ in range(1000)]
        with patch.object(analysis, "bisect_right", wraps=analysis.bisect_right) as lookup:
            result = analysis.analyze_document(document)
        self.assertEqual(result["attribution"]["linked"]["events"], 1004)
        # Eight unique CPU/API source events, each queried against observed labels.
        # A scan for every repeated kernel would make thousands of queries.
        self.assertLessEqual(lookup.call_count, 8 * len(analysis.SCOPES))

    def test_missing_duplicate_window_and_multiple_devices_rejected(self):
        document = fixture()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analysis.analyze_document({"traceEvents": document["traceEvents"][1:]})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analysis.analyze_document({"traceEvents": document["traceEvents"] + [document["traceEvents"][0]]})
        document["traceEvents"][-2]["args"]["device"] = 1
        with self.assertRaisesRegex(ValueError, "explicit device"):
            analysis.analyze_document(document)
        self.assertEqual(analysis.analyze_document(document, device=0)["kernel_activity"]["events"], 4)

    def test_clipping_overlapping_streams_and_invalid_events(self):
        values = [event("a", "kernel", -10, 30), event("b", "kernel", 10, 30), event("c", "kernel", 95, 20)]
        stats = analysis.interval_stats(values, 0, 100)
        self.assertAlmostEqual(stats["cumulative_ms"], 0.055)
        self.assertAlmostEqual(stats["union_ms"], 0.045)
        document = fixture()
        document["traceEvents"] += [event("invalid", "kernel", float("nan"), 2), event("zero", "kernel", 2, 0)]
        self.assertEqual(analysis.analyze_document(document)["invalid_complete_events"], 2)

    def test_compressed_receipt_integrity_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.json.gz"
            with gzip.open(trace, "wt") as stream:
                json.dump(fixture(), stream)
            sha = hashlib.sha256(trace.read_bytes()).hexdigest()
            record = {"status": "COMPLETE", "rank": 0, "target": [0, 1, 0],
                      "trace": {"sha256": sha, "bytes": trace.stat().st_size}}
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(record))
            self.assertEqual(analysis.analyze(trace, sha, receipt)["capture"]["rank"], 0)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                analysis.analyze(trace, "0" * 64, receipt)
            record["rank"] = 1
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "rank0"):
                analysis.analyze(trace, sha, receipt)

    def test_microbatch_range_excludes_optimizer_and_requires_single_matching_window(self):
        document = microbatch_fixture()
        result = analysis.analyze_document(document, capture_window="single_forward_backward_microbatch")
        self.assertEqual(result["window"]["microbatch_index"], 1)
        self.assertEqual(result["window"]["cpu_annotation"], "actor.selected_microbatch")
        self.assertIsNone(result["semantic_ranges_non_additive"]["optimizer"])
        self.assertEqual(result["kernel_activity"]["events"], 3)
        with self.assertRaisesRegex(ValueError, "does not match"):
            analysis.analyze_document(document)
        document["traceEvents"].append(event("actor.selected_update", "user_annotation", 0, 1000))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            analysis.analyze_document(document, capture_window="single_forward_backward_microbatch")
        document = microbatch_fixture()
        document["traceEvents"].append(event("actor.optimizer_step", "user_annotation", 610, 20))
        with self.assertRaisesRegex(ValueError, "optimizer range"):
            analysis.analyze_document(document, capture_window="single_forward_backward_microbatch")

    def test_microbatch_receipt_must_match_trace_and_explicit_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.json.gz"
            with gzip.open(trace, "wt") as stream:
                json.dump(microbatch_fixture(), stream)
            sha = hashlib.sha256(trace.read_bytes()).hexdigest()
            record = {"status": "COMPLETE", "rank": 0, "target": [0, 1, 0],
                      "capture_window": "single_forward_backward_microbatch", "microbatch_index": 1,
                      "trace": {"sha256": sha, "bytes": trace.stat().st_size}}
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(record))
            result = analysis.analyze(trace, sha, receipt)
            self.assertEqual(result["capture"]["capture_window"], "single_forward_backward_microbatch")
            self.assertEqual(result["capture"]["microbatch_index"], 1)
            for index in [None, 2, True]:
                receipt.write_text(json.dumps({**record, "microbatch_index": index}))
                with self.assertRaisesRegex(ValueError, "microbatch_index=1"):
                    analysis.analyze(trace, sha, receipt)
            old_record = {k: v for k, v in record.items() if k not in {"capture_window", "microbatch_index"}}
            receipt.write_text(json.dumps(old_record))
            with self.assertRaisesRegex(ValueError, "does not match"):
                analysis.analyze(trace, sha, receipt)
            receipt.write_text(json.dumps({**old_record, "microbatch_index": 1}))
            with self.assertRaisesRegex(ValueError, "Full-update receipt"):
                analysis.analyze(trace, sha, receipt)


if __name__ == "__main__":
    unittest.main()
