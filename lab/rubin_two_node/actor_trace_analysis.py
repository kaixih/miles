#!/usr/bin/env python3
"""Analyze one retained rank-0 actor capture offline; no runtime operations.

CPU ranges are inclusive. GPU attribution requires an External id or CUDA API
correlation, never temporal overlap alone. Category unions are non-additive.
The CLI requires the successful capture receipt and exact retained trace SHA.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

MAX_DECODED = 256 * 1024**2
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
API_CATEGORIES = {"cuda_runtime", "cuda_driver"}
CPU_CATEGORIES = {"cpu_op", "user_annotation"}
SCOPES = {
    "forward": "actor.forward_step",
    "backward": "actor.backward_step",
    "checkpoint_recompute_and_backward": "actor.checkpoint_recompute_and_backward",
    "checkpointed_forward": "actor.checkpointed_forward",
    "moe_forward": "actor.moe_forward",
    "communication_enqueue": "actor.comm_enqueue.",
    "optimizer": "actor.optimizer_step",
    "recompute": "actor.recompute_forward",  # Explicit range only; never inferred.
}
CAPTURE_WINDOWS = {
    "full_optimizer_update": "actor.selected_update",
    "single_forward_backward_microbatch": "actor.selected_microbatch",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def categories(event):
    return {s.strip() for s in str(event.get("cat", "")).split(",")}


def complete(event):
    return (isinstance(event, dict) and event.get("ph") == "X"
            and all(type(event.get(k)) in (int, float) and math.isfinite(event[k]) for k in ("ts", "dur"))
            and event["dur"] > 0)


def args(event):
    return event.get("args") if isinstance(event.get("args"), dict) else {}


def identifier(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return str(int(value))
    return None


def end(event):
    return event["ts"] + event["dur"]


def merged_intervals(spans):
    merged = []
    for left, right in sorted(spans):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def interval_stats(events, begin, finish):
    spans = [(max(begin, e["ts"]), min(finish, end(e))) for e in events
             if e["ts"] < finish and end(e) > begin]
    merged = merged_intervals(spans)
    cumulative = sum(b - a for a, b in spans)
    union = sum(b - a for a, b in merged)
    return {"events": len(spans), "cumulative_ms": cumulative / 1000, "union_ms": union / 1000,
            "overlap_ms": (cumulative - union) / 1000,
            "window_uncovered_ms": (finish - begin - union) / 1000}


def scope_name(event):
    name = str(event.get("name", ""))
    for label, expected in SCOPES.items():
        if name == expected or (expected.endswith(".") and name.startswith(expected)):
            return label
    return None


def kernel_family(name):
    """Disjoint name heuristic, independent of correlated semantic attribution."""
    name = name.lower()
    if any(token in name for token in ("nccl", "alltoall", "all_to_all", "allreduce", "reduce_scatter")):
        return "communication_name"
    if any(token in name for token in ("flash_fwd", "flash_bwd", "flashattn", "flash_attention", "fmha", "attention")):
        return "attention_name"
    if any(token in name for token in ("moe", "expert", "permute", "unpermute", "routing", "router")):
        return "moe_or_routing_name"
    if any(token in name for token in ("gemm", "cublas", "cutlass", "nvjet", "wgmma", "xmma")):
        return "matrix_kernel_name"
    if any(token in name for token in ("adam", "multi_tensor")):
        return "optimizer_or_multi_tensor_name"
    if any(token in name for token in ("elementwise", "vectorized", "pointwise", "unrolled", "layernorm", "layer_norm", "rmsnorm", "softmax", "gelu", "silu", "reduce_kernel", "copy_kernel", "rope")):
        return "elementwise_norm_or_reduction_name"
    return "other_name"


def analyze_document(document, device=None, capture_window="full_optimizer_update"):
    raw = document.get("traceEvents")
    if not isinstance(raw, list):
        raise ValueError("Expected Chrome traceEvents")
    events = [(i, e) for i, e in enumerate(raw) if complete(e)]
    if capture_window not in CAPTURE_WINDOWS:
        raise ValueError("Unsupported capture window")
    selected_annotation = CAPTURE_WINDOWS[capture_window]
    selected = [(i, e) for i, e in events if "user_annotation" in categories(e)
                and e.get("name") in CAPTURE_WINDOWS.values()]
    if len(selected) != 1:
        raise ValueError("Require exactly one complete selected CPU capture range")
    if selected[0][1].get("name") != selected_annotation:
        raise ValueError("Receipt capture window does not match the trace annotation")
    selected_index, update = selected[0]
    begin, finish, pid = update["ts"], end(update), update.get("pid")
    gpu = [(i, e) for i, e in events if categories(e) & GPU_CATEGORIES
           and e["ts"] < finish and end(e) > begin]
    devices = sorted({str(args(e).get("device", e.get("pid"))) for _, e in gpu})
    if device is None:
        if len(devices) != 1:
            raise ValueError("Require one observed GPU or an explicit device filter")
        device = devices[0]
    device = str(device)
    gpu = [(i, e) for i, e in gpu if str(args(e).get("device", e.get("pid"))) == device]
    kernels = [(i, e) for i, e in gpu if "kernel" in categories(e)]
    if not kernels:
        raise ValueError("Selected capture contains no kernels on the requested device")
    scopes = [(i, e, scope_name(e)) for i, e in events
              if "user_annotation" in categories(e) and scope_name(e)
              and e.get("pid") == pid and begin <= e["ts"] and end(e) <= finish]
    if capture_window == "single_forward_backward_microbatch" and any(label == "optimizer" for _, _, label in scopes):
        raise ValueError("Microbatch capture unexpectedly contains an optimizer range")
    spans_by_thread = defaultdict(lambda: defaultdict(list))
    for _, event, label in scopes:
        spans_by_thread[event.get("tid")][label].append((event["ts"], end(event)))
    containment = {}
    for thread, labels in spans_by_thread.items():
        containment[thread] = {}
        for label, spans in labels.items():
            starts, prefix_max_end = [], []
            for start, finish_scope in sorted(spans):
                starts.append(start)
                prefix_max_end.append(max(finish_scope, prefix_max_end[-1]) if prefix_max_end else finish_scope)
            containment[thread][label] = (starts, prefix_max_end)
    parent_cache = {}

    # Any containing scope exists iff the latest-start prefix has a sufficiently
    # large end. This also handles overlapping/non-nested ranges of one label.
    # Queries are O(labels * log(scopes)), cached once per source CPU event.
    # No cross-thread timing inference is made when autograd launches elsewhere.
    def parents(index, event):
        if index in parent_cache:
            return parent_cache[index]
        if event.get("pid") != pid:
            parent_cache[index] = frozenset()
            return parent_cache[index]
        labels = []
        for label, (starts, prefix_max_end) in containment.get(event.get("tid"), {}).items():
            position = bisect_right(starts, event["ts"]) - 1
            if position >= 0 and end(event) <= prefix_max_end[position]:
                labels.append(label)
        parent_cache[index] = frozenset(labels)
        return parent_cache[index]

    external, correlation = defaultdict(list), defaultdict(list)
    for index, event in events:
        cats = categories(event)
        if cats & CPU_CATEGORIES:
            key = identifier(args(event).get("External id"))
            if key is not None:
                external[key].append((index, event))
        if cats & API_CATEGORIES:
            key = identifier(args(event).get("correlation"))
            if key is not None:
                correlation[key].append((index, event))

    def resolve(candidates):
        if not candidates:
            return None
        # Nested runtime/driver calls can legitimately have the same correlation.
        # Multiple source events are accepted only if PID, thread and all scope
        # memberships agree; otherwise retain the ambiguity instead of guessing.
        signatures = {(e.get("pid"), e.get("tid"), parents(i, e)) for i, e in candidates}
        if len(signatures) != 1 or next(iter(signatures))[0] != pid:
            return {"ambiguous": True}
        return {"ambiguous": False, "labels": next(iter(signatures))[2],
                "cpu_event_indices": [i for i, _ in candidates[:4]]}

    labeled, by_family, by_name = defaultdict(list), defaultdict(list), defaultdict(list)
    linked, scoped, ambiguous, unlinked = [], [], [], []
    proof = defaultdict(list)
    methods = Counter()
    for index, event in kernels:
        ext = identifier(args(event).get("External id"))
        corr = identifier(args(event).get("correlation"))
        direct = resolve(external.get(ext, [])) if ext is not None else None
        launch = resolve(correlation.get(corr, [])) if corr is not None else None
        candidates = [x for x in (direct, launch) if x is not None]
        if any(x["ambiguous"] for x in candidates) or (len(candidates) == 2 and direct["labels"] != launch["labels"]):
            ambiguous.append(event)
            method = "ambiguous_or_conflicting"
        elif candidates:
            chosen = candidates[0]
            labels = chosen["labels"]
            linked.append(event)
            if labels:
                scoped.append(event)
            method = "external_and_correlation" if len(candidates) == 2 else ("external_id" if direct else "runtime_correlation")
            for label in labels:
                labeled[label].append(event)
                if len(proof[label]) < 4:
                    proof[label].append({"kernel_event_index": index, "kernel": event.get("name"),
                                         "external_id": ext, "correlation": corr, "method": method,
                                         "cpu_event_indices": sorted({i for x in candidates for i in x["cpu_event_indices"]})})
        else:
            unlinked.append(event)
            method = "unlinked"
        methods[method] += 1
        by_family[kernel_family(str(event.get("name", "")))].append(event)
        by_name[str(event.get("name", ""))].append(event)

    rows = {}
    for label, annotation in SCOPES.items():
        cpu = [e for _, e, name in scopes if name == label]
        rows[label] = None if not cpu else {
            "annotation": annotation, "cpu_inclusive": interval_stats(cpu, begin, finish),
            "correlated_gpu_kernels": interval_stats(labeled[label], begin, finish),
            "gpu_attribution_status": "OBSERVED" if labeled[label] else "NO_CORRELATED_KERNELS_OBSERVED",
            "proof_examples": proof[label],
            "scope": "same-thread CPU annotation containing the linked operation/API; async GPU execution retained within selected capture window",
        }
    all_kernels = [e for _, e in kernels]
    total = interval_stats(all_kernels, begin, finish)
    total_cumulative = total["cumulative_ms"]
    families = {k: interval_stats(v, begin, finish) for k, v in sorted(by_family.items())}
    named = sorted(({"name": name, "family": kernel_family(name), **interval_stats(es, begin, finish)}
                    for name, es in by_name.items()), key=lambda row: row["cumulative_ms"], reverse=True)
    launches = [e for _, e in events if categories(e) & API_CATEGORIES and e.get("pid") == pid
                and begin <= e["ts"] and end(e) <= finish]
    paired_gpu = [e for _, e in events if "gpu_user_annotation" in categories(e)
                  and e.get("name") == selected_annotation
                  and identifier(args(e).get("External id")) == identifier(args(update).get("External id"))
                  and identifier(args(update).get("External id")) is not None
                  and str(args(e).get("device", e.get("pid"))) == device]
    streams = defaultdict(list)
    for _, event in gpu:
        streams[str(args(event).get("stream", event.get("tid")))].append(event)
    timeline = {}
    for label in SCOPES:
        spans = [(max(begin, e["ts"]), min(finish, end(e))) for _, e, name in scopes if name == label]
        if spans:
            timeline["CPU " + label] = [[(a-begin)/1000, (b-a)/1000] for a, b in merged_intervals(spans)]
    for label, family_events in by_family.items():
        spans = [(max(begin, e["ts"]), min(finish, end(e))) for e in family_events]
        timeline["GPU " + label] = [[(a-begin)/1000, (b-a)/1000] for a, b in merged_intervals(spans)]
    return {
        "schema": "actor-update-trace-analysis-v1", "status": "ANALYZED", "timestamp_unit": "microseconds",
        "raw_event_count": len(raw), "valid_complete_events": len(events),
        "invalid_complete_events": sum(isinstance(e, dict) and e.get("ph") == "X" and not complete(e) for e in raw),
        "window": {"cpu_annotation": selected_annotation, "capture_window": capture_window,
                   "microbatch_index": 1 if capture_window == "single_forward_backward_microbatch" else None,
                   "event_index": selected_index,
                   "start_timestamp_us": begin, "duration_ms": (finish-begin)/1000, "cpu_pid": pid,
                   "cpu_tid": update.get("tid"), "device": device, "gpu_annotation_matches": len(paired_gpu),
                   "gpu_annotation_duration_ms": paired_gpu[0]["dur"]/1000 if len(paired_gpu) == 1 else None},
        "device_properties": document.get("deviceProperties", []),
        "kernel_activity": total, "all_gpu_activity": interval_stats([e for _, e in gpu], begin, finish),
        "gpu_streams": {k: interval_stats(v, begin, finish) for k, v in sorted(streams.items())},
        "attribution": {"methods": dict(methods), "linked": interval_stats(linked, begin, finish),
                        "unlinked": interval_stats(unlinked, begin, finish), "ambiguous": interval_stats(ambiguous, begin, finish),
                        "linked_to_semantic_range": interval_stats(scoped, begin, finish),
                        "linked_kernel_count_fraction": len(linked)/len(kernels),
                        "semantic_range_kernel_count_fraction": len(scoped)/len(kernels),
                        "linked_cumulative_duration_fraction": interval_stats(linked, begin, finish)["cumulative_ms"]/total_cumulative if total_cumulative else None},
        "semantic_ranges_non_additive": rows, "kernel_families_name_heuristic": families,
        "top_kernels": named[:30], "kernel_name_count": len(named),
        "cpu_cuda_api_calls": dict(Counter(str(e.get("cat")) + ":" + str(e.get("name")) for e in launches)),
        "timeline_union_intervals_ms": timeline,
        "limitations": [
            ("One forward/backward microbatch (index1 within update1), rank0; optimizer is outside this capture and remains null. This is not a full-update trace."
             if capture_window == "single_forward_backward_microbatch" else
             "One profiled optimizer update on one rank; this cannot establish the all-rank critical path or hardware-only causality."),
            "CPU ranges are inclusive. Forward/backward, MoE, recomputation and communication categories overlap and must not be summed.",
            "Pure recompute remains null without an explicit actor.recompute_forward annotation; checkpoint backward includes recompute plus backward.",
            "CPU communication enqueue duration is not GPU collective completion time or a directly measured network wait.",
            "GPU attribution uses unique External id/runtime correlation and same-thread CPU containment; missing or conflicting links remain unassigned.",
            "GPU kernels may execute after their CPU annotation ends; they are clipped only to the selected capture window, not the enqueue span.",
            "Kernel cumulative time double-counts overlapping streams; union coverage and uncovered time are not physical GPU utilization, occupancy or idle time.",
            "Name-based kernel families are heuristics and do not prove call sites or distinguish every GEMM/attention/MoE implementation.",
            "Capture-boundary CUDA synchronization and profiler instrumentation perturb this sample; compare separate unprofiled update timings.",
        ],
    }


def analyze(path, expected_sha, receipt_path, device=None):
    path, receipt_path = Path(path), Path(receipt_path)
    if file_sha(path) != expected_sha:
        raise ValueError("Retained trace SHA256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status"), receipt.get("rank"), receipt.get("target")) != ("COMPLETE", 0, [0, 1, 0]):
        raise ValueError("Require successful rank0 rollout0/step1 capture receipt")
    if receipt["trace"]["sha256"] != expected_sha or receipt["trace"]["bytes"] != path.stat().st_size:
        raise ValueError("Capture receipt does not bind the retained trace")
    capture_window = receipt.get("capture_window", "full_optimizer_update")
    if capture_window not in CAPTURE_WINDOWS:
        raise ValueError("Unsupported receipt capture window")
    if capture_window == "single_forward_backward_microbatch":
        if type(receipt.get("microbatch_index")) is not int or receipt["microbatch_index"] != 1:
            raise ValueError("Require explicit microbatch_index=1 in the capture receipt")
    elif receipt.get("microbatch_index") is not None:
        raise ValueError("Full-update receipt cannot specify a microbatch index")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        decoded = stream.read(MAX_DECODED + 1)
    if len(decoded) > MAX_DECODED:
        raise ValueError("Decoded trace exceeds 256MiB capture bound")
    result = analyze_document(json.loads(decoded), device=device, capture_window=capture_window)
    result["source"] = {"path": str(path.resolve()), "sha256": expected_sha, "bytes": path.stat().st_size,
                        "receipt": str(receipt_path.resolve()), "receipt_sha256": file_sha(receipt_path),
                        "analyzer_sha256": file_sha(__file__)}
    result["capture"] = {"rank": 0, "target": [0, 1, 0], "packing_fingerprint": receipt.get("packing_fingerprint"),
                         "capture_window": capture_window, "microbatch_index": receipt.get("microbatch_index"),
                         "options": receipt.get("profiler_options"), "scope": receipt.get("scope")}
    return result


def render_png(result, path):
    """Optional standard plotting output; always generated from this actual trace."""
    import matplotlib.pyplot as plt

    labels = {
        "CPU forward": "Forward · CPU", "CPU backward": "Backward · CPU",
        "CPU checkpoint_recompute_and_backward": "Recompute + bwd · CPU",
        "CPU optimizer": "Optimizer · CPU", "GPU matrix_kernel_name": "GEMM · GPU",
        "GPU attention_name": "Attention · GPU", "GPU moe_or_routing_name": "MoE / routing · GPU",
        "GPU communication_name": "Collectives · GPU",
        "GPU optimizer_or_multi_tensor_name": "Optimizer / multi-tensor · GPU",
        "GPU elementwise_norm_or_reduction_name": "Pointwise / norm · GPU", "GPU other_name": "Other · GPU",
    }
    rows = [(name, result["timeline_union_intervals_ms"][name]) for name in labels
            if name in result["timeline_union_intervals_ms"]]
    with plt.rc_context({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 17}):
        fig, axis = plt.subplots(figsize=(15, max(5, len(rows) * 0.43 + 1.4)))
        for index, (name, spans) in enumerate(rows):
            axis.broken_barh([(start / 1000, duration / 1000) for start, duration in spans],
                            (index - 0.35, 0.7), facecolors="#087f96" if name.startswith("CPU") else "#d4773e")
        axis.set_yticks(range(len(rows)), labels=[labels[name] for name, _ in rows])
        axis.invert_yaxis()
        axis.set_xlim(0, result["window"]["duration_ms"] / 1000)
        microbatch = result["window"].get("capture_window") == "single_forward_backward_microbatch"
        axis.set_xlabel("Seconds within the profiled microbatch" if microbatch else "Seconds within the profiled update")
        axis.set_title(("Actor update 1 · microbatch 1 · rank 0" if microbatch else "Actor update 1 · rank 0")
                       + "\nCPU ranges and GPU kernel families")
        axis.grid(axis="x", alpha=0.2)
        fig.text(.99, .01, "Overlapping rows are not additive. GPU families use kernel names.",
                 ha="right", fontsize=13, color="#667986")
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(path, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    opts = parser.parse_args()
    if opts.output.exists() or (opts.png and opts.png.exists()):
        raise ValueError("Refusing to overwrite an existing analysis artifact")
    result = analyze(opts.trace, opts.sha256, opts.receipt, opts.device)
    with opts.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if opts.png:
        render_png(result, opts.png)
    print(json.dumps({"output": str(opts.output), "kernel_count": result["kernel_activity"]["events"],
                      "attribution_fraction": result["attribution"]["linked_kernel_count_fraction"]}))


if __name__ == "__main__":
    main()
