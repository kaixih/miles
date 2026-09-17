#!/usr/bin/env python3
"""Validate retained TRTLLM captures and render real prefill/decode timelines.

Offline only. Main-run performance remains separate from these instrumented
GPU annotation windows and the diagnostic's unprofiled HTTP batch measurements.
The input directory is the retained run_profile output, not the operator root.
"""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import render_slide

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "lab/rubin_two_node"))
import analyze_sglang_graph_trace as old
import run_sglang_graph_diagnostic as wrapper
import sglang_graph_capture as capture


def sha(path):
    return wrapper.file_sha(path)


def read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Missing, symlinked or oversized JSON: " + str(path))
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float, allow_nan=False) + "\n")


def correlated_analysis(document):
    analysis = old.analyze_document(document)
    events = [e for e in document["traceEvents"] if old.complete(e)]
    for row in analysis["forwards"]:
        if row.get("status") != "PAIRED":
            continue
        cpu = [e for e in events if e.get("cat") == "user_annotation"
               and e.get("name") == row["annotation"]
               and e.get("args", {}).get("External id") == row["external_id"]]
        if len(cpu) != 1:
            continue
        c = cpu[0]
        launches = [e for e in events if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                    and old.GRAPH.fullmatch(e.get("name", ""))
                    and (e.get("pid"), e.get("tid")) == (c.get("pid"), c.get("tid"))
                    and c["ts"] <= e["ts"] and e["ts"] + e["dur"] <= c["ts"] + c["dur"]]
        correlations = {e.get("args", {}).get("correlation") for e in launches}
        correlations.discard(None)
        begin, end = row["gpu_start_timestamp_us"], row["gpu_start_timestamp_us"] + row["gpu_duration_ms"] * 1000
        kernels = [e for e in events if e.get("cat") == "kernel"
                   and e.get("args", {}).get("device", e.get("pid")) == row["device"]
                   and e["ts"] < end and e["ts"] + e["dur"] > begin
                   and e.get("args", {}).get("correlation") in correlations]
        graph_nodes = [e for e in kernels if e.get("args", {}).get("graph id", 0)
                       and e.get("args", {}).get("graph node id", 0)]
        row["graph_kernel_correlation"] = {
            "launch_correlations": sorted(correlations),
            "linked_gpu_kernel_count": len(kernels), "linked_graph_node_kernel_count": len(graph_nodes),
            "graph_ids": sorted({e["args"]["graph id"] for e in graph_nodes}),
            "graph_node_count": len({(e["args"]["graph id"], e["args"]["graph node id"]) for e in graph_nodes}),
            "kernel_names": dict(Counter(e["name"] for e in kernels)),
            "kernel_intervals": old.intervals(kernels, begin, end),
        }
        row["decode_replay_proven"] = row["stage"] == "DECODE" and bool(graph_nodes)
    analysis["schema"] = "trtllm-correlated-forward-analysis-v1"
    analysis["limitations"].append("Replay additionally requires GPU graph-node kernels linked by launch correlation; CPU cudaGraphLaunch duration is not GPU graph duration.")
    return analysis


def render(analysis, row, output, title, png):
    spec = importlib.util.spec_from_file_location("trtllm_trace_renderer", REPO / "reports/rubin-gb300-qwen3/render_trace.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    trace = Path(analysis["source"]["path"])
    with gzip.open(trace, "rt") as stream:
        doc = json.load(stream, parse_float=Decimal)
    origin = min(e["ts"] for e in doc["traceEvents"] if old.complete(e) and e.get("cat") in old.GPU
                 and e.get("args", {}).get("device", e.get("pid")) == row["device"])
    start = (row["gpu_start_timestamp_us"] - origin) / 1000
    result = renderer.render(trace, output, title, float(start), float(row["gpu_duration_ms"]),
                             device=str(row["device"]), max_visible_events=20000)
    if png:
        subprocess.run([str(renderer.NODE), str(renderer.ROOT / "capture_trace.mjs"),
                        str(output / "index.html"), str(output / "trace.png")], check=True, timeout=90)
        result["full_png"] = str(output / "trace.png")
        result["full_image_sha256"] = sha(output / "trace.png")
    result.update(render_slide.render(output, row, png, renderer.NODE, renderer.ROOT / "capture_trace.mjs"))
    result["source_trace"] = analysis["source"]
    result["selected_forward"] = row
    result["selection_rule"] = "Earliest complete bs=128 forward with paired CPU/GPU annotation and eager-prefill or correlated-graph-decode proof; never the fastest duration."
    write(output / "selection.json", result)
    return result


def build(args):
    root = args.directory.resolve()
    plan, terminal, summary = (read(root / f) for f in ("plan.json", "terminal.json", "on/summary.json"))
    if sha(root / "plan.json") != args.plan_sha256 or sha(root / "terminal.json") != args.terminal_sha256:
        raise ValueError("Plan or terminal differs from the SHA-verified retained capture")
    if (plan.get("schema") != "trtllm-prefill-decode-profile-v1"
            or plan.get("main_run_id") != args.main_run_id or terminal.get("status") != "COMPLETED"
            or terminal.get("results") != [summary] or plan.get("order") != ["on"]):
        raise ValueError("Wrong experiment identity or unsuccessful diagnostic")
    inputs = read(root / "frozen-token-input.json")
    if sha(root / "frozen-token-input.json") != plan["frozen_input_sha256"]:
        raise ValueError("Retained frozen input hash mismatch")
    request = capture.generation_request(inputs, "trtllm-offline-audit")
    if (request["input_ids_sha256"] != plan["input_ids_sha256"]
            or request["workload_sha256"] != plan["workload_sha256"]):
        raise ValueError("Retained workload differs from planned token IDs and sampling policy")
    info = json.loads(read(root / "on/server-info.json")["body"])
    if (info.get("moe_runner_backend") != "flashinfer_trtllm" or info.get("dtype") != "bfloat16"
            or info.get("cuda_graph_config", {}).get("decode", {}).get("backend") != "full"
            or info.get("cuda_graph_config", {}).get("prefill", {}).get("backend") != "disabled"):
        raise ValueError("Actual runtime backend or graph mode differs")
    if args.output.exists():
        raise ValueError("Analysis requires a fresh output directory")
    args.output.mkdir(parents=True)
    measurements = []
    for i in range(1, 4):
        record = read(root / f"on/measured-{i}.json")
        metrics = wrapper.batch_metrics(record["response"], 128, record["metrics"]["generation_request_seconds"])
        if (metrics != record["metrics"] or record["request"]["workload_sha256"] != plan["workload_sha256"]
                or record["request"]["request"]["input_ids"] != inputs["input_ids"]
                or record["flush"]["status"] != 200 or not record["flush"]["body"].startswith("Cache flushed.\n")):
            raise ValueError("Measurement/request/cache evidence mismatch")
        measurements.append(metrics)
    if measurements != summary["measurements"]:
        raise ValueError("Summary measurements differ from response-derived metrics")
    analyses = []
    for evidence in summary["capture"]["trace_evidence"]:
        path = root / "on/traces" / Path(evidence["path"]).name
        if path.is_symlink() or sha(path) != evidence["sha256"]:
            raise ValueError("Retained trace SHA mismatch")
        with gzip.open(path, "rb") as stream:
            decoded = stream.read(old.MAX_BYTES + 1)
        if len(decoded) > old.MAX_BYTES:
            raise ValueError("Trace exceeds the bounded decoded size")
        analysis = correlated_analysis(json.loads(decoded, parse_float=Decimal))
        analysis["source"] = {"path": str(path), "sha256": evidence["sha256"],
                              "analyzer_sha256": sha(Path(__file__))}
        analyses.append(analysis)
        write(args.output / (path.name + ".analysis.json"), analysis)
    selections = {}
    for stage in ("prefill", "decode"):
        candidates = []
        for analysis in analyses:
            for row in analysis["forwards"]:
                if row.get("status") != "PAIRED" or row["fields"].get("bs") != 128:
                    continue
                valid = (row["stage"] == "DECODE" and row.get("decode_replay_proven")) if stage == "decode" else (
                    row["stage"] == "EXTEND" and not any(row["graph_launch_calls_by_category"].values())
                    and any(row["kernel_launch_calls_by_category"].values()))
                if valid:
                    candidates.append((analysis, row))
        if not candidates:
            raise ValueError("No complete proven " + stage + " forward; preserve partial evidence")
        analysis, row = min(candidates, key=lambda v: v[1]["gpu_start_timestamp_us"])
        selections[stage] = render(analysis, row, args.output / stage,
                                   f"{args.platform.upper()} · TRTLLM BF16 · {stage}", args.png)
    result = {"schema": "trtllm-platform-profile-evidence-v1", "platform": args.platform,
              "main_run_id": args.main_run_id, "capture_run_id": plan["identity"]["run_id"],
              "image_reference": plan["identity"]["image_reference"],
              "plan_sha256": args.plan_sha256, "terminal_sha256": args.terminal_sha256,
              "frozen_input_sha256": plan["frozen_input_sha256"],
              "input_ids_sha256": plan["input_ids_sha256"], "workload_sha256": plan["workload_sha256"],
              "measurements": measurements, "profiles": selections,
              "decode_forward_count": sum(r["stage"] == "DECODE" for a in analyses for r in a["forwards"]),
              "correlated_decode_replay_count": sum(r.get("decode_replay_proven", False) for a in analyses for r in a["forwards"]),
              "scope": "Instrumented GPU annotation windows; kernel sums and interval unions retained separately. "
                       "The three HTTP measurements include prefill/decode/queue/response and are not main Miles timings."}
    write(args.output / "profile-evidence.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", required=True, type=Path)
    p.add_argument("--platform", choices=("rubin", "gb300"), required=True)
    p.add_argument("--main-run-id", required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--terminal-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--png", action="store_true")
    result = build(p.parse_args())
    print(json.dumps({k: result[k] for k in ("platform", "correlated_decode_replay_count", "decode_forward_count")}, indent=2))


if __name__ == "__main__":
    main()
