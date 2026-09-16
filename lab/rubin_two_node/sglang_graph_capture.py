#!/usr/bin/env python3
"""Offline payload/evidence helper for short, independent SGLang diagnostics.

This script never starts an engine, sends requests, or changes a running job.
Use the printed payload only after the operator verifies a dedicated diagnostic
engine and its deadline. Keep diagnostic requests/traces out of learning curves.
The pinned legacy profiler may put interleaved prefill in a DECODE-named file;
classify actual step annotations, never the filename or a graph-enabled flag.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

SGLANG_COMMIT = "aea7fb92c047c9c096eae66460acb25dec9ae5a9"
MAX_DECODED_BYTES = 128 * 1024**2
MAX_TOKEN_FILE_BYTES = 2 * 1024**2
STEP = re.compile(r"^step\[([A-Z_]+)\b([^\]]*)\]")
FIELD = re.compile(r"\b([a-z_]+)=([0-9]+)\b")
GRAPH_LAUNCH = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_ptsz|_v[0-9]+)?$")


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generation_request(token_input, run_id, max_new_tokens=64, context_limit=1536):
    """Build a small diagnostic request from frozen, externally tokenized inputs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("Use a simple unique diagnostic run ID")
    if type(max_new_tokens) is not int or not 8 <= max_new_tokens <= 64:
        raise ValueError("Diagnostic output must be bounded to 8–64 new tokens")
    if type(context_limit) is not int or not 1 <= context_limit <= 1536:
        raise ValueError("Context limit must be explicit and at most 1536")
    if not isinstance(token_input, dict):
        raise ValueError("Expected token-input object")
    inputs, source = token_input.get("input_ids"), token_input.get("source")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 128:
        raise ValueError("Expected one to 128 tokenized prompts")
    for prompt in inputs:
        if (not isinstance(prompt, list) or not 1 <= len(prompt) <= 512
                or len(prompt) + max_new_tokens > context_limit
                or any(type(token) is not int or not 0 <= token < 2**63 for token in prompt)):
            raise ValueError("Each prompt must contain 1–512 valid token IDs and fit the context limit")
    if (not isinstance(source, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("dataset_sha256", "")))
            or not isinstance(source.get("tokenizer_model"), str) or not source["tokenizer_model"]
            or not isinstance(source.get("chat_template_kwargs"), dict)):
        raise ValueError("Record dataset SHA256, tokenizer model and explicit chat-template kwargs")
    rows = source.get("selected_row_indices")
    if (not isinstance(rows, list) or len(rows) != len(inputs)
            or any(type(row) is not int or row < 0 for row in rows)):
        raise ValueError("Record one nonnegative dataset row index per prompt")
    request = {"rid": [f"{run_id}-{i}" for i in range(len(inputs))], "input_ids": inputs,
               "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                   "max_new_tokens": max_new_tokens, "ignore_eos": True,
                                   "sampling_seed": 1234},
               "stream": False}
    return {"mode": "OFFLINE_ONLY_NO_API", "sglang_commit": SGLANG_COMMIT,
            "request": request, "request_sha256": json_sha256(request),
            "workload_sha256": json_sha256({k: v for k, v in request.items() if k != "rid"}),
            "source": source, "input_ids_sha256": json_sha256(inputs),
            "prompt_token_counts": list(map(len, inputs)), "context_limit": context_limit,
            "interpretation": "Short generation-only diagnostic; ignore_eos applies only here. Preserve identical tokens, engine seed and sampling policy across OFF/ON. Do not use these outputs as learning metrics."}


def stage_payload(output_dir, profile_id, num_steps=4):
    path = PurePosixPath(output_dir)
    if (not path.is_absolute() or path.parts[:2] != ("/", "run-output")
            or ".." in path.parts or len(path.parts) < 3):
        raise ValueError("Use a scoped /run-output/... directory in the diagnostic container")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", profile_id):
        raise ValueError("Use a simple unique profile ID")
    if type(num_steps) is not int or not 1 <= num_steps <= 4:
        raise ValueError("This short diagnostic permits one to four steps per stage")
    return {"output_dir": str(path), "profile_id": profile_id, "num_steps": num_steps,
            "activities": ["CPU", "GPU"], "profile_by_stage": True,
            "with_stack": False, "record_shapes": False, "merge_profiles": False,
            "detailed_annotations": True}


def complete_event(event):
    return (isinstance(event, dict) and event.get("ph") == "X"
            and all(type(event.get(k)) in (int, float) and math.isfinite(event[k])
                    for k in ("ts", "dur")) and event["dur"] > 0)


def analyze_events(events):
    if not isinstance(events, list):
        raise ValueError("Expected a Chrome traceEvents list")
    valid = [e for e in events if complete_event(e)]
    categories = Counter(e.get("cat", "") for e in valid)
    steps = []
    for event in valid:
        match = STEP.match(str(event.get("name", "")))
        if event.get("cat") != "user_annotation" or not match:
            continue  # GPU annotation mirrors must not double-count forwards.
        fields = {key: int(value) for key, value in FIELD.findall(event["name"])}
        steps.append({"mode": match[1], "name": event["name"], "fields": fields,
                      **{k: event.get(k) for k in ("pid", "tid", "ts", "dur")}})
    launches = [e for e in valid if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                and GRAPH_LAUNCH.fullmatch(str(e.get("name", "")))]
    matched = []
    for event in launches:
        spans = [s for s in steps if s["mode"] == "DECODE"
                 and s["pid"] is not None and s["tid"] is not None
                 and s["pid"] == event.get("pid") and s["tid"] == event.get("tid")
                 and s["ts"] <= event["ts"]
                 and event["ts"] + event["dur"] <= s["ts"] + s["dur"]]
        if spans:
            matched.append({"name": event["name"], "ts": event["ts"],
                            "dur": event["dur"], "decode_step": spans[0]["name"]})
    modes = Counter(s["mode"] for s in steps)
    prefill = [s for s in steps if s["mode"] in {"EXTEND", "MIXED"}]
    evidence = bool(matched) and categories["kernel"] > 0
    return {"event_count": len(events), "complete_event_categories": dict(categories),
            "scheduler_forward_counts": dict(modes), "scheduler_forwards": steps,
            "mixed_forward_modes_observed": len(modes) > 1 or bool(modes["MIXED"]),
            "prefill_forward_observed": bool(prefill),
            "prefill_new_query_token_totals": [s["fields"].get("c_sq") for s in prefill],
            "prefill_scope": "Only these actual annotated forwards; long-prefill representativeness is not inferred",
            "graph_launch_count": len(launches), "decode_graph_launches": matched,
            "decode_graph_replay_proven": evidence,
            "decode_graph_evidence_rule": "CUDA graph launch on the same CPU PID/TID inside a DECODE step, with actual CUDA kernels in this trace",
            "cpu_and_gpu_activity_observed": categories["kernel"] > 0 and any(
                categories[c] > 0 for c in ("cpu_op", "cuda_runtime", "cuda_driver")),
            "interpretation": "Instrumented diagnostic only; absence of proof does not prove graphs disabled, and timings are not unprofiled training throughput"}


def inspect_trace(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refuse trace symlinks")
    before = path.stat()
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_DECODED_BYTES + 1)
    if len(raw) > MAX_DECODED_BYTES:
        raise ValueError("Trace exceeds the small-capture decoded-byte limit; do not use native full-training traces")
    data = json.loads(raw)  # Bounded, complete JSON; partial gzip/JSON is rejected.
    result = analyze_events(data.get("traceEvents"))
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Trace changed during inspection")
    return {"path": str(path.resolve()), "compressed_bytes": after.st_size,
            "decoded_bytes": len(raw), "sha256": hasher.hexdigest(), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    payload = sub.add_parser("payload", help="Print a request, with no network or GPU activity")
    payload.add_argument("--output-dir", required=True)
    payload.add_argument("--profile-id", required=True)
    payload.add_argument("--num-steps", type=int, default=4)
    request = sub.add_parser("request", help="Build an offline /generate body from frozen token IDs")
    request.add_argument("--token-input", type=Path, required=True)
    request.add_argument("--run-id", required=True)
    request.add_argument("--max-new-tokens", type=int, default=64)
    request.add_argument("--context-limit", type=int, default=1536)
    inspect = sub.add_parser("inspect", help="Validate small retained gzip traces and actual graph replay evidence")
    inspect.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "payload":
        result = {"mode": "OFFLINE_ONLY_NO_API", "sglang_commit": SGLANG_COMMIT,
                  "payload": stage_payload(args.output_dir, args.profile_id, args.num_steps),
                  "required_runtime": "Pinned legacy SGLANG_PROFILE_V2=0; identity/lease checked by the external operator",
                  "expected_suffixes": ["-EXTEND.trace.json.gz", "-DECODE.trace.json.gz"],
                  "caveats": ["First decode flushes prefill early; four prefill forwards are not guaranteed",
                              "Interleaved prefill can occur in the decode-named trace",
                              "Do not set profile_stages: the pinned legacy path ignores that field",
                              "Use a separate bounded generation-only run, not extra requests against the learning run"]}
    elif args.command == "request":
        if args.token_input.is_symlink():
            raise ValueError("Refuse token-input symlinks")
        with args.token_input.open("rb") as stream:
            raw = stream.read(MAX_TOKEN_FILE_BYTES + 1)
        if len(raw) > MAX_TOKEN_FILE_BYTES:
            raise ValueError("Token-input file exceeds the 2 MiB diagnostic bound")
        result = generation_request(json.loads(raw), args.run_id, args.max_new_tokens, args.context_limit)
        result["token_input_file_sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        results = [inspect_trace(p) for p in args.traces]
        result = {"mode": "OFFLINE_TRACE_EVIDENCE", "traces": results,
                  "decode_graph_replay_proven": any(r["decode_graph_replay_proven"] for r in results),
                  "prefill_forward_observed": any(r["prefill_forward_observed"] for r in results)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
