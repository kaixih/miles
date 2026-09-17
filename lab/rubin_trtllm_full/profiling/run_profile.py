#!/usr/bin/env python3
"""ON-only TRTLLM diagnostic; reuse the previously verified bounded capture.

This runs inside a dedicated, externally verified container after main training
has finished. Default is plan-only. It never allocates a node, starts Ray, edits
the installed backend, or modifies a main run. The caller owns the exact
container and must arm its independent absolute-deadline guard.
"""
import argparse
import datetime as dt
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

LAB = Path(__file__).resolve().parents[2] / "rubin_two_node"
sys.path.insert(0, str(LAB))
import run_sglang_graph_diagnostic as base
import sglang_graph_capture as capture

BACKEND = "flashinfer_trtllm"
DATASET_SHA = "f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8"
ORIGINAL_ENGINE_ARGV = base.engine_argv
ORIGINAL_HTTP = base.http


def load_json(path, maximum=2 * 1024**2):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Expected a small regular JSON file: " + str(path))
    return json.loads(path.read_text())


def normalized_inputs(value):
    # A mount path is provenance, not token identity.
    source = dict(value["source"])
    source.pop("tokenizer_model", None)
    return {"input_ids": value["input_ids"], "source": source}


def engine_argv(args, mode):
    if mode != "on":
        raise ValueError("This experiment captures decode graph ON only")
    argv = ORIGINAL_ENGINE_ARGV(args, mode)
    argv[argv.index("--moe-runner-backend") + 1] = BACKEND
    return argv


def checked_http(origin, path, payload=None, timeout=10):
    result = ORIGINAL_HTTP(origin, path, payload, timeout)
    if path == "/server_info":
        info = json.loads(result["body"])
        if (info.get("moe_runner_backend") != BACKEND or info.get("dtype") != "bfloat16"
                or info.get("attention_backend") != "triton"
                or info.get("bf16_gemm_backend") != "torch"):
            raise ValueError("Actual server backend/dtype differs from the matched TRTLLM diagnostic")
    return result


def tokenize(args):
    from transformers import AutoTokenizer
    actual = base.file_sha(args.dataset)
    if actual != DATASET_SHA:
        raise ValueError("GSM8K training dataset differs from the matched main recipe")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=True)
    result = base.freeze_inputs(args.dataset, tokenizer, args.model_path, actual)
    if base.file_sha(args.dataset) != actual:
        raise ValueError("Dataset changed during tokenization")
    capture.generation_request(result, "trtllm-frozen-input-audit")
    return result


def freeze(args):
    result = tokenize(args)
    if args.output.exists() or any(p.is_symlink() for p in [args.output, *args.output.parents]):
        raise ValueError("Frozen input requires a new nonsymlink output file")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(args.output), "sha256": base.file_sha(args.output),
                      "requests": len(result["input_ids"]),
                      "input_tokens": sum(map(len, result["input_ids"]))}, indent=2))


def validate_plan(args):
    match = re.fullmatch(r"[^\s@]+@sha256:([0-9a-f]{64})", args.image)
    if not match:
        raise ValueError("Require the tested immutable platform image reference")
    identity = load_json(args.identity_file)
    if identity.get("image_reference") != args.image or identity.get("main_run_id") != args.main_run_id:
        raise ValueError("Image/main identity differs from the operator's inspected container")
    base.IMAGE_DIGESTS = {match.group(1)}
    run_id = base.validate_identity(identity, args.execute)
    if base.file_sha(args.frozen_token_input) != args.frozen_input_sha256:
        raise ValueError("Frozen input SHA differs from the one shared between both platforms")
    inputs = load_json(args.frozen_token_input)
    request = capture.generation_request(inputs, "trtllm-frozen-input-audit")
    if len(inputs["input_ids"]) != 128 or inputs["source"]["dataset_sha256"] != DATASET_SHA:
        raise ValueError("Require exactly 128 frozen prompts from the matched dataset")
    ipaddress.IPv4Address(args.host)
    deadline = base.timestamp(args.deadline_utc)
    if (not 0 < deadline - time.time() <= 1800 or not 1024 <= args.port <= 65535
            or not 64 <= args.max_rss_gib <= 256 or ".." in args.output_dir.parts
            or args.output_dir.parts[:2] != ("/", "run-output") or len(args.output_dir.parts) < 3
            or ".." in args.cache_dir.parts or args.cache_dir.parts[:2] != ("/", "cache")):
        raise ValueError("Require future <=30-minute deadline, explicit port, bounded RSS and scoped paths")
    plan = {"schema": "trtllm-prefill-decode-profile-v1", "at": base.utc(), "identity": identity,
            "main_run_id": args.main_run_id, "deadline_utc": args.deadline_utc,
            "order": ["on"], "engine_argv": {"on": engine_argv(args, "on")},
            "frozen_input_sha256": args.frozen_input_sha256,
            "input_ids_sha256": request["input_ids_sha256"], "workload_sha256": request["workload_sha256"],
            "fresh_jit_cache_root": str(args.cache_dir / run_id),
            "workload": {"requests": 128, "input_tokens": sum(request["prompt_token_counts"]),
                         "output_tokens_per_request": 64, "warmups": 1, "measurements": 3,
                         "profile_batches": 1, "max_scheduler_forwards_per_stage": 4,
                         "decode_graph": True, "prefill_graph": False},
            "scope": "Separate initial-policy one-TP1-engine diagnostic after terminal main training. "
                     "Main correctness and performance contain no profiler samples.",
            "limits": {"artifact_bytes": 1024**3, "engine_rss_bytes": args.max_rss_gib * 1024**3},
            "source_sha256": {p.name: base.file_sha(p) for p in
                              (Path(__file__), Path(base.__file__), Path(capture.__file__))}}
    return plan, inputs, run_id, deadline


def run(args):
    plan, inputs, run_id, deadline = validate_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    cache = args.cache_dir / run_id
    for path in (args.output_dir, cache):
        if path.exists() or any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("Require fresh output and cache directories without symlink parents")
    args.output_dir.mkdir(parents=True)
    base.save(args.output_dir / "plan.json", plan)
    runtime = base.Runtime(args.output_dir, run_id, deadline, args.max_rss_gib * 1024**3)
    guard = threading.Thread(target=runtime.guard, daemon=True)
    guard.start()
    def interrupted(signum, frame):
        raise RuntimeError("TRTLLM diagnostic received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        cache.mkdir(parents=True)
        retokenized = tokenize(args)
        if normalized_inputs(retokenized) != normalized_inputs(inputs):
            raise ValueError("Installed tokenizer does not reproduce the shared frozen token IDs")
        base.save(args.output_dir / "frozen-token-input.json", inputs)
        versions = {}
        for name in ("torch", "sglang", "transformers", "triton", "flashinfer-python"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        base.save(args.output_dir / "package-versions.json", versions)
        base.engine_argv, base.http = engine_argv, checked_http
        result = base.run_mode(args, runtime, "on", inputs)
        proof = result["capture"]
        if not (proof["prefill_observed"] and proof["four_decode_forwards_verified"]
                and proof["decode_graph_replay_proven"]):
            raise ValueError("Retained capture lacks complete prefill plus four graph decode forwards")
        base.save(args.output_dir / "terminal.json", {"at": base.utc(), "status": "COMPLETED",
                                                     "results": [result]})
    except BaseException as error:
        base.save(args.output_dir / "terminal.json", {"at": base.utc(), "status": "FAILED", "error": repr(error)})
        raise
    finally:
        runtime.stop()
        runtime.done.set()
        guard.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze", help="CPU-only: freeze once; bind this identical file on both platforms")
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = commands.add_parser("run", help="Plan-only unless --execute is supplied")
    p.add_argument("--identity-file", required=True, type=Path)
    p.add_argument("--image", required=True)
    p.add_argument("--main-run-id", required=True)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--frozen-token-input", required=True, type=Path)
    p.add_argument("--frozen-input-sha256", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("/cache/sglang-trtllm-profile"))
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--deadline-utc", required=True)
    p.add_argument("--max-rss-gib", type=int, default=192)
    p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    freeze(args) if args.command == "freeze" else run(args)


if __name__ == "__main__":
    main()
