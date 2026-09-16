#!/usr/bin/env python3
"""Bounded generation-only OFF/ON diagnostic inside a dedicated container.

Default is a CPU-only plan. --execute starts one owned TP1 engine at a time.
The external operator must verify Docker identity, read-only model/data binds,
terminal main runs, one visible GPU, and the allocation's original deadline.
This helper never invokes Ray, Docker, Miles training, or changes model files.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import sglang_graph_capture as capture

RUN_ENV = "MILES_PROFILE_RUN_ID"
IMAGE_DIGESTS = {
    "a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c",
    "226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadline must include its UTC offset")
    return parsed.timestamp()


def save(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def validate_identity(identity, runtime=False):
    run = identity.get("run_id", "")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run)
            or identity.get("uid") != 28644 or identity.get("gid") != 30
            or identity.get("labels", {}).get("miles.graph_diagnostic") != run
            or not re.fullmatch(r"[0-9a-f]{64}", identity.get("container_id", ""))
            or not identity.get("container_name", "").startswith("miles-graph-diagnostic-")
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity.get("image_id", ""))
            or identity.get("image_reference", "").split("@sha256:")[-1] not in IMAGE_DIGESTS
            or "@sha256:" not in identity.get("image_reference", "")
            or identity.get("main_terminal_verified") is not True
            or not isinstance(identity.get("source_commits"), dict)
            or not identity.get("docker_inspect_verified_at")):
        raise ValueError("Require the operator's exact dedicated-container identity and terminal-main attestation")
    if runtime and ((os.getuid(), os.getgid()) != (28644, 30)
                    or os.environ.get(RUN_ENV) != run
                    or not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", os.environ.get("CUDA_VISIBLE_DEVICES", ""))):
        raise ValueError("Runtime UID/GID, run-ID environment or single visible GPU mismatch")
    return run


def engine_argv(args, mode):
    argv = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(args.model_path),
            "--host", args.host, "--port", str(args.port), "--tp-size", "1", "--pp-size", "1",
            "--base-gpu-id", "0", "--trust-remote-code", "--dtype", "bfloat16",
            "--context-length", "1536", "--mem-fraction-static", "0.55",
            "--max-running-requests", "128", "--attention-backend", "triton",
            "--bf16-gemm-backend", "torch", "--moe-runner-backend", "triton",
            "--random-seed", "1234", "--enable-memory-saver", "--enable-metrics",
            "--skip-server-warmup", "--disable-piecewise-cuda-graph"]
    if mode == "off":
        argv.append("--disable-cuda-graph")
    elif mode == "on":
        argv += ["--cuda-graph-config", '{"decode":{"backend":"full"},"prefill":{"backend":"disabled"}}']
    else:
        raise ValueError("Mode must be off or on")
    return argv


def freeze_inputs(dataset, tokenizer, model_path, dataset_sha):
    ids, indices = [], []
    with dataset.open() as stream:
        for index, line in enumerate(stream):
            if len(line) > 1024**2:
                raise ValueError("Oversized dataset row")
            row = json.loads(line)
            messages = row["prompt"]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if not isinstance(messages, list):
                raise ValueError("Expected prompt string or chat messages")
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if 1 <= len(tokens) <= 512:
                ids.append(tokens)
                indices.append(index)
            if len(ids) == 128:
                break
    if len(ids) != 128:
        raise ValueError("Need 128 eligible prompts; no duplication, truncation or template override")
    return {"input_ids": ids, "source": {"dataset_sha256": dataset_sha,
            "selected_row_indices": indices, "tokenizer_model": str(model_path),
            "chat_template_kwargs": {}, "add_generation_prompt": True,
            "tokenize_rendered_with_add_special_tokens": False}}


def http(origin, path, payload=None, timeout=10):
    raw = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(origin + path, data=raw,
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        body = response.read(8 * 1024**2 + 1)
        if len(body) > 8 * 1024**2:
            raise ValueError("API response exceeds diagnostic bound")
        return {"status": response.status, "body": body.decode(), "at": utc()}


def flush_cache(origin):
    response = http(origin, "/flush_cache?timeout=10", {}, timeout=15)
    if response["status"] != 200 or not response["body"].startswith("Cache flushed.\n"):
        raise ValueError("Pinned API did not confirm successful radix-cache flush")
    return response


def batch_metrics(response, expected, elapsed):
    data = json.loads(response["body"])
    if response["status"] != 200 or not isinstance(data, list) or len(data) != expected:
        raise ValueError("Incomplete diagnostic generation batch")
    meta = [row.get("meta_info", {}) for row in data]
    for row in meta:
        for key in ("prompt_tokens", "completion_tokens"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError("Missing actual prompt/completion token counts")
        if row["completion_tokens"] != 64:
            raise ValueError("Short diagnostic did not complete exactly 64 tokens per request")
    inputs = sum(row["prompt_tokens"] for row in meta)
    outputs = sum(row["completion_tokens"] for row in meta)
    if elapsed <= 0:
        raise ValueError("Generation request duration must be positive")
    return {"requests": expected, "generation_request_seconds": elapsed, "input_tokens": inputs,
            "output_tokens": outputs, "output_tokens_per_second": outputs / elapsed,
            "input_plus_output_tokens_per_second": (inputs + outputs) / elapsed,
            "cached_tokens_per_request": [row.get("cached_tokens") for row in meta],
            "prompt_tokens_per_request": [row["prompt_tokens"] for row in meta],
            "completion_tokens_per_request": [row["completion_tokens"] for row in meta],
            "raw_timing_fields_per_request": [{k: v for k, v in row.items()
                if any(word in k.lower() for word in ("time", "latency", "ttft", "itl"))} for row in meta],
            "scope": "HTTP request wall time includes prefill, decode, queueing and response delivery; not decode-only. One TP1 engine with cold radix cache per batch; excludes model load, graph construction, flush and warmup; differs from four-engine main host contention"}


def process_info(pid):
    p = Path("/proc") / str(pid)
    try:
        stat = p.joinpath("stat").read_text().rpartition(") ")[2].split()
        if stat[0] == "Z":
            return None
        status = dict(line.split(":", 1) for line in p.joinpath("status").read_text().splitlines() if ":" in line)
        env = dict(x.split(b"=", 1) for x in p.joinpath("environ").read_bytes().split(b"\0") if b"=" in x)
        return {"pid": pid, "ppid": int(stat[1]), "state": stat[0], "pgid": int(stat[2]),
                "sid": int(stat[3]), "start_ticks": int(stat[19]),
                "uid": int(status["Uid"].split()[0]), "rss_bytes": int(status.get("VmRSS", "0").split()[0]) * 1024,
                "run_id": env.get(RUN_ENV.encode(), b"").decode(),
                "run_env_present": RUN_ENV.encode() in env,
                "argv_sha256": hashlib.sha256(p.joinpath("cmdline").read_bytes()).hexdigest()}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        # A process can become a zombie after the initial stat read; its
        # environ can then become unreadable. Never excuse a live process.
        try:
            if p.joinpath("stat").read_text().rpartition(") ")[2].split()[0] == "Z":
                return None
        except (FileNotFoundError, ProcessLookupError):
            return None
        raise


def group_members(pgid):
    members = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            # Inspect status first: unrelated users' environ is not readable.
            try:
                fields = p.joinpath("stat").read_text().rpartition(") ")[2].split()
                if int(fields[2]) != pgid or fields[0] == "Z":
                    continue
                info = process_info(int(p.name))
                if info:
                    members.append(info)
            except (FileNotFoundError, ProcessLookupError):
                continue
    return members


def group_alive(pgid):
    """Read-only wait predicate; never authorizes a signal or reads environ."""
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = p.joinpath("stat").read_text().rpartition(") ")[2].split()
            if int(stat[2]) == pgid and stat[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


class OwnershipError(RuntimeError):
    def __init__(self, reason, member, leader):
        fields = ("pid", "ppid", "sid", "pgid", "uid", "start_ticks", "run_env_present", "run_id")
        self.detail = {"reason": reason, "member": {k: member.get(k) for k in fields} if member else None,
                       "original_leader": {k: leader.get(k) for k in fields} if leader else None}
        super().__init__("Engine ownership rejected; refusing to signal: " + json.dumps(self.detail))


def verify_owned_group(members, leader, run_id, uid):
    # Popen(start_new_session=True) must establish an authenticated session anchor.
    # setproctitle can overwrite the original /proc environ memory after launch;
    # an existing verified session cannot be joined from an unrelated session.
    if (not leader or leader.get("pid") != leader.get("pgid") or leader.get("pid") != leader.get("sid")
            or leader.get("uid") != uid or type(leader.get("start_ticks")) is not int
            or leader["start_ticks"] <= 0 or leader.get("run_env_present") is not True
            or leader.get("run_id") != run_id):
        raise OwnershipError("unverified_original_session_anchor", leader, leader)
    for member in members:
        if (member.get("sid") != leader["sid"] or member["pgid"] != leader["pgid"]
                or member["uid"] != uid or member["start_ticks"] < leader["start_ticks"]
                or (member["pid"] == leader["pid"] and member["start_ticks"] != leader["start_ticks"])):
            raise OwnershipError("session_group_uid_or_start_ticks_changed", member, leader)
        if (member.get("run_env_present") not in (True, False)
                or (member["run_env_present"] and member.get("run_id") != run_id)
                or (not member["run_env_present"] and member.get("run_id") not in (None, ""))):
            raise OwnershipError("explicit_run_environment_mismatch", member, leader)


class Runtime:
    def __init__(self, output, run_id, deadline, rss_limit):
        self.output, self.run_id, self.deadline, self.rss_limit = output, run_id, deadline, rss_limit
        self.lock, self.done = threading.RLock(), threading.Event()
        self.child, self.leader = None, None
        self.events = []

    def record(self, event_kind, **values):
        with self.lock:
            self.events.append({"at": utc(), "event": event_kind, **values})
            save(self.output / "events.json", self.events)

    def start(self, argv, log, env):
        with self.lock:
            if self.child is not None:
                raise RuntimeError("Previous engine has not exited")
            self.child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
            self.leader = process_info(self.child.pid)
            if not self.leader:
                raise RuntimeError("Engine exited before identity capture")
            verify_owned_group([self.leader], self.leader, self.run_id, os.getuid())
            self.record("engine_started", identity=self.leader, argv=argv)

    def stop(self):
        with self.lock:
            if self.child is None:
                return
            members = group_members(self.child.pid)
            verify_owned_group(members, self.leader, self.run_id, os.getuid())
            if members:
                os.killpg(self.child.pid, signal.SIGTERM)
            until = time.monotonic() + 5
            # TERM can make environ unreadable before the process reaches Z.
            # Waiting needs only liveness; any further signal still requires
            # the full fresh ownership verification below.
            while time.monotonic() < until and group_alive(self.child.pid):
                time.sleep(.2)
            remaining = group_members(self.child.pid)
            verify_owned_group(remaining, self.leader, self.run_id, os.getuid())
            if remaining:
                os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait(timeout=5)
            if group_alive(self.child.pid):
                raise RuntimeError("Owned engine group did not exit; do not start another mode")
            self.record("engine_stopped", pid=self.child.pid, returncode=self.child.returncode)
            self.child, self.leader = None, None

    def inspect_limits(self):
        if time.time() >= self.deadline:
            return "absolute_deadline"
        total = 0
        for parent, dirs, files in os.walk(self.output, onerror=lambda e: (_ for _ in ()).throw(e)):
            for name in dirs + files:
                path = Path(parent) / name
                if path.is_symlink():
                    return "output_symlink"
            for name in files:
                try:
                    total += (Path(parent) / name).stat().st_size
                except FileNotFoundError:
                    pass  # Atomic receipt replacement can remove a just-listed .tmp file.
        if total > 1024**3:
            return "artifact_limit_1GiB"
        with self.lock:
            members = group_members(self.child.pid) if self.child else []
            if self.child:
                verify_owned_group(members, self.leader, self.run_id, os.getuid())
            if sum(m["rss_bytes"] for m in members) > self.rss_limit:
                return "engine_RSS_limit"
        return None

    def guard(self):
        while not self.done.wait(1):
            identity_error = None
            try:
                reason = self.inspect_limits()
                if not reason:
                    continue
            except Exception as error:
                reason = "guard_error: " + repr(error)
                identity_error = error.detail if isinstance(error, OwnershipError) else None
            terminal = {"at": utc(), "status": "BOUNDED_STOP", "reason": reason,
                        "ownership_error": identity_error, "engine_cleanup": "not_attempted"}
            try:
                try:
                    self.record("guard_stopping", reason=reason, ownership_error=identity_error)
                except Exception as error:
                    terminal["event_write_error"] = repr(error)
                try:
                    self.stop()
                    terminal["engine_cleanup"] = "verified_stopped"
                except Exception as error:
                    terminal.update(engine_cleanup="FAILED", cleanup_error=repr(error),
                        cleanup_ownership_error=error.detail if isinstance(error, OwnershipError) else None)
            finally:
                try:
                    save(self.output / "terminal.json", terminal)
                except Exception as error:
                    print(json.dumps({**terminal, "terminal_write_error": repr(error)}), file=sys.stderr, flush=True)
                finally:
                    os._exit(124)  # Bound remains unchanged even when cleanup or receipt writing fails.


def wait_ready(runtime, origin, host, port):
    deadline = min(runtime.deadline, time.time() + 1200)
    while time.time() < deadline:
        if runtime.child.poll() is not None:
            raise RuntimeError("Engine exited during startup")
        # Require the actual owned launch PID to hold this listener before HTTP.
        owned = set()
        for fd in (Path("/proc") / str(runtime.child.pid) / "fd").iterdir():
            try:
                link = os.readlink(fd)
                if link.startswith("socket:["):
                    owned.add(link[8:-1])
            except FileNotFoundError:
                continue
        listens = any(fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port and fields[9] in owned
                      for name in ("tcp", "tcp6") for line in (Path("/proc/net") / name).read_text().splitlines()[1:]
                      for fields in [line.split()])
        if listens:
            try:
                response = http(origin, "/health", timeout=2)
                if response["status"] == 200:
                    return
            except (OSError, ValueError):
                pass
        time.sleep(2)
    raise TimeoutError("Engine startup exhausted its bounded budget")


def run_mode(args, runtime, mode, token_input):
    directory = runtime.output / mode
    directory.mkdir()
    env = {**os.environ, RUN_ENV: runtime.run_id, "SGLANG_PROFILE_V2": "0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # Separate fresh JIT caches, equal policy; model/tokenizer caches remain read-only inputs.
    for key, child in [("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                       ("FLASHINFER_WORKSPACE_BASE", "flashinfer")]:
        env[key] = str(args.cache_dir / runtime.run_id / mode / child)
    origin = f"http://{args.host}:{args.port}"
    with socket.socket() as probe:
        probe.bind((args.host, args.port))  # Refuse an occupied port before starting an engine.
    with (directory / "engine.log").open("w") as log:
        runtime.start(engine_argv(args, mode), log, env)
        try:
            wait_ready(runtime, origin, args.host, args.port)
            info = http(origin, "/server_info")
            save(directory / "server-info.json", info)
            parsed = json.loads(info["body"])
            if (parsed.get("model_path") != str(args.model_path) or parsed.get("tp_size") != 1
                    or parsed.get("pp_size") != 1 or parsed.get("base_gpu_id") != 0):
                raise ValueError("Server runtime model/layout differs from the owned diagnostic")
            graph = parsed.get("cuda_graph_config", {})
            if (graph.get("decode", {}).get("backend") != ("full" if mode == "on" else "disabled")
                    or graph.get("prefill", {}).get("backend") != "disabled"):
                raise ValueError("Resolved server graph policy differs from the requested decode-only comparison")
            result = {"mode": mode, "engine_identity": runtime.leader, "measurements": [],
                      "gc_policy": "Engine startup defaults with identical --skip-server-warmup in both modes; no explicit freeze_gc/unfreeze_gc request. Inspect retained engine.log for startup races/warnings."}
            for iteration in range(4):
                kind = "warmup" if iteration == 0 else "measured"
                flush = flush_cache(origin)
                request = capture.generation_request(token_input, f"{runtime.run_id}-{mode}-{iteration}")
                start = time.monotonic()
                response = http(origin, "/generate", request["request"], timeout=min(300, max(1, runtime.deadline - time.time())))
                elapsed = time.monotonic() - start
                metrics = batch_metrics(response, 128, elapsed)
                record = {"kind": kind, "iteration": iteration, "flush": flush, "request": request, "response": response, "metrics": metrics}
                save(directory / f"{kind}-{iteration}.json", record)
                if kind == "measured":
                    result["measurements"].append(metrics)
                runtime.record("batch_complete", mode=mode, kind=kind, iteration=iteration, metrics=metrics)
            traces = directory / "traces"
            traces.mkdir()
            profile_id = f"{runtime.run_id}-{mode}"
            payload = capture.stage_payload(str(traces), profile_id)
            flush = flush_cache(origin)
            claim = directory / "capture-claim.json"
            with claim.open("x") as stream:
                json.dump({"at": utc(), "engine": runtime.leader, "payload": payload}, stream)
            capture_deadline = min(runtime.deadline, time.time() + 180)
            arm = http(origin, "/start_profile", payload)
            if arm["status"] != 200:
                raise ValueError("Stage profiler arm failed")
            request = capture.generation_request(token_input, f"{runtime.run_id}-{mode}-profile")
            response = http(origin, "/generate", request["request"], timeout=max(1, capture_deadline - time.time()))
            batch_metrics(response, 128, 1)  # Validate completion; instrumented time is not a measurement.
            save(directory / "capture-request.json", {"flush": flush, "arm": arm, "request": request, "response": response})
            evidence, errors = [], []
            previous = None
            while time.time() < capture_deadline:
                paths = sorted(traces.glob("*.trace.json.gz"))
                sizes = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
                if len(paths) >= 2 and sizes == previous:
                    try:
                        evidence = [capture.inspect_trace(p) for p in paths]
                        break
                    except (ValueError, OSError, EOFError) as error:
                        errors.append(repr(error))
                previous = sizes
                time.sleep(2)
            result["capture"] = {"deadline_utc": dt.datetime.fromtimestamp(capture_deadline, dt.timezone.utc).isoformat(),
                                 "trace_evidence": evidence, "inspection_errors": errors[-4:],
                                 "prefill_observed": any(e["prefill_forward_observed"] for e in evidence),
                                 "decode_graph_replay_proven": any(e["decode_graph_replay_proven"] for e in evidence),
                                 "complete_trace_count": len(evidence),
                                 "decode_forward_count": sum(e["scheduler_forward_counts"].get("DECODE", 0) for e in evidence)}
            result["capture"]["four_decode_forwards_verified"] = (result["capture"]["decode_forward_count"] == 4
                and all(e["cpu_and_gpu_activity_observed"] for e in evidence))
            save(directory / "summary.json", result)
            return result
        finally:
            primary = sys.exc_info()[1]
            try:
                runtime.stop()
            except Exception as cleanup_error:
                if primary is not None:
                    save(directory / "cleanup-failure.json", {"at": utc(),
                        "primary_error": repr(primary), "cleanup_error": repr(cleanup_error)})
                    raise primary from cleanup_error
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/sglang-graph-diagnostic"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--order", choices=("off,on", "on,off"), required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--max-rss-gib", type=int, default=192)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    identity = json.loads(args.identity_file.read_text())
    run_id = validate_identity(identity, args.execute)
    ipaddress.IPv4Address(args.host)
    deadline = timestamp(args.deadline_utc)
    if (not 0 < deadline - time.time() <= 1800 or not 1024 <= args.port <= 65535
            or not 64 <= args.max_rss_gib <= 256 or ".." in args.output_dir.parts
            or args.output_dir.parts[:2] != ("/", "run-output") or len(args.output_dir.parts) < 3
            or ".." in args.cache_dir.parts or args.cache_dir.parts[:2] != ("/", "cache")):
        raise ValueError("Require future ≤30-minute deadline, explicit port, bounded RSS and scoped /run-output directory")
    plan = {"at": utc(), "identity": identity, "deadline_utc": args.deadline_utc,
            "order": args.order.split(","), "engine_argv": {mode: engine_argv(args, mode) for mode in ("off", "on")},
            "fresh_jit_cache_root": str(args.cache_dir / run_id),
            "workload": "128 eligible GSM8K prompts; 64 output tokens; default thinking template; seed1234; cold radix cache before every batch; warmup1/measured3/profile1 per mode",
            "scope": "One TP1 GPU, generation only, separate from terminal main; host contention differs from four-engine training. No causal hardware claim.",
            "limits": {"artifact_bytes": 1024**3, "engine_rss_bytes": args.max_rss_gib * 1024**3},
            "source_sha256": {p.name: file_sha(p) for p in (Path(__file__), Path(capture.__file__))}}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if args.output_dir.exists() or any(p.is_symlink() for p in [args.output_dir, *args.output_dir.parents]):
        raise ValueError("Require a fresh output directory under the verified bind, without symlink parents")
    cache = args.cache_dir / run_id
    if cache.exists() or any(p.is_symlink() for p in [cache, *cache.parents]):
        raise ValueError("Require a fresh scoped JIT-cache root without symlink parents")
    args.output_dir.mkdir(parents=True)
    save(args.output_dir / "plan.json", plan)
    runtime = Runtime(args.output_dir, run_id, deadline, args.max_rss_gib * 1024**3)
    guard = threading.Thread(target=runtime.guard, daemon=True)
    guard.start()
    def interrupted(signum, frame):
        raise RuntimeError(f"Diagnostic wrapper received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        cache.mkdir(parents=True)
        from transformers import AutoTokenizer
        versions = {}
        for name in ("torch", "sglang", "transformers", "triton", "flashinfer-python"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        save(args.output_dir / "package-versions.json", versions)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=True)
        dataset_sha = file_sha(args.dataset)
        inputs = freeze_inputs(args.dataset, tokenizer, args.model_path, dataset_sha)
        if file_sha(args.dataset) != dataset_sha:
            raise ValueError("Dataset changed while freezing inputs")
        save(args.output_dir / "frozen-token-input.json", inputs)
        results = [run_mode(args, runtime, mode, inputs) for mode in plan["order"]]
        save(args.output_dir / "terminal.json", {"at": utc(), "status": "COMPLETED", "results": results})
    except BaseException as error:
        save(args.output_dir / "terminal.json", {"at": utc(), "status": "FAILED", "error": repr(error)})
        raise
    finally:
        runtime.stop()
        runtime.done.set()
        guard.join(timeout=2)


if __name__ == "__main__":
    main()
