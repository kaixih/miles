"""Collect this experiment's read-only evidence and rebuild its metrics JSON."""

import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import zlib

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / "outputs/rubin-gb300-qwen3"
RUNS = {
    "rubin": {
        "root": "/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs",
        "destination": "rubin-a2",
        "display_name": "Rubin ES A2",
        "node": "vr-nvl72-ts2-l11-038-c15",
        "container": "miles-rubin-qwen3-j2198331-a2-0",
        "watchdog_host_path": "/tmp/miles-rubin-j2198331/run-a2-mb4096/watchdog.json",
        "hardware": {"name": "NVIDIA VR NVL72 ES", "memory_mib_per_gpu": 168896,
                     "compute_capability": "10.7", "engineering_sample": True},
    },
    "gb300": {
        "root": "/home/scratch.kaixih_ent/repro/miles-gb300-qwen3-gsm8k/20260915-j2198810-nfs",
        "destination": "gb300",
        "display_name": "GB300",
        "node": "gb300-nvl-012-compute04",
        "container": "miles-gb300-qwen3-j2198810-0",
        "watchdog_host_path": "/tmp/miles-gb300-j2198810/run/watchdog.json",
        "hardware": {"name": "NVIDIA GB300", "memory_mib_per_gpu": 284208,
                     "compute_capability": "10.3", "engineering_sample": False},
    },
}


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAVE_EVENT = re.compile(
    r"(?P<kind>saving checkpoint at|successfully saved checkpoint from) iteration\s+"
    r"(?P<iteration>\d+) to (?P<path>\S+)"
)


def scalar_flag(argv, name, *, default=None, integer=False):
    """Read one actual scalar option; reject duplicates or malformed values."""
    values = []
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            values.append(argv[index + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Ambiguous duplicate {name}")
    value = values[0] if values else default
    return int(value) if integer and value is not None else value


def actual_recipe_and_save_events(argv, log):
    max_tokens = scalar_flag(argv, "--max-tokens-per-gpu", integer=True)
    if max_tokens is None or max_tokens <= 0:
        raise ValueError("Actual Ray entrypoint must specify a positive max-tokens-per-gpu")
    logprob_tokens = scalar_flag(argv, "--log-probs-max-tokens-per-gpu", default=max_tokens, integer=True)
    interval = scalar_flag(argv, "--save-interval", integer=True)
    retain_interval = scalar_flag(argv, "--save-retain-interval", integer=True)
    save_path = scalar_flag(argv, "--save")
    total = scalar_flag(argv, "--num-rollout", default=50, integer=True)
    events, seen = [], set()
    for number, raw in enumerate(log.split("\n"), 1):
        line = ANSI.sub("", raw)
        match = SAVE_EVENT.search(line)
        if match is None or match["path"] != save_path:
            continue
        iteration = int(match["iteration"])
        if not 0 <= iteration < total:
            raise ValueError(f"Checkpoint iteration outside the recorded run: {iteration}")
        key = (iteration, match["kind"])
        if key in seen:
            continue
        seen.add(key)
        events.append({"rollout_id": iteration, "event": match["kind"], "line": number,
                       "excerpt": line[-1200:]})
    saved_ids = sorted({e["rollout_id"] for e in events})
    # The driver saves after actor.train. Exclude both that rollout and the next
    # one conservatively, because a later train_wait timer can include the save.
    exclusions = sorted({0, *(i for saved in saved_ids for i in (saved, saved + 1) if i < total)})
    recipe = {"max_training_tokens_per_gpu": max_tokens, "max_logprob_tokens_per_gpu": logprob_tokens,
              "checkpoint_save_interval": interval, "checkpoint_retain_interval": retain_interval,
              "checkpoint_path": save_path}
    evidence = {"save_interval_from_actual_argv": interval, "save_retain_interval_from_actual_argv": retain_interval,
                "save_path_from_actual_argv": save_path, "async_save_requested": "--async-save" in argv,
                "observed_save_rollout_ids": saved_ids, "observed_save_events": events,
                "timing_excluded_rollout_ids": exclusions,
                "timing_policy": "Exclude warmup0 and each observed checkpoint-save rollout plus its following rollout; conservative across nested timers.",
                "schedule_note": "Miles also saves on runtime epoch boundaries and final rollout; do not infer all saves from interval alone.",
                "detection_scope": "Exact pinned Megatron start/success log markers for the actual --save path; not a checkpoint-completeness validation."}
    return recipe, evidence, exclusions


LOG = "logs/qwen3_train.log"
FILES = ["planned-run-config.json", "train-launch.json", "train_exit.json",
         "train-driver-exit.json", "bootstrap.json", "gpu-telemetry-initial.csv",
         "preflight/runtime-provenance.json", "recovery/router-nofile-20260916.json",
         "recovery/router-workers-20260916.json", "recovery/four-engine-recovery-20260916.json",
         "recovery/four-engine-recovery-log.txt"]
MAX_LOG_BYTES = 1024**3
MAX_AUX_BYTES = 8 * 1024**2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def watchdog_source(config):
    """Bind the proven host file to its exact main node/container identity."""
    known = {
        ("vr-nvl72-ts2-l11-038-c15", "miles-rubin-qwen3-j2198331-a2-0"):
            "/tmp/miles-rubin-j2198331/run-a2-mb4096/watchdog.json",
        ("gb300-nvl-012-compute04", "miles-gb300-qwen3-j2198810-0"):
            "/tmp/miles-gb300-j2198810/run/watchdog.json",
    }
    expected = known.get((config.get("node"), config.get("container")))
    if expected is None or config.get("watchdog_host_path") != expected:
        raise ValueError("Unverified watchdog host path or main identity")
    return {"node": config["node"], "path": expected, "read_method": "host_ssh",
            "main_container": config["container"], "container_path": "/run-output/watchdog.json"}


def _remote_payload(config, prefix, files):
    """Executed read-only on dl3. Hash the prefix; transfer only a verified append."""
    import base64, datetime, hashlib, json, os, pathlib, shlex, signal, subprocess, time, zlib
    signal.alarm(80)
    started = time.monotonic()
    watchdog_provenance = watchdog_source(config)
    root = pathlib.Path(config["root"])
    limit = 1024**3
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    def packet(raw):
        return {"bytes": len(raw), "sha256": digest(raw),
                "data": base64.b64encode(zlib.compress(raw, 3)).decode()}
    path = root / "logs/qwen3_train.log"
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        size = before.st_size
        if not 0 < size <= limit:
            raise ValueError("Raw log exceeds bounded snapshot size")
        offset = prefix["bytes"]
        if not 0 <= offset <= limit:
            raise ValueError("Invalid prefix offset")
        full = hashlib.sha256()
        consumed = 0
        if offset <= size:
            while consumed < offset:
                block = stream.read(min(1024**2, offset - consumed))
                if not block:
                    raise ValueError("Log truncated during prefix verification")
                full.update(block)
                consumed += len(block)
        append = offset <= size and full.hexdigest() == prefix["sha256"]
        if not append:
            stream.seek(0)
            full, offset = hashlib.sha256(), 0
        tail = bytearray()
        while offset + len(tail) < size:
            block = stream.read(min(1024**2, size - offset - len(tail)))
            if not block:
                raise ValueError("Log truncated during snapshot")
            full.update(block)
            tail.extend(block)
        after = os.fstat(stream.fileno())
        current = path.stat()
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino) or after.st_size < size:
            raise ValueError("Log rotated or truncated during snapshot; retry")
    log = {"mode": "append" if append else "full", "offset": offset,
           "prefix_sha256": prefix["sha256"] if append else digest(b""),
           "bytes": size, "sha256": full.hexdigest(), "tail": packet(tail),
           "source_inode": before.st_ino, "source_mtime_ns": before.st_mtime_ns,
           "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    records = {}
    for name in files:
        source = root / name
        if source.is_file():
            if source.stat().st_size > 8 * 1024**2:
                raise ValueError("Auxiliary file exceeds bounded size: " + name)
            records[name] = packet(source.read_bytes())
    # The bind-mounted file persists after the completed main container stops.
    # This read never invokes Docker or depends on the main container lifecycle.
    command = ["cat", "--", watchdog_provenance["path"]]
    watchdog_started = time.monotonic()
    watchdog_raw = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", config["node"], shlex.join(command)],
        timeout=30)
    if len(watchdog_raw) > 8 * 1024**2 or not isinstance(json.loads(watchdog_raw), dict):
        raise ValueError("Invalid or oversized host watchdog record")
    watchdog_provenance.update(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               bytes=len(watchdog_raw), sha256=digest(watchdog_raw))
    return {"schema": "verified-log-append-v1", "root": config["root"], "log": log,
            "files": records, "watchdog": packet(watchdog_raw), "watchdog_source": watchdog_provenance,
            "watchdog_read_seconds": time.monotonic() - watchdog_started,
            "remote_seconds": time.monotonic() - started}


def decode_packet(packet, maximum):
    size = packet["bytes"]
    if type(size) is not int or not 0 <= size <= maximum:
        raise ValueError("Invalid packet size")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(packet["data"], validate=True), size + 1)
    if len(raw) != size or not decoder.eof or decoder.unused_data or sha(raw) != packet["sha256"]:
        raise ValueError("Packet size or SHA256 mismatch")
    return raw


def decode_payload(data, config, previous):
    if data.get("schema") != "verified-log-append-v1" or data.get("root") != config["root"]:
        raise ValueError("Wrong source or protocol")
    log = data["log"]
    if type(log["bytes"]) is not int or not 0 < log["bytes"] <= MAX_LOG_BYTES:
        raise ValueError("Invalid raw log size")
    if log["mode"] == "append":
        if log["offset"] != len(previous) or log["prefix_sha256"] != sha(previous):
            raise ValueError("Local prefix changed or source prefix mismatched")
        base = previous
    elif log["mode"] == "full" and log["offset"] == 0 and log["prefix_sha256"] == sha(b""):
        base = b""
    else:
        raise ValueError("Invalid append/full mode")
    raw = base + decode_packet(log["tail"], MAX_LOG_BYTES)
    if len(raw) != log["bytes"] or sha(raw) != log["sha256"]:
        raise ValueError("Reconstructed raw log size or SHA256 mismatch")
    if not set(data["files"]).issubset(FILES):
        raise ValueError("Unexpected remote file name")
    result = {name: decode_packet(value, MAX_AUX_BYTES) for name, value in data["files"].items()}
    if not {"planned-run-config.json", "train-launch.json"}.issubset(result):
        raise ValueError("Required launch evidence missing")
    expected_watchdog_source = watchdog_source(config)
    provenance = data.get("watchdog_source", {})
    if any(provenance.get(key) != value for key, value in expected_watchdog_source.items()):
        raise ValueError("Watchdog host provenance differs from the verified source")
    watchdog_raw = decode_packet(data["watchdog"], MAX_AUX_BYTES)
    if (not isinstance(json.loads(watchdog_raw), dict)
            or provenance.get("sha256") != sha(watchdog_raw)
            or provenance.get("bytes") != len(watchdog_raw)):
        raise ValueError("Invalid watchdog record")
    result[LOG] = raw
    result["watchdog.json"] = watchdog_raw
    receipt = {k: v for k, v in log.items() if k != "tail"}
    receipt.update(tail_bytes=log["tail"]["bytes"], tail_sha256=log["tail"]["sha256"],
                   compressed_base64_bytes=len(log["tail"]["data"]),
                   remote_seconds=data["remote_seconds"], watchdog_read_seconds=data["watchdog_read_seconds"],
                   source_root=config["root"], watchdog_source=provenance)
    result["collection-receipt.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    return result


def collect(item):
    """Fetch/validate one platform entirely in memory. Never mutate current files."""
    label, config = item
    watchdog_source(config)  # Reject an unverified path before any SSH call.
    destination = ROOT / config["destination"]
    old_path = destination / LOG
    previous = old_path.read_bytes() if old_path.is_file() else b""
    prefix = {"bytes": len(previous), "sha256": sha(previous)}
    code = inspect.getsource(watchdog_source) + "\n" + inspect.getsource(_remote_payload) + "\nimport json\nprint(json.dumps(_remote_payload(" + \
        repr(config) + "," + repr(prefix) + "," + repr(FILES) + ")))"
    raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "dl3",
                                   shlex.join(["python3", "-c", code])], timeout=90)
    data = json.loads(raw)
    files = decode_payload(data, config, previous)
    return label, metadata_for(label, config, files), files


def metadata_for(label, config, files):
    planned = json.loads(files["planned-run-config.json"])
    launch = json.loads(files["train-launch.json"])
    log = files[LOG].decode(errors="replace")
    entries = [line for line in log.split("\n") if line.startswith("Running entrypoint for job ")]
    if len(entries) != 1:
        raise ValueError(f"Expected one actual Ray entrypoint: {label}")
    argv = shlex.split(entries[0].split(": ", 1)[1])
    if "--use-pytorch-profiler" in argv:
        raise ValueError("Main run unexpectedly enables profiler")
    watchdog = json.loads(files["watchdog.json"])
    if watchdog.get("run_id") != planned["run_id"] or not watchdog.get("submission_id"):
        raise ValueError("Watchdog and launch identities differ")
    if not entries[0].startswith("Running entrypoint for job " + watchdog["submission_id"] + ": "):
        raise ValueError("Raw log and watchdog submission identities differ")
    actual_recipe, checkpoint_evidence, timing_exclusions = actual_recipe_and_save_events(argv, log)
    return {
        "display_name": config["display_name"],
        "artifact_subdirectory": config["destination"],
        "status": watchdog.get("terminal_status", watchdog.get("ray_status", "UNKNOWN")),
        "run_id": planned["run_id"], "submission_id": watchdog.get("submission_id"),
        "gpus": 4, "expected_rollouts": 50, "optimizer_steps_per_rollout": 4,
        "profiling_coverage_known": True, "profiled_rollouts": [],
        "profiled_optimizer_steps": [], "exclude_timing_rollouts": timing_exclusions,
        "checkpoint_save_evidence": checkpoint_evidence,
        "image": planned["image"], "hardware": config["hardware"],
        "git_commit_at_launch": launch["git_commit"],
        "source_sha256_at_launch": launch["source_sha256"],
        "started_utc": launch["started_utc"], "actual_ray_entrypoint_argv": argv,
        "graph": {"rollout_cuda_graph": False, "rollout_piecewise_cuda_graph": False},
        "kernels": {"sglang_attention": "triton", "sglang_moe": "triton",
                    "sglang_bf16_gemm": "torch", "megatron_attention_policy": "flash",
                    "packed_training_attention_backend": "Pending actual trace confirmation"},
        "recipe": {"model": "Qwen3-30B-A3B (standard, not Base)",
                   "hf_revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                   "dataset_reward": "GSM8K / strict #### scorer",
                   "batch_description": "256 prompts × 8 responses; 4 updates",
                   "length_description": "1024 response / 512 prompt tokens",
                   "learning_rate": "1e-6", "sampling_description": "T=1, top-p=1",
                   "group_filter": "None", "parallelism": "TP1 / PP1 / CP1 / EP4",
                   **actual_recipe},
        "source_remote_root": config["root"],
        "operational_incidents": [json.loads(files[name])
                                  for name in ["recovery/router-nofile-20260916.json",
                                               "recovery/router-workers-20260916.json",
                                               "recovery/four-engine-recovery-20260916.json"]
                                  if name in files],
        "comparison_scope": "Unmatched memory-fit configurations: Rubin A2 uses4096 tokens/GPU; original GB300 remains8192 pending user decision. Custom Rubin and upstream GB runtime versions also differ; this is not a controlled GPU-only speed comparison.",
    }


@contextlib.contextmanager
def collection_lock(root):
    """Reject concurrent writers; the persistent lock file contains no evidence."""
    with (root / ".collect_snapshot.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def commit_files(root, stage, relative_paths, obsolete=()):
    """Short commit after all validation; restore previous artifacts on any exception.

    comparison.json is installed last, serving as the snapshot commit marker.
    A process/host crash during filesystem renames is outside rollback protection.
    """
    backup = stage / ".rollback"
    done = []
    try:
        for name in [*obsolete, *relative_paths]:
            target, incoming, saved = root / name, stage / name, backup / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Refuse non-regular destination: " + str(target))
            existed = target.exists()
            if existed:
                saved.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, saved)
            done.append((name, existed))
            if name in relative_paths:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(incoming, target)
    except BaseException:
        for name, existed in reversed(done):
            target, saved = root / name, backup / name
            if target.exists():
                target.unlink()
            if existed:
                os.replace(saved, target)
        raise


def load_io_evidence(path):
    """Freeze an explicit window input; malformed structure cannot imply no I/O."""
    raw = path.read_bytes()
    if len(raw) > MAX_AUX_BYTES:
        raise ValueError("I/O window evidence exceeds bounded size")
    document = json.loads(raw)
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or not isinstance(document.get("windows"), list)):
        raise ValueError("Invalid I/O window evidence schema")
    try:
        stamp = dt.datetime.fromisoformat(document["snapshot_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timezone missing")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid I/O evidence snapshot timestamp") from exc
    ids = set()
    for window in document["windows"]:
        if (not isinstance(window, dict) or window.get("platform") not in RUNS
                or not isinstance(window.get("id"), str) or not window["id"]
                or window["id"] in ids or not isinstance(window.get("operation"), str)
                or not window["operation"] or not isinstance(window.get("status"), str)
                or not window["status"] or not {"start_utc", "end_utc"}.issubset(window)):
            raise ValueError("Malformed or ambiguous I/O window record")
        ids.add(window["id"])
    return {"document": document, "raw": raw, "path": str(path.resolve()), "sha256": sha(raw)}


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def io_exclusions(label, preliminary_run, metadata, evidence, helper):
    """Union I/O findings; the normal parser computes final eligibility/statistics."""
    document = evidence["document"]
    assessment = helper.assess_io_overlap(preliminary_run["rows"], label, document["windows"],
        document["snapshot_utc"], log_timestamps_are_utc=True)
    existing = metadata.get("exclude_timing_rollouts", [0])
    if any(type(index) is not int or index < 0 for index in existing):
        raise ValueError("Invalid pre-existing timing exclusion")
    combined = sorted(set(existing) | set(assessment["excluded_rollout_ids"]))
    helper_path = WORKSPACE / "lab/rubin_two_node/io_overlap.py"
    return {**metadata, "exclude_timing_rollouts": combined,
            "io_overlap": {"assessment": assessment,
                "exclusions_before_io": sorted(set(existing)), "exclusions_after_union": combined,
                "window_input": {"path": evidence["path"], "sha256": evidence["sha256"],
                    "retained_path": str((ROOT / "io-overlap-windows.snapshot.json").resolve()),
                    "snapshot_utc": document["snapshot_utc"], "evidence": document},
                "helper_source": {"path": str(helper_path.resolve()), "sha256": sha(helper_path.read_bytes())},
                "log_timestamps_are_utc": True,
                "timestamp_scope": "Explicit UTC convention for these two retained Miles run logs; window evidence keeps its own observation cutoff.",
                "learning_data_policy": "No rewards, update counts, or raw metric values are removed or rewritten."}}


def build_snapshot(stage, results, io_evidence):
    """Run the same metric parser on staged evidence, with final canonical links."""
    metadata = {label: record for label, record, _files in results}
    build = json.loads((WORKSPACE / "lab/rubin_two_node/rubin-build-report.json").read_text())
    metadata["rubin"]["versions"] = {
        c["name"]: {"version": c["version"], "source_ref": c.get("source_ref")}
        for c in build["components"]}
    gb = json.loads((stage / RUNS["gb300"]["destination"] / "preflight/runtime-provenance.json").read_text())
    metadata["gb300"]["versions"] = gb["packages"]
    parser = load_source_module("snapshot_metrics", WORKSPACE / "lab/rubin_two_node/summarize_qwen3_runs.py")
    io_helper = load_source_module("snapshot_io_overlap", WORKSPACE / "lab/rubin_two_node/io_overlap.py")
    collector_source = {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).read_bytes())}
    (stage / "io-overlap-windows.snapshot.json").write_bytes(io_evidence["raw"])
    runs = []
    for label, config in RUNS.items():
        relative = Path(config["destination"]) / LOG
        preliminary = parser.summarize_run(label, stage / relative, metadata[label])
        metadata[label] = io_exclusions(label, preliminary, metadata[label], io_evidence, io_helper)
        metadata[label]["collector_source"] = collector_source
        run = parser.summarize_run(label, stage / relative, metadata[label])
        run["source_log"] = str((ROOT / relative).resolve())
        runs.append(run)
    report = parser.report(runs, parser.legacy_baseline(WORKSPACE / "lab/rubin_two_node/qwen3-gb300-baseline.json"))
    (stage / "runs.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (stage / "comparison.json").write_text(json.dumps(parser.json_safe(report), indent=2, allow_nan=False) + "\n")
    return report


def collect_snapshot():
    """No current evidence is replaced until both reads and the summary succeed."""
    with collection_lock(ROOT):
        io_evidence = load_io_evidence(ROOT / "io-overlap-windows.json")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(collect, RUNS.items()))
        if {label for label, _, _ in results} != set(RUNS):
            raise ValueError("Incomplete platform collection")
        with tempfile.TemporaryDirectory(prefix=".snapshot-stage-", dir=ROOT) as temporary:
            stage = Path(temporary)
            install, obsolete = [], []
            for label, _metadata, files in results:
                destination = RUNS[label]["destination"]
                for filename, raw in files.items():
                    relative = Path(destination) / filename
                    path = stage / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    install.append(relative)
                obsolete.extend(Path(destination) / name for name in FILES if name not in files
                                and (ROOT / destination / name).exists())
            report = build_snapshot(stage, results, io_evidence)
            install.extend([Path("io-overlap-windows.snapshot.json"), Path("runs.json"), Path("comparison.json")])
            for run in report["runs"]:
                if sha((stage / RUNS[run["label"]]["destination"] / LOG).read_bytes()) != run["source_log_sha256"]:
                    raise ValueError("Parser log hash differs from transferred evidence")
            commit_files(ROOT, stage, install, obsolete)
            return report


def main():
    report = collect_snapshot()
    print(json.dumps({"output": str(ROOT / "comparison.json"), "runs": [
        {"label": run["label"], "rows": len(run["rows"]), "partial": run["partial"]}
        for run in report["runs"]]}))


if __name__ == "__main__":
    main()
