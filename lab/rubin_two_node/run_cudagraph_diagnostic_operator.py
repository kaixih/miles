#!/usr/bin/env python3
"""Plan-only host operator for a fresh diagnostic after exact main SUCCEEDED50/200.

Run on dl3 as UID28644:GID30. --execute is explicit authorization for the printed
scoped workflow, never a bypass of main completion/retention or original lease.
The host guard survives SSH loss and only stops the newly created diagnostic
container. Models, main outputs and diagnostic artifacts are never deleted.
"""
import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import run_cudagraph_main as main_driver
import sglang_graph_capture as capture
import summarize_qwen3_runs as metrics
from watch_qwen3_run import JobsAPI, is_target

SOURCES = ("run_sglang_graph_diagnostic.py", "sglang_graph_capture.py")


def utc(value=None):
    return dt.datetime.fromtimestamp(time.time() if value is None else value, dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(argv, timeout=60):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def remote(c, argv, timeout=60):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", c["node"], shlex.join(argv)], timeout)


def node(c, action, **values):
    code = inspect.getsource(node_action) + "\nnode_action(" + repr({"action": action, "c": c, **values}) + ")"
    return json.loads(remote(c, ["python3", "-B", "-c", code], 120))


def allocation(c, raw, now):
    fields = dict(item.split("=", 1) for item in raw.split() if "=" in item)
    if (fields.get("JobId") != str(c["job_id"]) or fields.get("JobState") != "RUNNING"
            or fields.get("NodeList") != c["node"] or fields.get("NumNodes") != "1"
            or not fields.get("UserId", "").endswith("(28644)")):
        raise ValueError("Allocation identity/state/node/UID mismatch")
    end = dt.datetime.fromisoformat(fields["EndTime"])
    end = end.replace(tzinfo=dt.timezone.utc) if end.tzinfo is None else end
    if abs(end.timestamp() - c["lease_timestamp"]) > 1 or end.timestamp() - now <= 2400:
        raise ValueError("Original allocation lease changed or ≤40 minutes remain")
    return {"checked_at": utc(now), "raw": raw, "original_lease": utc(end.timestamp())}


def container_identity(c, info, image_id, diagnostic=False, require_running=True):
    name = c["diagnostic_name"] if diagnostic else c["container"]
    if (info.get("Name") != "/" + name or info.get("Image") != image_id
            or info.get("Config", {}).get("Image") != c["image"]
            or info["Config"].get("User") != "28644:30"
            or (require_running and not info.get("State", {}).get("Running"))):
        raise ValueError("Container name/image/normal-UID/running identity mismatch")
    expected = [("/opt/diagnostic", c["node_base"] + "/source", False),
                ("/models/Qwen3-30B-A3B", c["node_models"] + "/Qwen3-30B-A3B", False),
                ("/inputs/train.jsonl", c["node_run_dir"] + "/inputs/train.jsonl", False),
                ("/run-output", c["node_base"] + "/run", True), ("/cache", c["node_base"] + "/cache", True)] if diagnostic else [
                ("/opt/miles", c["node_repo"], False), (c["models"], c["node_models"], False),
                ("/run-output", c["node_run_dir"], True)]
    mounts = {m["Destination"]: m for m in info.get("Mounts", [])}
    for dest, source, writable in expected:
        m = mounts.get(dest, {})
        if m.get("Source") != source or m.get("RW") is not writable or m.get("Type") != "bind":
            raise ValueError("Unexpected mount: " + dest)
    if diagnostic and info["Config"].get("Labels", {}).get("miles.graph_diagnostic") != c["diagnostic_run"]:
        raise ValueError("Diagnostic label mismatch")
    return {k: info[k] for k in ("Id", "Name", "Image", "State", "Mounts")}


def completion(c, state, jobs, driver, train, summary=None):
    if (state.get("run_id") != c["run_id"] or state.get("state") != "finished"
            or state.get("terminal_status") != "SUCCEEDED" or state.get("stop_request_count") != 0
            or state.get("expected_rollouts") != 50
            or state.get("soft_deadline_at") != utc(c["soft_timestamp"])
            or state.get("deadline_at") != utc(c["hard_timestamp"])
            or state.get("lease_deadline_at") != utc(c["lease_timestamp"])):
        raise ValueError("Watchdog run/completion/original-budget mismatch")
    matches = [j for j in jobs if j.get("submission_id") == state.get("submission_id")]
    if (len(matches) != 1 or matches[0].get("status") != "SUCCEEDED" or not is_target(matches[0], c["run_id"])
            or any(j.get("status") not in {"SUCCEEDED", "FAILED", "STOPPED"} for j in jobs)):
        raise ValueError("Exact locked main Ray submission is not SUCCEEDED or another job is active")
    if driver.get("exit_code") != 0 or driver.get("run_id") != c["run_id"] or train.get("exit_code") != 0:
        raise ValueError("Main driver/train exits must both be zero for this run")
    if summary is not None and (summary.get("partial") or summary.get("completed_training_rollouts") != list(range(50))
            or sorted({s["logged_id"] for row in summary["rows"] for s in row["train_steps"]}) != list(range(200))):
        raise ValueError("Retained log does not prove exactly 50 completed rounds and 200 updates")
    return matches[0]


def retain_login_main(c, main_plan, destination):
    """Freeze closed login-host artifacts; node-local outputs are a separate tree."""
    login = Path(c["run_dir"])
    names = ["train_exit.json", "train-driver-exit.json", "train-launch.json", "cudagraph-main-plan.json", "logs/qwen3_train.log"]
    names += [name for name in ("logs/train-driver.log", "bootstrap.json") if (login / name).is_file()]
    selected = [(name, login / name, None) for name in names]
    sources = main_plan.get("source_sha256", {})
    if not sources:
        raise ValueError("Original main plan lacks frozen source hashes")
    for name, expected in sources.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe original source path")
        selected.append(("source/" + name, Path(c["repo"]) / name, expected))
    if c.get("source_manifest"):
        selected.append(("source-manifest.json", Path(c["source_manifest"]), main_plan["source_manifest_sha256"]))
    if main_plan.get("driver_path"):
        selected.append(("driver-source.py", Path(main_plan["driver_path"]), main_plan["driver_sha256"]))
    destination.mkdir()
    files, total = {}, 0
    for name, source, expected in selected:
        if source.resolve() != source or source.stat().st_uid != 28644 or not source.is_file():
            raise ValueError("Login artifact/source path or owner mismatch: " + str(source))
        before = source.stat(); total += before.st_size
        limit = (512 if name.startswith("logs/") else 128) * 1024**2
        if before.st_size > limit or total > 1024**3:
            raise ValueError("Login evidence exceeds the small-artifact bound")
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            left = before.st_size
            while left:
                block = src.read(min(1024**2, left))
                if not block: raise ValueError("Login artifact shrank: " + str(source))
                dst.write(block); left -= len(block)
        digest = sha(target); after = source.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or target.stat().st_size != before.st_size or sha(source) != digest
                or (expected is not None and digest != expected)):
            raise ValueError("Closed login artifact or frozen source changed: " + str(source))
        files[name] = {"source": str(source), "bytes": before.st_size, "sha256": digest,
                       "verification": "stable_source_and_destination_sha256", "uid": 28644}
    verify_copy(destination, files)
    return files


def verify_login_sources(files):
    for item in files.values():
        source = Path(item["source"])
        if (source.resolve() != source or source.stat().st_uid != 28644
                or source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]):
            raise ValueError("Login artifact/source changed after retention: " + str(source))


def node_action(v):
    import hashlib, json, os, pathlib, shutil, subprocess, time
    c, action = v["c"], v["action"]
    base, source = pathlib.Path(c["node_base"]), pathlib.Path(c["node_run_dir"])
    assert (os.getuid(), os.getgid()) == (28644, 30)
    def safe(p):
        assert not any(x.is_symlink() for x in [p, *p.parents]), str(p)
        assert p.stat().st_uid == 28644, str(p)
    def digest(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024**2), b""): h.update(b)
        return h.hexdigest()
    safe(source)
    if action == "read":
        tracker = source / "checkpoints/latest_checkpointed_iteration.txt"
        assert tracker.read_text().strip() == "49", "Final checkpoint49 tracker missing"
        cp = source / "checkpoints/iter_0000049"
        shards = {p.name: p.stat().st_size for p in cp.glob("*.distcp")}
        assert shards and all(shards.values()) and (cp / ".metadata").is_file()
        assert (source / "checkpoints/rollout/global_dataset_state_dict_49.pt").stat().st_size > 0
        print(json.dumps({"watchdog": json.loads((source / "watchdog.json").read_text()), "checkpoint49_shard_sizes": shards}))
    elif action == "prepare":
        assert not base.exists()
        safe(base.parent)
        base.mkdir(mode=0o700)
        for name in ["source", "run", "cache"]: (base / name).mkdir(mode=0o700)
        probe = base / "run/.host-writer-probe"
        probe.write_text(c["diagnostic_run"]); safe(probe)
        assert probe.read_text() == c["diagnostic_run"]
        print(json.dumps({"base": str(base), "uid": os.getuid(), "host_probe": True}))
    elif action == "snapshot":
        out = base / v["phase"]; out.mkdir()
        result, total = {}, 0
        for parent, dirs, files in os.walk(source, followlinks=False):
            rel = pathlib.Path(parent).relative_to(source)
            dirs[:] = [d for d in dirs if (len(rel.parts) == 0 and d in {"logs", "telemetry", "recovery", "checkpoints"})
                       or (rel.parts and rel.parts[0] in {"logs", "telemetry", "recovery"})
                       or (str(rel) == "checkpoints" and d in {"iter_0000049", "rollout"})]
            for name in files:
                p = pathlib.Path(parent, name); r = p.relative_to(source)
                if "checkpoints" in r.parts:
                    if name not in {"latest_checkpointed_iteration.txt", ".metadata", "metadata.json", "common.pt", "global_dataset_state_dict_49.pt"}: continue
                elif p.suffix not in {".json", ".jsonl", ".log", ".csv", ".txt"}: continue
                if v["phase"] == "main-final" and not (r.parts[0] == "telemetry" or "telemetry" in name or name == "watchdog.json"): continue
                safe(p); before = p.stat(); total += before.st_size
                assert before.st_size <= 128*1024**2 and total <= 512*1024**2, "Small evidence bound exceeded"
                target = out / r; target.parent.mkdir(parents=True, exist_ok=True)
                with p.open("rb") as f, target.open("xb") as dest:
                    left = before.st_size
                    while left:
                        chunk = f.read(min(1024**2, left)); assert chunk, "Source shrank"
                        dest.write(chunk); left -= len(chunk)
                after = p.stat(); copied = digest(target)
                with p.open("rb") as f:
                    h = hashlib.sha256(); left = before.st_size
                    while left:
                        chunk = f.read(min(1024**2, left)); assert chunk; h.update(chunk); left -= len(chunk)
                assert h.hexdigest() == copied, "Captured source prefix changed"
                append_allowed = r.parts[0] == "telemetry" or "telemetry" in name or p.suffix == ".csv"
                stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                assert stable or append_allowed, "Critical evidence changed"
                result[str(r)] = {"bytes": before.st_size, "sha256": copied,
                    "source_before_bytes": before.st_size, "source_after_bytes": after.st_size,
                    "source_stable": stable, "verification": "source_prefix_rehash"}
        print(json.dumps(result))
    elif action == "manifest":
        root = base / "run"; result = {}; total = 0
        for parent, dirs, files in os.walk(root):
            for name in dirs + files: safe(pathlib.Path(parent, name))
            for name in files:
                p = pathlib.Path(parent, name); size = p.stat().st_size; total += size
                assert total <= 1100*1024**2, "Diagnostic retention bound exceeded"
                result[str(p.relative_to(root))] = {"bytes": size, "sha256": digest(p)}
        print(json.dumps(result))
    elif action == "identity":
        for name, expected in v["hashes"].items(): assert digest(base / "source" / name) == expected
        model = pathlib.Path(c["node_models"]) / "Qwen3-30B-A3B"; safe(model)
        files = {}
        for p in sorted(model.rglob("*")):
            assert not p.is_symlink(), str(p)
            if p.is_file():
                safe(p); meta = {"bytes": p.stat().st_size}
                if p.suffix != ".safetensors":
                    assert p.stat().st_size < 64*1024**2; meta["sha256"] = digest(p)
                files[str(p.relative_to(model))] = meta
        assert files and digest(source / "inputs/train.jsonl") == c["train_sha256"]
        ident = v["identity"]; ident["initial_model_inventory"] = files
        ident["initial_model_id"] = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (base / "run/diagnostic-identity.json").write_text(json.dumps(ident, indent=2))
        print(json.dumps(ident))
    elif action == "guard":
        guard = v["guard_source"]
        (base / "source/container_deadline_guard.py").write_text(guard)
        with (base / "guard.log").open("x") as log:
            proc = subprocess.Popen(["python3", "-u", str(base / "source/container_deadline_guard.py")],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = int(pathlib.Path('/proc',str(proc.pid),'stat').read_text().rpartition(') ')[2].split()[19])
        print(json.dumps({"pid": proc.pid, "start_ticks": ticks, "deadline": v["deadline"],
                          "source_sha256": hashlib.sha256(guard.encode()).hexdigest()}))
    else: raise ValueError(action)


def verify_copy(destination, manifest):
    for name, item in manifest.items():
        p = destination / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or p.resolve() != p
                or p.stat().st_uid != 28644 or p.stat().st_size != item["bytes"] or sha(p) != item["sha256"]):
            raise ValueError("Retained file identity/hash mismatch: " + name)


def retain(c, child, manifest, destination, timeout=300):
    destination.mkdir()
    subprocess.run(["rsync", "-rt", "--no-perms", "--omit-dir-times", "--partial", "--protect-args",
                    c["node"] + ":" + c["node_base"] + "/" + child + "/", str(destination) + "/"],
                   check=True, timeout=timeout)
    verify_copy(destination, manifest)


def guard_source(c, container_id, deadline):
    return """import json,subprocess,time,pathlib
c=CONFIG; expected=IDENTITY; deadline=DEADLINE
path=pathlib.Path(c['node_base'])/'guard-state.json'
path.write_text(json.dumps({'state':'armed','container_id':expected,'deadline':deadline}))
while time.time()<deadline: time.sleep(max(.001,min(1,deadline-time.time())))
d=json.loads(subprocess.check_output(['docker','inspect',expected],text=True,timeout=10))[0]
assert d['Id']==expected and d['Name']=='/'+c['diagnostic_name'] and d['Config']['Image']==c['image'] and d['Image']==c['image_id']
assert d['Config']['Labels']['miles.graph_diagnostic']==c['diagnostic_run'] and d['Config']['User']=='28644:30'
if d['State']['Running']: subprocess.run(['docker','stop','--time','10',expected],check=True,timeout=25)
path.write_text(json.dumps({'state':'deadline_container_stopped','container_id':expected,'deadline':deadline,'at':time.time()}))
""".replace("CONFIG", repr(c)).replace("IDENTITY", repr(container_id)).replace("DEADLINE", repr(deadline))


def docker_command(c):
    args = ["docker", "run", "--detach", "--name", c["diagnostic_name"], "--label", "miles.graph_diagnostic=" + c["diagnostic_run"],
            "--gpus", "device=0", "--network", "host", "--ipc", "host", "--privileged", "--user", "28644:30",
            "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--ulimit", "nofile=65535:65535",
            "--workdir", "/opt/diagnostic"]
    for source, target, readonly in [(c["node_base"] + "/source", "/opt/diagnostic", True),
        (c["node_models"] + "/Qwen3-30B-A3B", "/models/Qwen3-30B-A3B", True),
        (c["node_run_dir"] + "/inputs/train.jsonl", "/inputs/train.jsonl", True),
        (c["node_base"] + "/run", "/run-output", False), (c["node_base"] + "/cache", "/cache", False)]:
        args += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in {"MILES_PROFILE_RUN_ID": c["diagnostic_run"], "CUDA_VISIBLE_DEVICES": "0", "HOME": "/cache",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "SGLANG_PROFILE_V2": "0", "PYTHONUNBUFFERED": "1",
        "NCCL_CUMEM_ENABLE": "1", "NCCL_NVLS_ENABLE": "0", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONPYCACHEPREFIX": "/cache/pycache", "XDG_CACHE_HOME": "/cache/xdg", "CUDA_CACHE_PATH": "/cache/cuda"}.items():
        args += ["--env", key + "=" + value]
    return args + [c["image"], "sleep", "infinity"]


def execute(c, order, port):
    if (os.getuid(), os.getgid()) != (28644, 30): raise ValueError("Run on dl3 as UID28644:GID30")
    root = Path(c["durable"])
    if (root.exists() or any(p.is_symlink() for p in [root,*root.parents])
            or Path(c["run_dir"]).stat().st_uid != 28644): raise ValueError("Fresh owned diagnostic root required")
    def alloc(): return allocation(c, run(["env", "TZ=UTC", "scontrol", "show", "job", "-o", str(c["job_id"])]), time.time())
    def inspected(name): return json.loads(remote(c, ["docker", "inspect", name]))[0]
    api = JobsAPI(c["dashboard"])
    driver = json.loads(Path(c["run_dir"], "train-driver-exit.json").read_text())
    train = json.loads(Path(c["run_dir"], "train_exit.json").read_text())
    first_allocation = alloc(); actual = node(c, "read"); jobs = api.list_jobs()
    main_job = completion(c, actual["watchdog"], jobs, driver, train)
    main_info = inspected(c["container"])
    image_id = json.loads(remote(c, ["docker", "image", "inspect", c["image"]]))[0]["Id"]
    c["image_id"] = image_id
    selected = container_identity(c, main_info, image_id)
    main_plan = json.loads(Path(c["run_dir"], "cudagraph-main-plan.json").read_text())
    if (main_plan.get("run_id") != c["run_id"] or main_plan.get("git_commit") != c["source_commit"]
            or main_plan.get("preflight", {}).get("container_id") != main_info["Id"]
            or main_plan.get("preflight", {}).get("image_id") != image_id):
        raise ValueError("Original main plan/source/container identity mismatch")
    root.mkdir(parents=True)
    probe = root / ".writer-probe"; probe.write_text(c["diagnostic_run"])
    assert probe.stat().st_uid == 28644 and probe.read_text() == c["diagnostic_run"]; probe.unlink()
    write(root / "operator-plan.json", {"config": c, "order": order, "allocation": first_allocation, "main_container": selected, "ray_job": main_job})
    write(root / "main-driver-evidence.json", {"driver_exit": driver, "original_main_plan": main_plan,
        "train_exit": train, "source_sha256": {name: sha(Path(c["run_dir"], name)) for name in
        ("train_exit.json", "train-driver-exit.json", "cudagraph-main-plan.json", "train-launch.json")}})
    login_files = retain_login_main(c, main_plan, root / "main-login")
    for name, expected in [("train_exit.json", train), ("train-driver-exit.json", driver), ("cudagraph-main-plan.json", main_plan)]:
        if json.loads((root / "main-login" / name).read_text()) != expected:
            raise ValueError("Login identity record changed during initial checks: " + name)
    summary = metrics.summarize_run(c["platform"], root / "main-login/logs/qwen3_train.log", {"status": "SUCCEEDED"})
    completion(c, actual["watchdog"], jobs, driver, train, summary)
    write(root / "main-login-retention.json", {"at": utc(), "files": login_files,
        "stable_source_and_destination_sha256_verified": True})
    node(c, "prepare")
    manifest = node(c, "snapshot", phase="main-before")
    retain(c, "main-before", manifest, root / "main-before")
    write(root / "main-retention.json", {"at": utc(), "files": manifest, "source_prefix_and_destination_sha256_verified": True,
                                        "login_files": login_files,
                                        "completed_rollouts": summary["completed_training_rollouts"], "optimizer_updates": 200})
    # Recheck exact identities immediately before the only main-container mutation.
    alloc(); repeated = node(c, "read")
    verify_login_sources(login_files)
    completion(c, repeated["watchdog"], api.list_jobs(),
        json.loads(Path(c["run_dir"], "train-driver-exit.json").read_text()),
        json.loads(Path(c["run_dir"], "train_exit.json").read_text()), summary)
    fresh = inspected(c["container"]); container_identity(c, fresh, image_id)
    if fresh["Id"] != main_info["Id"]: raise ValueError("Main container replaced")
    remote(c, ["docker", "stop", "--time", "15", fresh["Id"]], 40)
    write(root / "main-stopped.json", {"at": utc(), "container_id": fresh["Id"], "retention_verified_before_stop": True})
    final = node(c, "snapshot", phase="main-final"); retain(c, "main-final", final, root / "main-final")
    write(root / "main-final-retention.json", final)
    hashes = {name: sha(Path(__file__).with_name(name)) for name in SOURCES}
    for name in SOURCES:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(Path(__file__).with_name(name)), c["node"] + ":" + c["node_base"] + "/source/" + name], check=True, timeout=30)
    alloc()
    diagnostic_id = remote(c, docker_command(c)).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", diagnostic_id): raise ValueError("Docker did not return an exact new container ID")
    try:
        info = inspected(diagnostic_id); container_identity(c, info, image_id, True)
        deadline = min(time.time() + 1800, c["lease_timestamp"] - 120)
        identity = {"run_id": c["diagnostic_run"], "container_name": c["diagnostic_name"], "container_id": info["Id"],
            "image_reference": c["image"], "image_id": image_id, "uid": 28644, "gid": 30,
            "labels": {"miles.graph_diagnostic": c["diagnostic_run"]}, "docker_inspect_verified_at": utc(),
            "main_terminal_verified": True, "source_commits": {"miles_main": c["source_commit"], "sglang": capture.SGLANG_COMMIT},
            "diagnostic_source_sha256": hashes, "main_submission_id": main_job["submission_id"]}
        identity = node(c, "identity", hashes=hashes, identity=identity); write(root / "diagnostic-identity.json", identity)
        probe_code = "from pathlib import Path;import os;assert(os.getuid(),os.getgid())==(28644,30);p=Path('/run-output/.host-writer-probe');assert p.read_text()==" + repr(c["diagnostic_run"]) + ";p.unlink();q=Path('/run-output/.container-writer-probe');q.write_text('container28644');assert q.stat().st_uid==28644"
        remote(c, ["docker", "exec", diagnostic_id, "python3", "-c", probe_code])
        remote(c, ["python3", "-c", "from pathlib import Path;p=Path(" + repr(c["node_base"] + "/run/.container-writer-probe") + ");assert p.read_text()=='container28644' and p.stat().st_uid==28644;p.unlink()"])
        guard = node(c, "guard", guard_source=guard_source(c, diagnostic_id, deadline), deadline=deadline)
        for _ in range(20):
            try:
                armed = json.loads(remote(c, ["cat", c["node_base"] + "/guard-state.json"], 10)); break
            except subprocess.CalledProcessError: time.sleep(.2)
        else: raise RuntimeError("Host deadline guard did not arm; no engine launched")
        if armed != {"state": "armed", "container_id": diagnostic_id, "deadline": deadline}: raise ValueError("Guard identity mismatch")
        guard_check = "import os;from pathlib import Path;p=" + repr(guard["pid"]) + ";os.kill(p,0);assert int(Path('/proc',str(p),'stat').read_text().rpartition(') ')[2].split()[19])==" + repr(guard["start_ticks"])
        remote(c, ["python3", "-c", guard_check])
        write(root / "guard-armed.json", {**guard, **armed})
        argv = ["docker", "exec", diagnostic_id, "python3", "-u", "/opt/diagnostic/run_sglang_graph_diagnostic.py",
            "--identity-file", "/run-output/diagnostic-identity.json", "--model-path", "/models/Qwen3-30B-A3B",
            "--dataset", "/inputs/train.jsonl", "--output-dir", "/run-output/diagnostic", "--host", c["node_ip"],
            "--port", str(port), "--order", order, "--deadline-utc", utc(deadline), "--execute"]
        with (root / "wrapper-console.log").open("x") as log:
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", c["node"], shlex.join(argv)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time() + 10))
            except BaseException as error:
                write(root / "wrapper-exit.json", {"at": utc(), "exit_code": None, "error": repr(error),
                    "original_deadline": deadline, "host_guard_remains_armed": True})
                raise
        write(root / "wrapper-exit.json", {"at": utc(), "exit_code": result.returncode})
    finally:
        before = {}
        try:
            before = node(c, "manifest")
            write(root / "diagnostic-source-before.json", {"at": utc(), "files": before})
            retain(c, "run", before, root / "node-output", timeout=min(300, max(1, c["lease_timestamp"] - time.time() - 45)))
            after = node(c, "manifest")
            if before != after: raise ValueError("Diagnostic output changed during retention")
            evidence = []
            for name in before:
                if name.endswith(".trace.json.gz"):
                    try: evidence.append(capture.inspect_trace(root / "node-output" / name))
                    except (ValueError, OSError, EOFError) as error: evidence.append({"path": name, "valid_complete_trace": False, "error": repr(error)})
            write(root / "diagnostic-retention.json", {"at": utc(), "files": before, "source_before_after_and_destination_verified": True, "traces": evidence})
        except BaseException as error:
            destination = root / "node-output"
            partial = {str(p.relative_to(destination)): p.stat().st_size for p in destination.rglob("*") if p.is_file() and not p.is_symlink()} if destination.exists() else {}
            write(root / "diagnostic-retention-failure.json", {"at": utc(), "state": "FAILED", "error": repr(error),
                "source_manifest_before": before, "partial_destination_file_sizes": partial,
                "interpretation": "No successful retention claimed. Preserve partial durable files and unchanged node-local output; source may have changed before the exact diagnostic stop."})
            raise
        finally:
            final_info = inspected(diagnostic_id)
            container_identity(c, final_info, image_id, True, require_running=False)
            if final_info["Id"] != diagnostic_id: raise ValueError("Diagnostic container replaced")
            if final_info["State"]["Running"]:
                remote(c, ["docker", "stop", "--time", "10", diagnostic_id], 30)
            write(root / "diagnostic-stopped.json", {"at": utc(), "container_id": diagnostic_id})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--run-id", required=True)
    p.add_argument("--order", required=True, choices=("off,on", "on,off"))
    p.add_argument("--port", type=int, default=31081); p.add_argument("--execute", action="store_true")
    a = p.parse_args(); c = main_driver._config(json.loads(a.config.read_text()))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{5,79}", a.run_id) or not 1024 <= a.port <= 65535:
        raise ValueError("Explicit unique diagnostic run ID and port required")
    c.update(diagnostic_run=a.run_id, diagnostic_name="miles-graph-diagnostic-" + a.run_id,
             node_base=str(Path(c["node_run_dir"]).parent / ("graph-diagnostic-" + a.run_id)),
             durable=str(Path(c["run_dir"]) / "diagnostics" / a.run_id))
    print(json.dumps({"mode": "EXECUTE" if a.execute else "PLAN_ONLY_NO_REMOTE_CALLS", "config": c,
        "order": a.order, "port": a.port, "docker_argv": docker_command(c),
        "required_gate": "Exact SUCCEEDED main, zero exits, immutable original lease >40min,50/200 logs; verified small evidence retained before exact main stop",
        "diagnostic_deadline": "min(actual launch+1800, original lease-120); independent exact-container host guard"}, indent=2), flush=True)
    if a.execute: execute(c, a.order, a.port)


if __name__ == "__main__": main()
