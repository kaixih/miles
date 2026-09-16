#!/usr/bin/env python3
"""Explicit fresh diagnostic after a failed, stopped diagnostic; never restarts main.

Uses the prior host guard's exact deadline, not a new thirty-minute allowance.
Default plan mode only reads local evidence. Execute on dl3 as28644:30.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import run_cudagraph_diagnostic_operator as O

RECEIPTS = {
    "operator-plan.json", "main-driver-evidence.json", "main-login-retention.json",
    "main-retention.json", "main-stopped.json", "diagnostic-identity.json",
    "diagnostic-stopped.json", "guard-armed.json", "diagnostic-after-stop-retention.json",
}
MAIN_KEYS = ("platform", "run_id", "job_id", "node", "node_ip", "image", "container",
             "repo", "node_repo", "models", "node_models", "run_dir", "node_run_dir",
             "source_commit", "train_sha256", "eval_sha256", "lease_timestamp",
             "soft_timestamp", "hard_timestamp")


def load(path):
    return json.loads(Path(path).read_text())


def evidence(prior, manifest_path):
    manifest = load(manifest_path)
    hashes = manifest.get("files", {})
    if set(hashes) != RECEIPTS:
        raise ValueError("Evidence manifest must bind exactly the documented prior receipts")
    records = {}
    for name, digest in hashes.items():
        p = prior / name
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or p.resolve() != p
                or p.stat().st_size > 16*1024**2 or O.sha(p) != digest):
            raise ValueError("Prior receipt path/hash mismatch: " + name)
        records[name] = load(p)
    return records, hashes


def deadline_check(deadline, prior_guard, lease, now, minimum=600):
    if (abs(deadline-prior_guard.get("deadline", 0)) > 0.000001 or prior_guard.get("state") != "armed"
            or deadline > lease-120 or deadline-now < minimum):
        raise ValueError("Original diagnostic deadline changed, expired, or insufficient budget remains")


def stopped(c, info, expected_id, diagnostic=False):
    O.container_identity(c, info, c["image_id"], diagnostic, require_running=False)
    if info["Id"] != expected_id or info.get("State", {}).get("Running") is not False:
        raise ValueError("Exact prior container must already be stopped")


def inherited_order(records):
    order = records["operator-plan.json"].get("order")
    if order not in ("off,on", "on,off"):
        raise ValueError("Prior operator must record exactly off,on or on,off order")
    return order


def validate_prior(c, prior, records, deadline, now):
    old = records["operator-plan.json"]
    old_c = old["config"]
    if any(c.get(k) != old_c.get(k) for k in MAIN_KEYS):
        raise ValueError("Original main config, source, input paths, or lease differs")
    order = inherited_order(records)
    if old_c.get("diagnostic_run") == c["diagnostic_run"]:
        raise ValueError("Preserve prior order and select a fresh diagnostic ID")
    if str(prior) != old_c.get("durable"):
        raise ValueError("Prior evidence directory is not the original operator directory")
    old_id = records["diagnostic-identity.json"]
    old_stop = records["diagnostic-stopped.json"]
    guard = records["guard-armed.json"]
    if (old_id.get("run_id") != old_c.get("diagnostic_run")
            or old_id.get("container_name") != old_c.get("diagnostic_name")
            or old_id.get("container_id") != old_stop.get("container_id")
            or old_id.get("container_id") != guard.get("container_id")
            or old_id.get("image_id") != old_c.get("image_id")
            or old_id.get("image_reference") != c["image"]):
        raise ValueError("Prior diagnostic identity, stop receipt or guard differs")
    deadline_check(deadline, guard, c["lease_timestamp"], now)
    after_stop = records["diagnostic-after-stop-retention.json"]
    if (after_stop.get("container_id") != old_id["container_id"]
            or after_stop.get("stopped_before_retention") is not True
            or after_stop.get("source_before_after_and_destination_verified") is not True
            or not after_stop.get("files")):
        raise ValueError("Failed diagnostic was not stably retained after its exact stop")
    O.verify_copy(prior / "node-output-after-stop", after_stop["files"])
    if load(prior / "node-output-after-stop/diagnostic-identity.json") != old_id:
        raise ValueError("Retained prior diagnostic identity differs")
    retained = records["main-login-retention.json"]
    if retained.get("stable_source_and_destination_sha256_verified") is not True:
        raise ValueError("Prior login retention was not verified")
    O.verify_copy(prior / "main-login", retained["files"])
    O.verify_login_sources(retained["files"])
    details = records["main-driver-evidence.json"]
    original = load(prior / "main-login/cudagraph-main-plan.json")
    driver = load(prior / "main-login/train-driver-exit.json")
    train = load(prior / "main-login/train_exit.json")
    if (original != details["original_main_plan"] or driver != details["driver_exit"]
            or train != details["train_exit"] or original.get("run_id") != c["run_id"]
            or original.get("git_commit") != c["source_commit"]
            or original.get("preflight", {}).get("container_id") != old["main_container"]["Id"]
            or original["preflight"].get("image_id") != old_c["image_id"]):
        raise ValueError("Original main plan/exits/container does not match retained evidence")
    for name, digest in original.get("source_sha256", {}).items():
        item = retained["files"].get("source/" + name, {})
        if item.get("sha256") != digest:
            raise ValueError("Original main frozen source is not bound by retained evidence")
    if not original.get("source_sha256"):
        raise ValueError("Missing original main source hashes")
    main_retained = records["main-retention.json"]
    if (main_retained.get("source_prefix_and_destination_sha256_verified") is not True
            or main_retained.get("login_files") != retained["files"]
            or main_retained.get("completed_rollouts") != list(range(50))
            or main_retained.get("optimizer_updates") != 200):
        raise ValueError("Original main retention does not prove the completed recipe")
    O.verify_copy(prior / "main-before", main_retained["files"])
    main_stop = records["main-stopped.json"]
    if (main_stop.get("container_id") != old["main_container"]["Id"]
            or main_stop.get("retention_verified_before_stop") is not True):
        raise ValueError("Original main stop receipt mismatch")
    summary = O.metrics.summarize_run(c["platform"], prior / "main-login/logs/qwen3_train.log", {"status": "SUCCEEDED"})
    watchdog = load(prior / "main-before/watchdog.json")
    job = O.completion(c, watchdog, [old["ray_job"]], driver, train, summary)
    if old_id.get("main_submission_id") != job.get("submission_id"):
        raise ValueError("Prior diagnostic did not bind the same completed main submission")
    return {"old_config": old_c, "old_identity": old_id, "summary": summary,
            "watchdog": watchdog, "job": job, "driver": driver, "train": train,
            "main_id": main_stop["container_id"], "order": order}


def execute(c, prior, records, receipt_hashes, deadline, hashes, port):
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError("Execute only as dl3 normal UID28644:GID30")
    proof = validate_prior(c, prior, records, deadline, time.time())
    old_c = proof["old_config"]
    c["image_id"] = old_c["image_id"]
    root = Path(c["durable"])
    if root.exists() or root.resolve() != root or root.parent.stat().st_uid != 28644:
        raise ValueError("Fresh normal-UID recovery directory required")
    def allocated():
        return O.allocation(c, O.run(["env", "TZ=UTC", "scontrol", "show", "job", "-o", str(c["job_id"])]), time.time())
    def inspect_id(identity):
        return json.loads(O.remote(c, ["docker", "inspect", identity]))[0]
    allocation = allocated()
    stopped(c, inspect_id(proof["main_id"]), proof["main_id"])
    stopped(old_c, inspect_id(proof["old_identity"]["container_id"]), proof["old_identity"]["container_id"], True)
    current = O.node(c, "read")
    O.completion(c, current["watchdog"], [proof["job"]], proof["driver"], proof["train"], proof["summary"])
    if current["watchdog"] != proof["watchdog"]:
        raise ValueError("Stopped main watchdog changed since verified retention")
    for name, digest in hashes.items():
        if O.sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("Reviewed diagnostic helper changed: " + name)
    deadline_check(deadline, records["guard-armed.json"], c["lease_timestamp"], time.time())
    root.mkdir()
    O.write(root / "recovery-plan.json", {"config": c, "prior_directory": str(prior),
        "prior_receipt_sha256": receipt_hashes, "source_sha256": hashes, "allocation": allocation,
        "fixed_original_deadline": deadline, "deadline_extended": False, "order": proof["order"],
        "port": port, "main_submission_id": proof["job"]["submission_id"],
        "interpretation": "Fresh standalone initial-policy diagnostic; main remains stopped and unchanged"})
    O.node(c, "prepare")
    for name in O.SOURCES:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(Path(__file__).with_name(name)),
                        c["node"] + ":" + c["node_base"] + "/source/" + name], check=True, timeout=30)
    allocated()
    stopped(c, inspect_id(proof["main_id"]), proof["main_id"])
    stopped(old_c, inspect_id(proof["old_identity"]["container_id"]), proof["old_identity"]["container_id"], True)
    deadline_check(deadline, records["guard-armed.json"], c["lease_timestamp"], time.time())
    diagnostic_id = O.remote(c, O.docker_command(c)).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", diagnostic_id):
        raise ValueError("Exact new diagnostic container ID required")
    try:
        info = inspect_id(diagnostic_id); O.container_identity(c, info, c["image_id"], True)
        # Arm immediately; all later setup errors still leave a bounded container.
        guard = O.node(c, "guard", guard_source=O.guard_source(c, diagnostic_id, deadline), deadline=deadline)
        for _ in range(20):
            try:
                armed = json.loads(O.remote(c, ["cat", c["node_base"] + "/guard-state.json"], 10)); break
            except subprocess.CalledProcessError: time.sleep(.2)
        else: raise RuntimeError("Recovery host guard failed to arm")
        if armed != {"state": "armed", "container_id": diagnostic_id, "deadline": deadline}:
            raise ValueError("Recovery guard identity/deadline mismatch")
        check = "import os;from pathlib import Path;p=" + repr(guard["pid"]) + ";os.kill(p,0);assert int(Path('/proc',str(p),'stat').read_text().rpartition(') ')[2].split()[19])==" + repr(guard["start_ticks"])
        O.remote(c, ["python3", "-c", check])
        O.write(root / "guard-armed.json", {**guard, **armed, "inherited_from": str(prior / "guard-armed.json")})
        identity = {"run_id": c["diagnostic_run"], "container_name": c["diagnostic_name"], "container_id": diagnostic_id,
            "image_reference": c["image"], "image_id": c["image_id"], "uid": 28644, "gid": 30,
            "labels": {"miles.graph_diagnostic": c["diagnostic_run"]}, "docker_inspect_verified_at": O.utc(),
            "main_terminal_verified": True, "source_commits": {"miles_main": c["source_commit"], "sglang": O.capture.SGLANG_COMMIT},
            "diagnostic_source_sha256": hashes, "main_submission_id": proof["job"]["submission_id"]}
        identity = O.node(c, "identity", hashes=hashes, identity=identity)
        if any(identity.get(k) != proof["old_identity"].get(k) for k in ["initial_model_inventory", "initial_model_id"]):
            raise ValueError("Initial model differs from prior verified diagnostic input")
        O.write(root / "diagnostic-identity.json", identity)
        probe = "from pathlib import Path;import os;assert(os.getuid(),os.getgid())==(28644,30);p=Path('/run-output/.host-writer-probe');assert p.read_text()==" + repr(c["diagnostic_run"]) + ";p.unlink();q=Path('/run-output/.container-writer-probe');q.write_text('container28644');assert q.stat().st_uid==28644"
        O.remote(c, ["docker", "exec", diagnostic_id, "python3", "-c", probe])
        O.remote(c, ["python3", "-c", "from pathlib import Path;p=Path(" + repr(c["node_base"] + "/run/.container-writer-probe") + ");assert p.read_text()=='container28644' and p.stat().st_uid==28644;p.unlink()"])
        O.prepare_gb_editable_sources(c, diagnostic_id, {**guard, **armed})
        deadline_check(deadline, records["guard-armed.json"], c["lease_timestamp"], time.time())
        argv = ["docker", "exec", diagnostic_id, "python3", "-u", "/opt/diagnostic/run_sglang_graph_diagnostic.py",
            "--identity-file", "/run-output/diagnostic-identity.json", "--model-path", "/models/Qwen3-30B-A3B",
            "--dataset", "/inputs/train.jsonl", "--output-dir", "/run-output/diagnostic", "--host", c["node_ip"],
            "--port", str(port), "--order", proof["order"], "--deadline-utc", O.utc(deadline), "--execute"]
        with (root / "wrapper-console.log").open("x") as log:
            try:
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", c["node"], shlex.join(argv)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time()+10))
                O.write(root / "wrapper-exit.json", {"at": O.utc(), "exit_code": result.returncode, "original_deadline": deadline})
            except BaseException as error:
                O.write(root / "wrapper-exit.json", {"at": O.utc(), "exit_code": None, "error": repr(error), "original_deadline": deadline})
                raise
    finally:
        # Stop only this exact new diagnostic, then retain stable closed files.
        final = inspect_id(diagnostic_id); O.container_identity(c, final, c["image_id"], True, require_running=False)
        if final["Id"] != diagnostic_id: raise ValueError("Recovery diagnostic replaced")
        if final["State"]["Running"]: O.remote(c, ["docker", "stop", "--time", "10", diagnostic_id], 30)
        O.write(root / "diagnostic-stopped.json", {"at": O.utc(), "container_id": diagnostic_id})
        try:
            before = O.node(c, "manifest")
            O.retain(c, "run", before, root / "node-output", timeout=min(300, max(1, c["lease_timestamp"]-time.time()-45)))
            if O.node(c, "manifest") != before: raise ValueError("Stopped diagnostic output changed during retention")
            O.write(root / "diagnostic-retention.json", {"at": O.utc(), "files": before,
                "source_before_after_and_destination_verified": True, "container_stopped_before_retention": True})
        except BaseException as error:
            O.write(root / "diagnostic-retention-failure.json", {"at": O.utc(), "error": repr(error), "partial_files_preserved": True})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prior-directory", type=Path, required=True)
    p.add_argument("--prior-evidence-manifest", type=Path, required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--deadline-utc", required=True)
    p.add_argument("--wrapper-sha256", required=True); p.add_argument("--capture-sha256", required=True)
    p.add_argument("--port", type=int, default=31081); p.add_argument("--execute", action="store_true")
    a = p.parse_args(); c = O.main_driver._config(load(a.config))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{5,79}", a.run_id) or not 1024 <= a.port <= 65535:
        raise ValueError("Fresh scoped diagnostic run ID and valid port required")
    if not a.prior_directory.is_absolute(): raise ValueError("Absolute prior evidence directory required")
    hashes = dict(zip(O.SOURCES, [a.wrapper_sha256, a.capture_sha256]))
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()): raise ValueError("Reviewed source hashes required")
    c.update(diagnostic_run=a.run_id, diagnostic_name="miles-graph-diagnostic-" + a.run_id,
             node_base=str(Path(c["node_run_dir"]).parent / ("graph-diagnostic-" + a.run_id)),
             durable=str(Path(c["run_dir"]) / "diagnostics" / a.run_id))
    requested_deadline = O.main_driver._timestamp(a.deadline_utc)
    records, receipt_hashes = evidence(a.prior_directory, a.prior_evidence_manifest)
    deadline_check(requested_deadline, records["guard-armed.json"], c["lease_timestamp"], time.time())
    # The receipt's original numeric deadline is authoritative; CLI formatting
    # can round sub-microsecond precision but never creates a new allowance.
    deadline = records["guard-armed.json"]["deadline"]
    print(json.dumps({"mode": "EXECUTE" if a.execute else "PLAN_ONLY_NO_REMOTE_CALLS", "config": c,
        "prior_directory": str(a.prior_directory), "prior_receipt_sha256": receipt_hashes,
        "source_sha256": hashes, "fixed_original_deadline": O.utc(deadline), "deadline_extended": False,
        "order": inherited_order(records), "port": a.port, "docker_argv": O.docker_command(c)}, indent=2), flush=True)
    if a.execute: execute(c, a.prior_directory, records, receipt_hashes, deadline, hashes, a.port)


if __name__ == "__main__": main()
