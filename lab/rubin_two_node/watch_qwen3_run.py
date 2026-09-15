#!/usr/bin/env python3
"""Bound one exact Ray submission, requesting a synchronous checkpoint first.

Run inside the head container before submission. Pass the SAME sentinel to Miles
as --save-trigger-sentinel, and include RUBIN_RUN_ID in the Ray runtime env.
The soft deadline requests a save at the next training boundary. A completed
save permits stopping; the hard deadline stops even if a save is unfinished.
Ray API outages can delay stopping. A stopped run is partial, never a claim that
all requested rollouts completed. This program never deletes checkpoints.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import time
import urllib.parse
import urllib.request

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
STATUSES = TERMINAL | {"PENDING", "RUNNING"}


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadlines require an explicit timezone")
    return parsed.timestamp()


def is_target(job, run_id):
    env = (job.get("runtime_env") or {}).get("env_vars") or {}
    if job.get("type") != "SUBMISSION" or env.get("RUBIN_RUN_ID") != run_id:
        return False
    if not isinstance(job.get("submission_id"), str) or not job["submission_id"]:
        return False
    try:
        argv = shlex.split(job.get("entrypoint") or "")
    except ValueError:
        return False
    return bool(len(argv) > 1 and re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(argv[0]).name)
                and Path(argv[1]).name == "train.py")


class JobsAPI:
    def __init__(self, address, timeout=10):
        parsed = urllib.parse.urlsplit(address)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("Ray address must be a dashboard HTTP(S) origin without credentials")
        self.address, self.timeout = address.rstrip("/"), timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path):
        request = urllib.request.Request(self.address + path, method=method,
                                         data=b"" if method == "POST" else None)
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def list_jobs(self):
        jobs = self.request("GET", "/api/jobs/")
        if not isinstance(jobs, list) or any(not isinstance(j, dict) or j.get("status") not in STATUSES for j in jobs):
            raise ValueError("Unknown Ray job list/status")
        return jobs

    def get_job(self, job_id):
        job = self.request("GET", "/api/jobs/" + urllib.parse.quote(job_id, safe=""))
        if not isinstance(job, dict) or job.get("status") not in STATUSES:
            raise ValueError("Unknown Ray job/status")
        return job

    def stop_job(self, job_id):
        result = self.request("POST", "/api/jobs/" + urllib.parse.quote(job_id, safe="") + "/stop")
        if not isinstance(result, dict) or type(result.get("stopped")) is not bool:
            raise ValueError("Unknown Ray stop response")
        return result["stopped"]


def checkpoint_info(save_dir):
    tracker = save_dir / "latest_checkpointed_iteration.txt"
    try:
        raw = tracker.read_text().strip()
    except FileNotFoundError:
        return None
    if not raw.isdecimal():
        raise ValueError("Expected a numeric training checkpoint tracker")
    iteration = int(raw)
    directory = save_dir / f"iter_{iteration:07d}"
    return {"iteration": iteration, "directory": str(directory), "directory_exists": directory.is_dir()}


def request_checkpoint(state, sentinel, save_dir, now):
    baseline = checkpoint_info(save_dir)
    # Exclusive creation prevents taking ownership of an unrelated save request.
    with sentinel.open("x") as stream:
        json.dump({"run_id": state["run_id"], "requested_at": utc(now)}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    state.update(checkpoint_requested_at=utc(now), checkpoint_baseline=baseline,
                 sentinel_created=True, sentinel_seen=True)


def checkpoint_ready(state, sentinel, save_dir):
    current = checkpoint_info(save_dir)
    state["latest_checkpoint"] = current
    baseline = state.get("checkpoint_baseline")
    advanced = current is not None and (baseline is None or current["iteration"] > baseline["iteration"])
    # Miles removes its sentinel only AFTER force_sync save and rollout state save.
    return bool(state.get("sentinel_created") and state.get("sentinel_seen")
                and advanced and current["directory_exists"] and not sentinel.exists())


def stop_reason(state, sentinel, save_dir, now, soft_deadline, hard_deadline):
    if now >= hard_deadline:
        return "hard_deadline"
    if now >= soft_deadline:
        if not state.get("sentinel_created"):
            request_checkpoint(state, sentinel, save_dir, now)
        if checkpoint_ready(state, sentinel, save_dir):
            state["checkpoint_confirmed_at"] = utc(now)
            return "soft_deadline_checkpoint_confirmed"
    return None


def progress(log):
    if log is None or not log.exists():
        return {}
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 262144))
        text = stream.read().decode(errors="replace")
    matches = re.findall(r"train op=train_step rollout=(\d+) step=(\d+).*?outcome=NORMAL valid_step=true", text)
    return {"last_logged_valid_step": {"rollout": int(matches[-1][0]), "step": int(matches[-1][1])}} if matches else {}


def persist(path, state):
    state["heartbeat_at"] = utc(time.time())
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def watch(api, args):
    jobs = api.list_jobs()
    if any(j["status"] not in TERMINAL or is_target(j, args.run_id) for j in jobs):
        raise ValueError("Cluster has active jobs or this run ID was already used")
    if args.sentinel.exists():
        raise ValueError("Sentinel already exists; inspect the previous request")
    existing = {j.get("submission_id") for j in jobs}
    state = dict(state="armed", run_id=args.run_id, pid=os.getpid(), armed_at=utc(time.time()),
                 submission_id=None, ray_address=api.address, soft_deadline_at=utc(args.soft_deadline),
                 deadline_at=utc(args.hard_deadline), lease_deadline_at=utc(args.lease_deadline),
                 expected_rollouts=args.num_rollout, stop_request_count=0, api_error_count=0,
                 training_complete_verified=False, sentinel=str(args.sentinel), save_dir=str(args.save_dir))
    persist(args.state_path, state)
    last_summary = None
    while True:
        now = time.time()
        job = None
        try:
            if state["submission_id"] is None:
                matches = [j for j in api.list_jobs() if is_target(j, args.run_id) and j["submission_id"] not in existing]
                if len(matches) > 1:
                    raise RuntimeError("Multiple submissions share this run ID")
                if matches:
                    job = matches[0]
                    state.update(submission_id=job["submission_id"], state="monitoring")
                elif now >= args.hard_deadline:
                    state.update(state="expired_no_job", outcome="no_training_submission")
                    persist(args.state_path, state)
                    return 3
            else:
                job = api.get_job(state["submission_id"])
                if job.get("submission_id") != state["submission_id"] or not is_target(job, args.run_id):
                    raise RuntimeError("Locked submission identity changed")
            if job is not None:
                state.update(ray_status=job["status"], driver_exit_code=job.get("driver_exit_code"))
                if job["status"] in TERMINAL:
                    state.update(state="finished", terminal_status=job["status"], completed_at=utc(now),
                                 outcome="partial" if state["stop_request_count"] or job["status"] != "SUCCEEDED" else "succeeded_unvalidated")
                    try:
                        state["latest_checkpoint"] = checkpoint_info(args.save_dir)
                        state.update(progress(args.log))
                    except OSError as exc:
                        state["final_artifact_read_error"] = type(exc).__name__
                    persist(args.state_path, state)
                    print(json.dumps(state, sort_keys=True), flush=True)
                    return 0
        except RuntimeError as exc:
            state.update(state="rejected", reason=str(exc))
            persist(args.state_path, state)
            return 2
        except Exception as exc:
            state["api_error_count"] += 1
            state["last_error_type"] = type(exc).__name__

        if state["submission_id"] is not None:
            try:
                reason = stop_reason(state, args.sentinel, args.save_dir, now, args.soft_deadline, args.hard_deadline)
                state.update(progress(args.log))
            except Exception as exc:
                state["checkpoint_error_type"] = type(exc).__name__
                reason = "hard_deadline" if time.time() >= args.hard_deadline else None
            if reason:
                state.update(state="stopping", stop_reason=reason, outcome="partial")
                state["stop_request_count"] += 1
                try:
                    state["last_stop_response"] = api.stop_job(state["submission_id"])
                except Exception as exc:
                    state["last_stop_error_type"] = type(exc).__name__
        try:
            persist(args.state_path, state)
        except OSError as exc:
            print(json.dumps({"event": "state_write_error", "type": type(exc).__name__}), flush=True)
        summary = {k: state.get(k) for k in ("state", "submission_id", "ray_status", "stop_reason", "latest_checkpoint", "last_logged_valid_step")}
        if summary != last_summary:
            print(json.dumps(summary, sort_keys=True), flush=True)
            last_summary = summary
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_API_SERVER_ADDRESS"))
    parser.add_argument("--soft-deadline", type=timestamp, required=True)
    parser.add_argument("--hard-deadline", type=timestamp, required=True)
    parser.add_argument("--lease-deadline", type=timestamp, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--num-rollout", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{5,127}", args.run_id):
        parser.error("Invalid unique run ID")
    if not time.time() < args.soft_deadline < args.hard_deadline < args.lease_deadline:
        parser.error("Require now < soft < hard < allocation lease deadline")
    if args.num_rollout <= 0 or not 0 < args.poll_seconds <= 60:
        parser.error("Positive rollout count and polling interval in (0,60] required")
    if not args.state_path.parent.is_dir() or not args.sentinel.parent.is_dir():
        parser.error("Preflighted state/sentinel parent directories must exist")
    os.umask(0o077)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with args.state_path.with_name(args.state_path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.state_path.exists():
            parser.error("Refuse existing state: a restart must not silently reset the budget")
        return watch(JobsAPI(args.ray_address), args)


if __name__ == "__main__":
    raise SystemExit(main())
