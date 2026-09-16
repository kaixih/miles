#!/usr/bin/env python3
"""Capture one independent replay engine. Run on its compute host as UID28644.

Default prints a plan without Docker or HTTP calls. Execution requires an explicit
container/digest, engine URL/PID/start ticks, and the submitted profile-plan.json.
Start while real replay generation is active; this tool never generates requests.
It records four consecutive scheduler forwards, not four requests or both stages.
The existing replay guard remains responsible for total trace size and lease.
HTTP/API outages can delay stopping; this is not a hard real-time limit.
"""

import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
RUN_ENV = "MILES_PROFILE_RUN_ID"
MAX_DECODED_TRACE = 512 * 1024**2


class IdentityError(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def origin(value):
    p = urllib.parse.urlsplit(value)
    if (p.scheme != "http" or not p.hostname or not p.port or p.username or p.password
            or p.path not in {"", "/"} or p.query or p.fragment):
        raise ValueError("Require an explicit HTTP host:port origin without credentials")
    return value.rstrip("/")


def assert_job(plan, job):
    env = (job.get("runtime_env") or {}).get("env_vars") or {}
    if (job.get("type") != "SUBMISSION" or job.get("submission_id") != plan["submission_id"]
            or env.get(RUN_ENV) != plan["run_id"]
            or sha((job.get("entrypoint") or "").encode()) != plan["entrypoint_sha256"]
            or job.get("status") not in TERMINAL | {"PENDING", "RUNNING"}):
        raise IdentityError("Exact replay job identity mismatch; no API mutation permitted")


def assert_container(info, args, plan, image_id):
    config = info.get("Config", {})
    if (info.get("Name") != "/miles-profile-" + plan["run_id"]
            or args.container != "miles-profile-" + plan["run_id"]
            or config.get("Labels", {}).get("miles.profile_run") != plan["run_id"]
            or config.get("Image") != args.image or info.get("Image") != image_id
            or config.get("User") != "28644:30" or not info.get("State", {}).get("Running")):
        raise IdentityError("Expected a running, labeled, normal-UID profile container with the exact image")
    mounts = [m for m in info.get("Mounts", []) if m.get("Destination") == "/run-output"]
    if len(mounts) != 1 or mounts[0].get("Type") != "bind" or not mounts[0].get("RW"):
        raise IdentityError("Expected one writable profile-output bind mount")
    return Path(mounts[0]["Source"])


def remaining_capture(plan, now, timeout):
    deadline = plan.get("deadline_timestamp")
    armed = plan.get("armed_timestamp")
    if (not isinstance(deadline, (int, float)) or not isinstance(armed, (int, float))
            or not math.isfinite(deadline) or not math.isfinite(armed) or not armed < deadline):
        raise ValueError("Use the submitted plan with its original absolute guard deadline")
    if not 1 <= timeout <= 180:
        raise ValueError("Capture timeout must be 1..180 seconds")
    return min(deadline, now + timeout)


def trace_evidence(path):
    if path.is_symlink():
        raise IdentityError("Trace symlinks are forbidden")
    before = path.stat()
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_DECODED_TRACE + 1)
    if len(raw) > MAX_DECODED_TRACE:
        raise ValueError("Trace exceeds bounded CPU inspection size")
    data = json.loads(raw)
    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError("Expected Chrome traceEvents list")
    categories = Counter(e.get("cat", "") for e in events
        if isinstance(e, dict) and e.get("ph") == "X"
        and isinstance(e.get("ts"), (int, float)) and math.isfinite(e["ts"])
        and isinstance(e.get("dur"), (int, float)) and math.isfinite(e["dur"]) and e["dur"] > 0)
    if categories["cpu_op"] == 0 or categories["kernel"] == 0:
        raise ValueError("Trace does not prove both CPU operators and CUDA kernels")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Trace still changing")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"path": str(path), "bytes": after.st_size, "sha256": h.hexdigest(),
            "event_count": len(events), "complete_event_categories": dict(categories),
            "stage_name_hints": sorted({str(e.get("name", ""))[:160] for e in events
                if isinstance(e, dict) and re.search(r"decode|prefill|extend", str(e.get("name", "")), re.I)})[:30],
            "scope": "Actual CPU/CUDA trace; stage-name hints alone do not prove complete prefill/decode coverage"}


ENGINE_PROBE = r'''
import json,os,pathlib,sys
pid,expected_start,run_id=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
p=pathlib.Path('/proc')/str(pid)
stat=p.joinpath('stat').read_text().rpartition(') ')[2].split()
assert int(stat[19])==expected_start, 'engine PID reused'
argv=[x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]
assert len(argv)>3 and argv[1:3]==['-m','sglang.launch_server'], 'not direct SG engine'
env=dict(x.split(b'=',1) for x in p.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
assert env.get(b'MILES_PROFILE_RUN_ID',b'').decode()==run_id, 'not this replay engine'
assert env.get(b'SGLANG_PROFILE_V2',b'').lower() in (b'',b'0',b'false'), 'V2 unsupported'
def value(flag,default=None):
 values=[argv[i+1] for i,x in enumerate(argv[:-1]) if x==flag]
 assert len(values)<=1, flag
 return values[0] if values else default
port=int(value('--port'))
owned=set()
for fd in p.joinpath('fd').iterdir():
 try:
  link=os.readlink(fd)
  if link.startswith('socket:['):owned.add(link[8:-1])
 except FileNotFoundError:pass
listeners=[]
for name in ('tcp','tcp6'):
 for line in pathlib.Path('/proc/net',name).read_text().splitlines()[1:]:
  fields=line.split()
  if fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==port and fields[9] in owned:
   listeners.append(fields[9])
assert listeners, 'engine PID does not own this listening port'
print(json.dumps({'pid':pid,'start_ticks':expected_start,'host':value('--host'), 'port':port,
 'started_timestamp':int(next(x.split()[1] for x in pathlib.Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))+expected_start/os.sysconf('SC_CLK_TCK'),
 'tp_size':int(value('--tp-size','1')),'pp_size':int(value('--pp-size','1')),
 'base_gpu_id':int(value('--base-gpu-id','0')),'model_path':value('--model-path'),
 'profile_v2':False,'run_id':run_id,'argv_sha256':__import__('hashlib').sha256(p.joinpath('cmdline').read_bytes()).hexdigest()}))
'''


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=15)


def request(url, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=5) as response:
        return {"status": response.status, "body": response.read(2 * 1024**2).decode(), "at": utc()}


def persist(path, state):
    state["heartbeat_at"] = utc()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def execute(args, plan):
    if (os.getuid(), os.getgid()) != (28644, 30):
        raise ValueError("Run on the compute host as UID28644:GID30")
    if (plan["submission_id"] != "qwen3-profile-" + plan["run_id"]
            or plan["output_dir"] != "/run-output/replay"):
        raise IdentityError("Expected this operator's separate replay output and submission")
    dashboard = origin(plan["ray_address"])
    job_url = dashboard + "/api/jobs/" + urllib.parse.quote(plan["submission_id"], safe="")
    image_id = json.loads(command(["docker", "image", "inspect", args.image]))[0]["Id"]

    def checked_job():
        job = json.loads(request(job_url)["body"])
        assert_job(plan, job)
        return job

    def checked_engine():
        info = json.loads(command(["docker", "inspect", args.container]))[0]
        output = assert_container(info, args, plan, image_id)
        probe = json.loads(command(["docker", "exec", "--user", "28644:30", args.container,
            "python3", "-c", ENGINE_PROBE, str(args.engine_pid), str(args.engine_start_ticks), plan["run_id"]]))
        url = urllib.parse.urlsplit(args.engine_url)
        if probe["host"] != url.hostname or probe["port"] != url.port or probe["tp_size"] != 1 or probe["pp_size"] != 1:
            raise IdentityError("Engine URL or TP/PP differs from its actual process")
        started = dt.datetime.fromisoformat(info["State"]["StartedAt"].replace("Z", "+00:00")).timestamp()
        if started > plan["armed_timestamp"] or probe["started_timestamp"] + 1 < started:
            raise IdentityError("Container was restarted, or engine predates the fresh container")
        return output, probe, info["Id"]

    output, engine, container_id = checked_engine()
    actual_plan = output / "replay/profile-plan.json"
    if json.loads(actual_plan.read_text()) != plan:
        raise IdentityError("Input plan differs from actual mounted replay plan")
    if checked_job()["status"] != "RUNNING":
        raise IdentityError("Exact profile replay must already be RUNNING")
    deadline = remaining_capture(plan, time.time(), args.capture_timeout)
    if deadline - time.time() < 15:
        raise ValueError("Too little time before the original profile deadline")
    info = json.loads(request(args.engine_url + "/server_info")["body"])
    for key in ("host", "port", "tp_size", "pp_size", "base_gpu_id", "model_path"):
        if info.get(key) != engine[key]:
            raise IdentityError("Resolved server identity mismatch: " + key)
    state_path = output / "replay/sglang-capture.json"
    # Persistent exclusive claim: never automatically re-arm after a crash/timeout.
    with state_path.with_suffix(".claim").open("x") as claim:
        claim.write(json.dumps({"pid": os.getpid(), "at": utc(), "run_id": plan["run_id"]}))
    traces = output / "replay/traces/sglang"
    traces.mkdir(exist_ok=False)
    profile_id = plan["run_id"] + "-engine" + str(args.engine_pid)
    payload = {"output_dir": "/run-output/replay/traces/sglang", "num_steps": 4,
               "activities": ["CPU", "GPU"], "profile_by_stage": False, "with_stack": False,
               "record_shapes": False, "merge_profiles": False, "detailed_annotations": False,
               "profile_id": profile_id}
    trace = traces / (profile_id + "-TP-0.trace.json.gz")
    state = {"state": "verified", "run_id": plan["run_id"], "submission_id": plan["submission_id"],
             "container_id": container_id, "image": args.image, "engine": engine,
             "engine_url": args.engine_url, "payload": payload, "payload_sha256": sha(json.dumps(payload, sort_keys=True).encode()),
             "profile_deadline_timestamp": plan["deadline_timestamp"], "capture_deadline_timestamp": deadline,
             "started_at": utc(), "start_requests": 0, "stop_profile_requests": 0, "stop_job_requests": 0,
             "scope": "Four consecutive active scheduler forwards; actual stage scope must be read from trace"}
    persist(state_path, state)

    def stop_exact_job(reason):
        job = checked_job()
        state["stop_reason"] = reason
        if job["status"] not in TERMINAL:
            state["stop_job_requests"] += 1
            state["stop_job_http"] = request(job_url + "/stop", "POST")
        state["state"] = "incomplete_stop_requested"
        persist(state_path, state)

    try:
        if checked_engine()[1] != engine or checked_job()["status"] != "RUNNING":
            raise IdentityError("Replay changed immediately before arm")
        if time.time() >= deadline:
            stop_exact_job("Capture deadline elapsed during identity checks; no profiler was armed")
            return
        state.update(state="arming", start_requests=1)
        persist(state_path, state)
        try:
            state["start_http"] = request(args.engine_url + "/start_profile", "POST", payload)
        except Exception as exc:
            state["start_error"] = repr(exc)  # Ambiguous HTTP timeout: never retry start.
            if isinstance(exc, urllib.error.HTTPError):
                state["start_http"] = {"status": exc.code, "body": exc.read(4096).decode(errors="replace")}
        state["state"] = "observing"
        persist(state_path, state)
        while time.time() < deadline:
            checked_engine()
            job = checked_job()
            if trace.exists():
                try:
                    state["trace"] = trace_evidence(trace)
                    state["state"] = "captured"
                    persist(state_path, state)
                    return
                except (OSError, ValueError, EOFError) as exc:
                    state["last_trace_error"] = repr(exc)
            if job["status"] in TERMINAL:
                state.update(state="incomplete_job_terminal", ray_status=job["status"])
                persist(state_path, state)
                return
            persist(state_path, state)
            time.sleep(min(2, max(0, deadline - time.time())))
        checked_engine()
        checked_job()
        state["stop_profile_requests"] += 1
        try:
            state["stop_profile_http"] = request(args.engine_url + "/stop_profile", "POST")
            if trace.exists():
                state["trace"] = trace_evidence(trace)
                state["state"] = "partial_capture_manually_stopped"
                persist(state_path, state)
                return
        except Exception as exc:
            state["stop_profile_error"] = repr(exc)
            if isinstance(exc, urllib.error.HTTPError):
                state["stop_profile_http"] = {"status": exc.code, "body": exc.read(4096).decode(errors="replace")}
        stop_exact_job("Capture deadline; capture completion or profiler cancellation unproven")
    except IdentityError as exc:
        state.update(state="rejected_identity_change", error=str(exc))
        persist(state_path, state)
        raise  # Do not mutate anything after an ownership mismatch.
    except Exception as exc:
        state["error"] = repr(exc)
        try:
            stop_exact_job("Observer error after single arm; retain existing guard")
        finally:
            persist(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine-url", type=origin, required=True)
    parser.add_argument("--engine-pid", type=int, required=True)
    parser.add_argument("--engine-start-ticks", type=int, required=True)
    parser.add_argument("--capture-timeout", type=int, default=180)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    remaining_capture(plan, time.time(), args.capture_timeout)
    if not re.search(r"@sha256:[0-9a-f]{64}$", args.image):
        parser.error("Use the exact immutable image digest")
    if args.execute:
        execute(args, plan)
    else:
        print(json.dumps({"mode": "PLAN_ONLY_NO_DOCKER_OR_HTTP", "container": args.container,
                          "image": args.image, "engine_url": args.engine_url, "engine_pid": args.engine_pid,
                          "submission_id": plan["submission_id"], "profile_by_stage": False, "num_steps": 4,
                          "capture_timeout_seconds": args.capture_timeout, "original_deadline": plan["deadline_timestamp"]}, indent=2))


if __name__ == "__main__":
    main()
