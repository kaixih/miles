#!/usr/bin/env python3
"""Submit one preflighted graph-on Qwen3/GSM8K run; plan-only by default.

Run on the login host as the prepared container's normal UID. Root orchestration
must first allocate, stage inputs, bootstrap the single four-GPU Ray container,
and complete the communication check. This driver does none of those actions.

  python3 run_cudagraph_main.py --config /new/run/driver-config.json
  nohup python3 run_cudagraph_main.py --config /new/run/driver-config.json \
      --execute > /new/run/logs/driver-console.log 2>&1 &

Config requires platform, run_id, job_id, node, node_ip, image (digest), repo,
node_repo, models, node_models, run_dir (durable login view), node_run_dir,
cache_dir, container_prefix, ray_port, dashboard_port, megatron_path, nccl_iface,
lease_deadline (timezone required), gate_log, source_commit, train_sha256 and
eval_sha256. UID/GID default to 28644/30. Optional source_manifest is an absolute
JSON path with git_commit and source_sha256 (relative path -> SHA256), for a
frozen source snapshot without .git. All paths must already be prepared.
The supplied lease must be the allocation's verified absolute expiration time.

The fixed recipe has 50 rollouts, 256 prompts x 8 samples, global batch 512,
512/1024 prompt/response limits, 4096 training tokens/GPU, decode CUDA Graph on,
prefill graphs off, and a final full checkpoint at interval 50. Native retention
interval 1000000 keeps the latest nonzero checkpoint in this 50-rollout run.
Other model,
optimizer, reward and evaluation settings remain the existing launcher defaults.
An immutable watchdog uses soft=lease-90min and hard=lease-60min. It is detached
inside the container and must report fresh exact armed state before submission.
Existing run evidence is refused; failures require inspection, never a budget reset.

Node-local outputs still need the caller's established durable retention step.
The driver writes orchestration/exit evidence but never stops a container or
labels an exit-zero/STOPPED Ray job as completed learning. Inspect the watchdog,
actual Ray terminal job, 50 rollout/200 update logs and checkpoint separately.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


REQUIRED = {
    "platform", "run_id", "job_id", "node", "node_ip", "image", "repo", "node_repo",
    "models", "node_models", "run_dir", "node_run_dir", "cache_dir", "container_prefix",
    "ray_port", "dashboard_port", "megatron_path", "nccl_iface", "lease_deadline",
    "gate_log", "source_commit", "train_sha256", "eval_sha256",
}
SOURCE_FILES = [
    "lab/rubin_two_node/orchestrate_rubin.py",
    "lab/rubin_two_node/run_qwen3_30b_a3b_gsm8k_rubin.py",
    "lab/rubin_two_node/gsm8k_verl_reward.py",
    "lab/rubin_two_node/watch_qwen3_run.py",
]


def _utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def _timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Lease requires an explicit timezone")
    return parsed.timestamp()


def _config(raw):
    if REQUIRED - raw.keys() or raw.keys() - REQUIRED - {"uid", "gid", "source_manifest"}:
        raise ValueError("Supply exactly the documented config fields plus optional uid/gid")
    c = {"uid": 28644, "gid": 30, **raw}
    if type(c["uid"]) is not int or c["uid"] <= 0 or type(c["gid"]) is not int or c["gid"] < 0:
        raise ValueError("Use a numeric normal-user UID and numeric GID")
    address = ipaddress.ip_address(c["node_ip"])
    if address.version != 4 or address.is_loopback or address.is_unspecified:
        raise ValueError("Use the explicit compute-node IPv4 address")
    if c["platform"] not in {"rubin", "gb300"}:
        raise ValueError("platform must be rubin or gb300")
    for key in ("run_id", "node", "container_prefix", "nccl_iface"):
        if not isinstance(c[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,126}", c[key]):
            raise ValueError("Invalid identifier: " + key)
    if len(c["run_id"]) < 6 or not re.fullmatch(r"[0-9]+", str(c["job_id"])):
        raise ValueError("Invalid run or allocation ID")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", c["image"]):
        raise ValueError("Use an immutable image digest")
    for key, length in (("source_commit", 40), ("train_sha256", 64), ("eval_sha256", 64)):
        if not re.fullmatch("[0-9a-f]{" + str(length) + "}", c[key]):
            raise ValueError("Invalid hash: " + key)
    for key in ("repo", "node_repo", "models", "node_models", "run_dir", "node_run_dir",
                "cache_dir", "megatron_path", "gate_log"):
        if not Path(c[key]).is_absolute() or ".." in Path(c[key]).parts:
            raise ValueError("Use an absolute, normalized path: " + key)
    for output, inputs in ((c["run_dir"], [c["repo"], c["models"]]),
                           (c["node_run_dir"], [c["node_repo"], c["node_models"]])):
        if any(Path(output) == Path(p) or Path(p) in Path(output).parents
               for p in inputs):
            raise ValueError("Output must not be a model/source directory or its child")
    if c.get("source_manifest") and not Path(c["source_manifest"]).is_absolute():
        raise ValueError("Source manifest path must be absolute")
    for key in ("ray_port", "dashboard_port"):
        if type(c[key]) is not int or not 1024 <= c[key] <= 65535:
            raise ValueError("Invalid port: " + key)
    if c["ray_port"] == c["dashboard_port"]:
        raise ValueError("Ray and dashboard ports must differ")
    c["lease_timestamp"] = _timestamp(c["lease_deadline"])
    c["soft_timestamp"] = c["lease_timestamp"] - 90 * 60
    c["hard_timestamp"] = c["lease_timestamp"] - 60 * 60
    c["container"] = c["container_prefix"] + "-0"
    c["dashboard"] = f"http://{c['node_ip']}:{c['dashboard_port']}"
    return c


def _launcher_args(c):
    return [
        "--run-id", c["run_id"], "--megatron-path", c["megatron_path"],
        "--prompt-data-path", "/run-output/inputs/train.jsonl",
        "--eval-prompt-data-path", "/run-output/inputs/test-fixed-256.jsonl",
        "--extra-env-vars", "RUBIN_RUN_ID=" + c["run_id"],
        "--sglang-enable-cuda-graph", "--max-tokens-per-gpu", "4096",
        "--rollout-batch-size", "256", "--n-samples-per-prompt", "8", "--global-batch-size", "512",
        "--rollout-max-prompt-len", "512", "--rollout-max-response-len", "1024",
        "--save-interval", "50", "--save-retain-interval", "1000000",
        "--save-trigger-sentinel", "/run-output/checkpoint-now",
    ]


def _plan(c):
    common = ["--nodes", c["node"], "--node-ips", c["node_ip"], "--job-id", str(c["job_id"]),
              "--uid", str(c["uid"]), "--gid", str(c["gid"]), "--image", c["image"],
              "--repo", c["repo"], "--node-repo", c["node_repo"], "--models", c["models"],
              "--node-models", c["node_models"], "--run-dir", c["run_dir"],
              "--node-run-dir", c["node_run_dir"], "--node-local-output", "--cache-dir", c["cache_dir"],
              "--container-prefix", c["container_prefix"], "--ray-port", str(c["ray_port"]),
              "--dashboard-port", str(c["dashboard_port"]), "--recipe", "qwen3-gsm8k", "--num-rollout", "50",
              "--env", "NCCL_SOCKET_IFNAME=" + c["nccl_iface"],
              "--env", "GLOO_SOCKET_IFNAME=" + c["nccl_iface"],
              "--env", "PYTHONPATH=/opt/miles:" + c["megatron_path"],
              "--launcher-args", shlex.join(_launcher_args(c))]
    watch = ["python3", "-B", "/opt/miles/lab/rubin_two_node/watch_qwen3_run.py",
             "--run-id", c["run_id"], "--ray-address", c["dashboard"],
             "--soft-deadline", _utc(c["soft_timestamp"]), "--hard-deadline", _utc(c["hard_timestamp"]),
             "--lease-deadline", _utc(c["lease_timestamp"]), "--sentinel", "/run-output/checkpoint-now",
             "--save-dir", "/run-output/checkpoints", "--state-path", "/run-output/watchdog.json",
             "--num-rollout", "50"]
    return {"schema_version": 1, "run_id": c["run_id"], "config": c,
            "mode": "PLAN_ONLY_NO_REMOTE_CALLS", "common_orchestrator_args": common,
            "train_command": [sys.executable, "-u", str(Path(c["repo"], SOURCE_FILES[0])), "train",
                              *common, "--gate-log", c["gate_log"]], "watchdog_command": watch,
            "recipe": {"rollouts": 50, "optimizer_updates": 200, "prompts_per_rollout": 256,
                       "samples_per_prompt": 8, "global_response_batch": 512,
                       "prompt_limit": 512, "response_limit": 1024, "training_tokens_per_gpu": 4096,
                       "decode_cuda_graph_requested": True, "prefill_cuda_graph_requested": False,
                       "save_interval": 50, "save_retain_interval": 1000000},
            "limitations": ["Requires preflighted storage, bootstrap and communication evidence.",
                            "Decode graph use requires actual capture/replay evidence; prefill stays eager.",
                            "Watchdog Ray API outages can delay a scoped stop; deadlines are never reset.",
                            "Exit zero is not proof of 50-rollout completion; validate Ray/log/checkpoint evidence.",
                            "Caller must retain node-local outputs before allocation expiry."]}


def _run(argv, timeout=60):
    return subprocess.check_output(argv, text=True, timeout=timeout)


def _node_python(c, code):
    return json.loads(_run(["ssh", "-o", "BatchMode=yes", c["node"],
                            shlex.join(["python3", "-B", "-c", code])]))


def _source(c):
    repo = Path(c["repo"])
    provenance = {}
    if c.get("source_manifest"):
        raw = Path(c["source_manifest"]).read_bytes()
        manifest = json.loads(raw)
        commit = manifest["git_commit"]
        hashes = manifest["source_sha256"]
        if not set(SOURCE_FILES) <= hashes.keys():
            raise ValueError("Snapshot manifest omits selected sources")
        for name, expected in hashes.items():
            p = repo / name
            if Path(name).is_absolute() or ".." in Path(name).parts or p.is_symlink():
                raise ValueError("Unsafe source manifest path")
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError("Snapshot source changed: " + name)
        provenance["source_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
        dirty = _run(["git", "-C", str(repo), "status", "--porcelain", "--", *SOURCE_FILES])
        if dirty:
            raise ValueError("Commit the selected driver/launcher sources before execution")
    if commit != c["source_commit"]:
        raise ValueError("Source commit changed")
    return {"git_commit": commit, **provenance, "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_sha256": {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in SOURCE_FILES}}


def _bootstrap(c, evidence):
    expected = {"image": c["image"], "nodes": [c["node"]], "node_ips": [c["node_ip"]],
                "job_id": str(c["job_id"]), "run_dir": c["run_dir"], "uid": c["uid"], "gid": c["gid"],
                "state": "ray_ready_training_not_started"}
    if any(evidence.get(k) != value for k, value in expected.items()):
        raise ValueError("Bootstrap evidence does not match the explicit run configuration")


def _armed(c, state, now):
    expected = {"state": "armed", "run_id": c["run_id"], "submission_id": None,
                "ray_address": c["dashboard"], "soft_deadline_at": _utc(c["soft_timestamp"]),
                "deadline_at": _utc(c["hard_timestamp"]), "lease_deadline_at": _utc(c["lease_timestamp"]),
                "sentinel": "/run-output/checkpoint-now", "save_dir": "/run-output/checkpoints",
                "expected_rollouts": 50, "stop_request_count": 0}
    if any(state.get(k) != value for k, value in expected.items()):
        raise ValueError("Watchdog identity/budget/state mismatch")
    if not 0 <= now - _timestamp(state["heartbeat_at"]) < 20 or now >= c["soft_timestamp"]:
        raise ValueError("Watchdog heartbeat stale or soft deadline reached")


def _preflight(c, source):
    # Read-only host/container validation. No docker start, image pull or Ray mutation.
    code = """import json,subprocess,hashlib
c=CONFIG
d=json.loads(subprocess.check_output(['docker','inspect',c['container']],text=True))[0]
assert d['State']['Running'] and d['Config']['Image']==c['image']
assert d['Config']['User']==str(c['uid'])+':'+str(c['gid'])
mounts={m['Destination']:m for m in d['Mounts']}
for target,source,readonly in [('/opt/miles',c['node_repo'],True),(c['models'],c['node_models'],True),('/run-output',c['node_run_dir'],False)]:
 m=mounts[target];assert m['Source']==source and m['RW']==(not readonly)
env=dict(x.split('=',1) for x in d['Config']['Env'] if '=' in x)
for k,v in {'RAY_ADDRESS':c['node_ip']+':'+str(c['ray_port']),'RAY_API_SERVER_ADDRESS':c['dashboard'],'MILES_SCRIPT_EXTERNAL_RAY':'1','NCCL_CUMEM_ENABLE':'1','NCCL_NVLS_ENABLE':'0','NCCL_SOCKET_IFNAME':c['nccl_iface'],'GLOO_SOCKET_IFNAME':c['nccl_iface']}.items():assert env.get(k)==v,k
assert c['megatron_path'] in env.get('PYTHONPATH','').split(':')
assert any(x['Name']=='nofile' and x['Soft']>=65535 and x['Hard']>=65535 for x in d['HostConfig']['Ulimits'])
inner=INNER
result=json.loads(subprocess.check_output(['docker','exec','--user',str(c['uid'])+':'+str(c['gid']),c['container'],'python3','-B','-c',inner],text=True))
print(json.dumps({'container_id':d['Id'],'image_id':d['Image'],'inputs_and_sources':result}))
"""
    inner = """import json,hashlib,os
from pathlib import Path
c=CONFIG
assert (os.getuid(),os.getgid())==(c['uid'],c['gid'])
root=Path('/run-output');assert root.is_dir() and root.stat().st_uid==c['uid']
for n in ['watchdog.json','watchdog.json.lock','watchdog-launch.json','checkpoint-now','train_exit.json']:
 assert not (root/n).exists(),n
for p in [root/'checkpoints',root/'logs/qwen3_train.log']:
 assert not p.exists(),str(p)
for n,key in [('train.jsonl','train_sha256'),('test-fixed-256.jsonl','eval_sha256')]:
 assert hashlib.sha256((root/'inputs'/n).read_bytes()).hexdigest()==c[key],n
for name,expected in SOURCES.items():
 assert hashlib.sha256((Path('/opt/miles')/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'dataset_hashes_verified':True,'source_hashes_verified':True,'new_run_evidence_absent':True}))
""".replace("CONFIG", repr(c)).replace("SOURCES", repr(source["source_sha256"]))
    return _node_python(c, code.replace("CONFIG", repr(c)).replace("INNER", repr(inner)))


def _start_watcher(c, plan):
    # The child is detached inside Docker, with its own output file, so login SSH
    # disconnects cannot leave sampling without the fixed-deadline watchdog.
    inner = """import json,os,subprocess,time
from pathlib import Path
argv=ARGV;root=Path('/run-output')
for n in ['watchdog.json','watchdog.json.lock','watchdog-launch.json']:
 assert not (root/n).exists(),n
with (root/'logs/watchdog.log').open('x') as log:
 p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
with (root/'watchdog-launch.json').open('x') as f:json.dump({'pid':p.pid,'start_ticks':start_ticks,'argv':argv},f)
deadline=time.monotonic()+30
while time.monotonic()<deadline:
 if p.poll() is not None:raise RuntimeError('Watchdog exited before arming')
 state=root/'watchdog.json'
 if state.exists():
  data=json.loads(state.read_text())
  if data.get('state')=='armed':
   print(json.dumps({'state':data,'pid':p.pid,'start_ticks':start_ticks}));break
 time.sleep(.25)
else:raise TimeoutError('Watchdog not armed; inspect, do not restart/reset it')
""".replace("ARGV", repr(plan["watchdog_command"]))
    code = """import subprocess,json
argv=ARGV
print(subprocess.check_output(argv,text=True,timeout=40))
""".replace("ARGV", repr(["docker", "exec", "--user", f"{c['uid']}:{c['gid']}", c["container"],
                              "python3", "-B", "-c", inner]))
    result = _node_python(c, code)
    _armed(c, result["state"], time.time())
    return result


def _write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _check_watcher(c, plan, launched):
    inner = """import json,os
from pathlib import Path
state=json.loads(Path('/run-output/watchdog.json').read_text())
pid=PID;os.kill(pid,0)
assert state['pid']==pid
assert int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])==TICKS
assert Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')==[x.encode() for x in ARGV]
print(json.dumps(state))
""".replace("PID", repr(launched["pid"])).replace("TICKS", repr(launched["start_ticks"])).replace("ARGV", repr(plan["watchdog_command"]))
    code = "import subprocess\nprint(subprocess.check_output(ARGV,text=True,timeout=20))".replace(
        "ARGV", repr(["docker", "exec", "--user", f"{c['uid']}:{c['gid']}", c["container"], "python3", "-B", "-c", inner]))
    state = _node_python(c, code)
    _armed(c, state, time.time())
    return {**launched, "state": state}


def _execute(c, plan):
    if (os.getuid(), os.getgid()) != (c["uid"], c["gid"]):
        raise ValueError("Run as the explicitly prepared normal UID/GID")
    if time.time() >= c["soft_timestamp"]:
        raise ValueError("Lease has less than the immutable 90-minute soft reserve")
    root = Path(c["run_dir"])
    if not root.is_dir() or root.stat().st_uid != c["uid"] or not (root / "logs").is_dir():
        raise ValueError("Run output/log directories must already be preflighted")
    for name in ["cudagraph-main-plan.json", "cudagraph-main.claim", "train-launch.json",
                 "train-driver-exit.json", "train_exit.json", "logs/qwen3_train.log", "logs/train-driver.log"]:
        if (root / name).exists():
            raise ValueError("Refuse existing run evidence: " + name)
    source = _source(c)
    _bootstrap(c, json.loads((root / "bootstrap.json").read_text()))
    gate = Path(c["gate_log"]).read_bytes()
    if b"ALLREDUCE_OK" not in gate or b"world=4" not in gate:
        raise ValueError("Communication gate must pass for four GPUs")
    spec = importlib.util.spec_from_file_location("_cudagraph_orchestrator", Path(c["repo"], SOURCE_FILES[0]))
    orchestration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestration)
    args = orchestration._parser().parse_args(["train", *plan["common_orchestrator_args"], "--gate-log", c["gate_log"]])
    orchestration._check_allocation(args)
    cluster = orchestration._cluster_status(args)
    if cluster.returncode:
        raise RuntimeError("Prepared Ray cluster does not match one node/four GPUs")
    inspected = _preflight(c, source)
    _write_new(root / "cudagraph-main.claim", {"run_id": c["run_id"], "pid": os.getpid(), "at": _utc(time.time())})
    actual = {**plan, **source, "mode": "EXECUTE", "preflight": inspected,
              "gate_sha256": hashlib.sha256(gate).hexdigest(), "started_at": _utc(time.time())}
    _write_new(root / "cudagraph-main-plan.json", actual)
    armed = _start_watcher(c, plan)
    if _source(c) != source:
        raise ValueError("Selected source files changed after watchdog armed")
    armed = _check_watcher(c, plan, armed)
    _write_new(root / "train-launch.json", {**source, "stage": "train", "started_utc": _utc(time.time()),
                                           "command": plan["train_command"], "watchdog": armed})
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with (root / "logs/train-driver.log").open("x") as log:
            result = subprocess.run(plan["train_command"], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=max(1, c["hard_timestamp"] - time.time() + 120))
        code, reason = result.returncode, "orchestrator_exit"
    except subprocess.TimeoutExpired:
        code, reason = 124, "driver_wait_timeout_watchdog_keeps_original_deadline"
    _write_new(root / "train-driver-exit.json", {"exit_code": code, "reason": reason,
                                                "finished_utc": _utc(time.time()), "run_id": c["run_id"]})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    c = _config(json.loads(args.config.read_text()))
    plan = _plan(c)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    return _execute(c, plan)


if __name__ == "__main__":
    raise SystemExit(main())
