#!/usr/bin/env python3
"""Prepare one Docker/Ray container per allocated Rubin node; submit separately.

Run this standard-library host utility on dl3. It intentionally does not import
Miles: the login host need not contain the image's Torch/Ray dependencies. The
in-container training launcher uses Miles' normal execute_train contract.

Example:
  python3 orchestrate_rubin.py plan --nodes NODE_C17 NODE_C18
  python3 orchestrate_rubin.py bootstrap --nodes NODE_C17 NODE_C18
  python3 orchestrate_rubin.py train --nodes NODE_C17 NODE_C18 --gate-log GATE_LOG

bootstrap starts Ray only. train requires a successful eight-rank MNNVL log.
No allocation, checkpoint download, image build, or existing-container cleanup
is performed. All created persistent output uses the host user's UID/GID.
"""

import argparse
import concurrent.futures
import ipaddress
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "bootstrap", "status", "train"])
    parser.add_argument("--nodes", nargs=2, required=True, metavar=("HEAD_NODE", "WORKER_NODE"))
    parser.add_argument("--node-ips", nargs=2, default=["10.102.74.84", "10.102.74.85"])
    parser.add_argument("--job-id", default="2179787")
    parser.add_argument("--uid", type=int, default=28644)
    parser.add_argument("--gid", type=int, default=30)
    parser.add_argument("--image", default="gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c")
    parser.add_argument("--repo", default="/home/scratch.kaixih_ent/repo/miles-rubin-cu134")
    parser.add_argument("--models", default="/home/scratch.kaixih_ent/models")
    parser.add_argument("--run-dir", default="/home/scratch.kaixih_ent/repro/miles-rubin-two-node/20260914-j2179787")
    parser.add_argument("--cache-dir", default="/tmp/miles-rubin-j2179787/qwen35")
    parser.add_argument("--ray-port", type=int, default=26379)
    parser.add_argument("--dashboard-port", type=int, default=28265)
    parser.add_argument("--num-cpus", type=int, default=32,
                        help="Logical Ray CPUs per node; bounds eager Python worker startup")
    parser.add_argument("--container-prefix", default="miles-rubin-qwen35-j2179787")
    parser.add_argument("--gate-log", type=Path)
    parser.add_argument("--num-rollout", type=int, default=2)
    parser.add_argument("--launcher-args", default="",
                        help="Additional shell-quoted arguments forwarded to the training launcher")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="Additional environment on both nodes; match the communication check")
    return parser


def _run(command, *, capture=False, check=True):
    print("+ " + shlex.join(command), flush=True)
    return subprocess.run(command, text=True, check=check, capture_output=capture)


def _ssh(node, command, *, capture=False, check=True):
    return _run(["ssh", "-o", "BatchMode=yes", node, shlex.join(command)], capture=capture, check=check)


def _name(args, rank):
    return f"{args.container_prefix}-{rank}"


def _environment(args, rank):
    env = {
        "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/opt/miles:/opt/Megatron-LM",
        "HOME": "/cache/home",
        "PYTHONPYCACHEPREFIX": "/cache/pycache", "XDG_CACHE_HOME": "/cache/xdg",
        "TMPDIR": "/cache/tmp", "RAY_TMPDIR": "/cache/ray",
        "TRITON_CACHE_DIR": "/cache/triton", "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
        "TORCH_EXTENSIONS_DIR": "/cache/torch_extensions", "CUDA_CACHE_PATH": "/cache/cuda",
        "HF_HOME": "/cache/huggingface", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "FLA_DISABLE_BACKEND_DISPATCH": "1", "FLA_CONV_BACKEND": "triton",
        "MASTER_ADDR": args.node_ips[0], "MILES_SCRIPT_EXTERNAL_RAY": "1",
        "RAY_ADDRESS": f"{args.node_ips[0]}:{args.ray_port}",
        "RAY_API_SERVER_ADDRESS": f"http://{args.node_ips[0]}:{args.dashboard_port}",
        "RAY_USAGE_STATS_ENABLED": "0", "RAY_DEDUP_LOGS": "0",
        "NCCL_DEBUG": "INFO", "NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "1",
        "NCCL_SOCKET_IFNAME": "mp0", "GLOO_SOCKET_IFNAME": "mp0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1", "MAX_JOBS": "8",
        "no_proxy": ",".join(["localhost", "127.0.0.1", *args.node_ips, *args.nodes]),
        "NO_PROXY": ",".join(["localhost", "127.0.0.1", *args.node_ips, *args.nodes]),
        "MILES_RUBIN_NODE_RANK": str(rank),
    }
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Expected KEY=VALUE: {item}")
        env[key] = value
    return env


def _docker(args, rank, *, probe=False):
    cmd = ["docker", "run", "--gpus", "all", "--network", "host", "--ipc", "host",
           "--privileged", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
           "--shm-size", "16g", "--user", f"{args.uid}:{args.gid}", "--workdir", "/opt/miles"]
    cmd += ["--rm"] if probe else ["--detach", "--name", _name(args, rank)]
    for source, target, readonly in [
        (args.repo, "/opt/miles", True), (args.models, args.models, True),
        (args.run_dir, "/run-output", False), (args.cache_dir, "/cache", False),
        ("/dev/nvidia-caps", "/dev/nvidia-caps", False),
        ("/dev/nvidia-caps-imex-channels", "/dev/nvidia-caps-imex-channels", False),
    ]:
        cmd += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in _environment(args, rank).items():
        cmd += ["--env", f"{key}={value}"]
    return cmd + [args.image]


def _ray(args, rank):
    cmd = ["ray", "start", "--node-ip-address", args.node_ips[rank], "--num-gpus", "4",
           "--num-cpus", str(args.num_cpus),
           "--disable-usage-stats", "--node-manager-port", "26380", "--object-manager-port", "26381",
           "--runtime-env-agent-port", "26382", "--min-worker-port", "26400", "--max-worker-port", "26999"]
    if rank == 0:
        return cmd + ["--head", "--port", str(args.ray_port), "--dashboard-host", args.node_ips[0],
                      "--dashboard-port", str(args.dashboard_port), "--temp-dir", "/cache/ray"]
    return cmd + ["--address", f"{args.node_ips[0]}:{args.ray_port}"]


def _training_command(args):
    return ["docker", "exec", _name(args, 0), "python3",
            "/opt/miles/lab/rubin_two_node/run_qwen3_5_35b_a3b_rubin.py",
            "--model-dir", args.models, "--data-dir", args.models,
            "--output-dir", "/run-output", "--megatron-path", "/opt/Megatron-LM",
            "--num-rollout", str(args.num_rollout), *shlex.split(args.launcher_args)]


def _check_allocation(args):
    result = _run(["scontrol", "show", "job", "-o", args.job_id], capture=True).stdout
    fields = dict(item.split("=", 1) for item in result.split() if "=" in item)
    if fields.get("JobState") != "RUNNING":
        raise RuntimeError(f"Job {args.job_id} is not RUNNING")
    allocated = _run(["scontrol", "show", "hostnames", fields["NodeList"]], capture=True).stdout.split()
    if set(args.nodes) != set(allocated):
        raise RuntimeError(f"Requested nodes {args.nodes} differ from allocated nodes {allocated}")


def _prepare_node(args, rank):
    node = args.nodes[rank]
    config = {"uid": args.uid, "gid": args.gid, "cache": args.cache_dir,
              "run": args.run_dir, "repo": args.repo, "models": args.models,
              "ports": [26380, 26381, 26382, 26400, 26999] + ([args.ray_port, args.dashboard_port] if rank == 0 else [])}
    code = """import json, os, pathlib, shutil, socket, tempfile
c=json.loads(CONFIG)
assert (os.getuid(),os.getgid())==(c['uid'],c['gid']), 'Unexpected host writer identity'
for name in ['repo','models']:
    assert pathlib.Path(c[name]).is_dir(), c[name]
for name in ['/dev/nvidia-caps','/dev/nvidia-caps-imex-channels']:
    assert pathlib.Path(name).is_dir(), name
for name in [c['cache'],c['run']]:
    p=pathlib.Path(name); p.mkdir(parents=True,exist_ok=True)
    assert p.stat().st_uid==c['uid'], 'Unexpected directory owner: '+name
    t=pathlib.Path(tempfile.mkdtemp(prefix='host-probe-',dir=p)); (t/'ok').write_text('ok'); (t/'ok').unlink(); t.rmdir()
    print(json.dumps({'path':name,'uid':p.stat().st_uid,'gid':p.stat().st_gid,'free_bytes':shutil.disk_usage(p).free}))
for child in ['home','tmp','ray','triton','flashinfer','torch_extensions','cuda','huggingface','pycache','xdg']:
    (pathlib.Path(c['cache'])/child).mkdir(exist_ok=True)
(pathlib.Path(c['run'])/'logs').mkdir(exist_ok=True)
for port in c['ports']:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            s.bind(('0.0.0.0',port))
        except OSError as exc:
            raise RuntimeError(f'Ray port {port} unavailable: {exc}') from exc
""".replace("CONFIG", repr(json.dumps(config)))
    _ssh(node, ["python3", "-c", code])
    exists = _ssh(node, ["docker", "container", "inspect", _name(args, rank)], capture=True, check=False)
    if exists.returncode == 0:
        raise RuntimeError(f"Container {_name(args, rank)} already exists on {node}; inspect it before retrying")
    image = _ssh(node, ["docker", "image", "inspect", args.image], capture=True, check=False)
    if image.returncode:
        _ssh(node, ["docker", "pull", args.image])
    probe = """import json,os,pathlib,tempfile
result=[]
for base in ['/cache','/run-output']:
 p=pathlib.Path(tempfile.mkdtemp(prefix='container-probe-',dir=base)); (p/'nested').mkdir(); (p/'nested'/'ok').write_text('ok')
 result.append({'path':str(p),'uid':(p/'nested'/'ok').stat().st_uid,'gid':(p/'nested'/'ok').stat().st_gid})
print(json.dumps(result))
"""
    result = _ssh(node, _docker(args, rank, probe=True) + ["python3", "-c", probe], capture=True)
    paths = json.loads(result.stdout.strip().splitlines()[-1])
    for item in paths:
        if (item["uid"], item["gid"]) != (args.uid, args.gid):
            raise RuntimeError(f"Unexpected container writer: {item}")
        source = item["path"].replace("/cache/", args.cache_dir + "/", 1).replace("/run-output/", args.run_dir + "/", 1)
        # The login view must be able to read and delete shared output; local
        # caches are deleted through the normal compute-node SSH identity.
        cleanup = "from pathlib import Path; p=Path(" + repr(source) + "); assert (p/'nested'/'ok').read_text()=='ok'; (p/'nested'/'ok').unlink(); (p/'nested').rmdir(); p.rmdir(); assert not p.exists()"
        if item["path"].startswith("/run-output/"):
            _run([sys.executable, "-c", cleanup])
        else:
            _ssh(node, ["python3", "-c", cleanup])
    _ssh(node, _docker(args, rank) + ["sleep", "infinity"])
    return _ssh(node, ["docker", "inspect", "--format", "{{.Image}}", _name(args, rank)], capture=True).stdout.strip()


def _cluster_status(args):
    code = """import json,ray
ray.init(address=ADDRESS,logging_level='ERROR')
nodes=[{'ip':n['NodeManagerAddress'],'gpus':n['Resources'].get('GPU',0)} for n in ray.nodes() if n['Alive']]
print(json.dumps({'nodes':nodes,'resources':ray.cluster_resources()},sort_keys=True))
assert sorted((n['ip'],n['gpus']) for n in nodes)==sorted(EXPECTED), 'Expected exactly two four-GPU nodes'
""".replace("ADDRESS", repr(f"{args.node_ips[0]}:{args.ray_port}")).replace("EXPECTED", repr([(ip, 4) for ip in args.node_ips]))
    return _ssh(args.nodes[0], ["docker", "exec", _name(args, 0), "timeout", "--kill-after=5s", "40s",
                               "python3", "-c", code], capture=True, check=False)


def _bootstrap(args):
    _check_allocation(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        image_ids = list(pool.map(lambda rank: _prepare_node(args, rank), range(2)))
    if len(set(image_ids)) != 1:
        raise RuntimeError(f"The two nodes resolved different images: {image_ids}")
    for rank, node in enumerate(args.nodes):
        _ssh(node, ["docker", "exec", _name(args, rank), *_ray(args, rank)])
    deadline = time.monotonic() + 120
    while True:
        result = _cluster_status(args)
        if result.returncode == 0:
            print(result.stdout, flush=True)
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(result.stderr + result.stdout)
        time.sleep(5)
    metadata = {"image": args.image, "image_id": image_ids[0], "nodes": args.nodes,
                "node_ips": args.node_ips, "job_id": args.job_id, "run_dir": args.run_dir,
                "cache_dir_per_node": args.cache_dir, "uid": args.uid, "gid": args.gid,
                "state": "ray_ready_training_not_started"}
    Path(args.run_dir, "bootstrap.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("RAY_READY_8_GPUS; training has not been started", flush=True)


def _train(args):
    _check_allocation(args)
    if args.gate_log is None:
        raise RuntimeError("train requires --gate-log from the successful eight-rank MNNVL check")
    text = args.gate_log.read_text(errors="replace")
    for marker in ["ALLREDUCE_OK", "P2P/MNNVL", "world=8"]:
        if marker not in text:
            raise RuntimeError(f"Communication log lacks {marker!r}: {args.gate_log}")
    result = _cluster_status(args)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    log_path = Path(args.run_dir, "logs", "qwen35_train.log")
    command = ["ssh", "-o", "BatchMode=yes", args.nodes[0], shlex.join(_training_command(args))]
    print("+ " + shlex.join(command), flush=True)
    with log_path.open("x") as log:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = process.wait()
    Path(args.run_dir, "train_exit.json").write_text(json.dumps({"exit_code": code, "gate_log": str(args.gate_log)}) + "\n")
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    args = _parser().parse_args()
    for value in args.node_ips:
        ip = ipaddress.ip_address(value)
        if ip.is_loopback or ip.is_unspecified:
            raise ValueError(f"Use a real compute-node IP: {value}")
    if args.action == "plan":
        for rank, node in enumerate(args.nodes):
            print(json.dumps({"node": node, "docker": _docker(args, rank) + ["sleep", "infinity"],
                              "ray_start": _ray(args, rank)}, indent=2))
        print(json.dumps({"head_train": _training_command(args)}, indent=2))
    elif args.action == "bootstrap":
        _bootstrap(args)
    elif args.action == "status":
        result = _cluster_status(args)
        print(result.stdout + result.stderr)
        raise SystemExit(result.returncode)
    else:
        _train(args)


if __name__ == "__main__":
    main()
