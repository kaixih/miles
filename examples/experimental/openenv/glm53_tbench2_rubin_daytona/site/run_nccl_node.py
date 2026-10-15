#!/usr/bin/env python3
"""Run this node's share of the NCCL smoke inside its run container (one torchrun, 4 GPUs).

Dispatch on every node of the allocation at the same time, as the ordinary UID:

    python3 run_nccl_node.py --spec run-spec.json --probe nccl-smoke-<attempt>

Reads the node's bootstrap receipt for the container ID, writes NCCL debug logs plus per-rank
records under <node_root>/work/<probe>, and exits non-zero unless torchrun exited 0 with no GPU
process left behind. No container creation, credentials, cleanup or automatic retry.
"""

import argparse
import json
import os
import re
import socket
import subprocess
from pathlib import Path


def read(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--port", type=int, default=29611)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    uid, gid, job = spec["uid"], spec["gid"], str(spec["job_id"])
    node = socket.gethostname().split(".")[0]
    order = [n["node"] for n in spec["nodes"]]  # spec order == numeric IP order == Miles placement order
    assert (os.getuid(), os.getgid()) == (uid, gid) and os.environ.get("SLURM_JOB_ID") == job and node in order
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", args.probe)
    root = Path(spec["node_root"])
    receipt = json.loads((root / "bootstrap-node.json").read_text())
    assert receipt["node"] == node and receipt["status"] == "idle_container_ready_no_install_no_training"
    cid = receipt["container_id"]
    assert read(["docker", "inspect", "--format", "{{.State.Running}} {{.Image}}", cid]) == f"true {spec['image_id']}"
    assert not read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]), "GPU busy"
    node_rank = order.index(node)
    assert spec["nodes"][0]["ip"] == spec["head_ip"], "torchrun's static rendezvous needs node_rank 0 to be the master_addr node"
    nic = receipt["nic"]
    worker = root / "work/tools/nccl_smoke.py"
    assert worker.is_file(), f"{worker} missing; run the stage-tools step first"

    os.umask(0o077)
    output = root / "work" / args.probe
    output.mkdir(mode=0o700)  # exclusive: a repeated attempt gets a new probe name
    env = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P,NET,ENV",
           "NCCL_DEBUG_FILE": f"/work/{args.probe}/nccl.%h.%p.log", "NCCL_CUMEM_ENABLE": "1",
           "NCCL_SOCKET_IFNAME": "=" + nic, "GLOO_SOCKET_IFNAME": nic, "OMP_NUM_THREADS": "1",
           "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "SMOKE_OUTPUT_DIR": f"/work/{args.probe}"}
    command = ["docker", "exec", f"--user={uid}:{gid}", "--workdir=/work"]
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [cid, "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", "300s",
                "/opt/sglang/bin/python3", "-B", "-m", "torch.distributed.run",
                f"--nnodes={len(order)}", "--nproc_per_node=4", f"--node_rank={node_rank}",
                f"--master_addr={spec['head_ip']}", f"--master_port={args.port}", "--max_restarts=0",
                "/work/tools/nccl_smoke.py"]
    result = {"node": node, "node_rank": node_rank, "container_id": cid, "nic": nic, "argv": command}
    (output / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "torchrun.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    result["torchrun_exit"] = completed.returncode
    result["remaining_gpu_processes"] = read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    result["normal_exit"] = completed.returncode == 0 and not result["remaining_gpu_processes"]
    logs = list(output.glob("nccl.*.log"))
    text = "".join(p.read_text(errors="replace") for p in logs)
    result["nccl_log_files"] = len(logs)
    mentions = {k: text.count(k) for k in ("MNNVL", "NVLS", "via P2P/CUMEM", "via P2P/", "via NET/", "via SHM")}
    result["transport_mentions"] = mentions
    # Cross-node traffic over the NVLink fabric shows up as P2P (CUMEM/MNNVL) channels; any NET channel means
    # the socket/IB path carried data instead, which the handoff's MNNVL check must not accept silently.
    result["transport_ok"] = mentions["via NET/"] == 0 and (mentions["MNNVL"] > 0 or mentions["via P2P/"] > 0)
    (output / "node-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("node", "normal_exit", "torchrun_exit", "transport_ok", "transport_mentions")}), flush=True)
    raise SystemExit(0 if result["normal_exit"] and result["transport_ok"] else 1)


if __name__ == "__main__":
    main()
