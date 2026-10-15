#!/usr/bin/env python3
"""Write run-spec.json for one allocation: the bindings every node-side tool reads.

Run on the Slurm login node after the model staging receipts exist:

    python3 make_run_spec.py --job-id <jobid> --run-id <run-id> \
        --nic <fabric interface> --model-receipts <dir with <node>.json from stage_model_node.py> \
        --out run-spec.json

Queries each allocated node for its IPv4 address on --nic through srun; --image-ref/--image-id name the pinned
container image (kept out of this file so no internal registry address is committed). Queries, orders nodes by that
address (Miles places by numeric IP order; the first is the Ray head), and records the image,
container name, node root and per-node verified model path.
"""

import argparse
import ipaddress
import json
import os
import re
import subprocess
from pathlib import Path



def read(argv):
    return subprocess.check_output(argv, text=True, timeout=120).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nic", required=True)
    parser.add_argument("--model-receipts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-ref", required=True, help="image reference by digest, e.g. <registry>/<repo>@sha256:...")
    parser.add_argument("--image-id", required=True, help="docker image id (sha256:...) every node must have")
    args = parser.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    job = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", read(["scontrol", "show", "job", "-o", args.job_id])))
    assert job["JobState"] == "RUNNING", job.get("JobState")
    hosts = read(["scontrol", "show", "hostnames", job["NodeList"]]).splitlines()
    assert len(hosts) == int(job["NumNodes"])
    lines = read(["srun", f"--jobid={args.job_id}", "--overlap", f"--nodes={len(hosts)}", f"--ntasks={len(hosts)}",
                  "--ntasks-per-node=1", "--cpu-bind=none", "bash", "-c",
                  f"echo $(hostname -s) $(ip -4 -o addr show dev {args.nic} | awk '{{split($4,a,\"/\"); print a[1]}}')"]).splitlines()
    ips = {}
    for line in lines:
        host, *addr = line.split()
        assert len(addr) == 1, f"{host}: expected one IPv4 address on {args.nic}, got {addr}"
        ips[host] = addr[0]
    assert set(ips) == set(hosts), (sorted(ips), sorted(hosts))
    nodes = []
    for host in sorted(hosts, key=lambda h: ipaddress.IPv4Address(ips[h])):
        receipt = json.loads((args.model_receipts / f"{host}.json").read_text())
        assert receipt["node"] == host and receipt["status"] in ("EXISTING_CACHE_VERIFIED", "COPIED_FROM_SOURCE_AND_VERIFIED")
        nodes.append({"node": host, "ip": ips[host], "model_path": receipt["model_path"], "model_status": receipt["status"]})
    spec = {
        "job_id": int(args.job_id), "job_name": job["JobName"], "run_id": args.run_id,
        "start": job["StartTime"], "end": job["EndTime"], "uid": os.getuid(), "gid": os.getgid(),
        "image_ref": args.image_ref, "image_id": args.image_id, "container_name": args.run_id,
        "node_root": os.environ.get("NODE_ROOT", f"/raid/dldata/{os.environ.get('USER', 'miles')}-{args.run_id}"), "nic": args.nic,
        "head_ip": nodes[0]["ip"], "nodes": nodes,
    }
    args.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
