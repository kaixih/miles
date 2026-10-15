#!/usr/bin/env python3
"""CPU-only check that every node of the run registered with Ray and advertises 4 GPUs.

Run inside the head container as the ordinary UID after the per-node `rubin_node.py` starters
are up:  python3 ray_ready.py --spec /work/tools/run-spec.json --output /work/ray-ready-<n>.json
Loads no weights, uses no CUDA, starts or stops nothing.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=0)
def inspect_node(expected):
    return {"node": socket.gethostname(), "expected": expected["node"], "uid": os.getuid(), "gid": os.getgid(),
            "gloo_ifname": os.environ.get("GLOO_SOCKET_IFNAME"), "nccl_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            "external_ip": os.environ.get("MILES_NODE_EXTERNAL_IP"), "ray_tmpdir": os.environ.get("RAY_TMPDIR"),
            "node_id": ray.get_runtime_context().get_node_id()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert not args.output.exists()
    expected_ips = {n["ip"]: n for n in spec["nodes"]}
    want = len(expected_ips)
    # Only the driver copy is cleared: workers inherit their own node's Ray daemon NIC settings.
    os.environ.pop("NCCL_SOCKET_IFNAME", None)
    os.environ.pop("GLOO_SOCKET_IFNAME", None)
    result = {"ready": False, "job_id": spec["job_id"], "head_ip": spec["head_ip"], "expected_nodes": want}
    try:
        ray.init(address=f"{spec['head_ip']}:6379", log_to_driver=False, runtime_env={"env_vars": {}})
        deadline = time.time() + args.timeout
        while True:
            alive = [n for n in ray.nodes() if n["Alive"]]
            ips = {n["NodeManagerAddress"] for n in alive}
            assert ips <= set(expected_ips), f"unexpected Ray members: {ips - set(expected_ips)}"
            if len(alive) == want:
                break
            if time.time() > deadline:
                raise TimeoutError(f"{len(alive)}/{want} Ray nodes registered")
            print(f"RAY_WAIT nodes={len(alive)}/{want}", flush=True)
            time.sleep(5)
        assert len(ips) == len(alive), f"duplicate Ray members share an IP: {sorted(n['NodeManagerAddress'] for n in alive)}"
        tasks, resources = [], {}
        for ray_node in alive:
            ip = ray_node["NodeManagerAddress"]
            resources[ip] = ray_node["Resources"].get("GPU")
            assert resources[ip] == 4, f"{ip} advertises GPU={resources[ip]}"
            tasks.append(inspect_node.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray_node["NodeID"], soft=False), runtime_env={"env_vars": {}}).remote(expected_ips[ip]))
        records = ray.get(tasks, timeout=120)
        for record in records:
            assert record["node"] == record["expected"] and record["uid"] == spec["uid"], record
            assert record["external_ip"] == expected_ips[record["external_ip"]]["ip"], record
            assert record["gloo_ifname"] and record["nccl_ifname"] == "=" + record["gloo_ifname"], record
        result.update(ready=True, nodes=records, gpu_per_node=resources, total_gpus=sum(resources.values()),
                      cluster_gpus=ray.cluster_resources().get("GPU"), dashboard=f"http://{spec['head_ip']}:8265")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("RAY_READY=" + json.dumps(result, sort_keys=True), flush=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
