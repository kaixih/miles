"""Per-node Ray bring-up inside an existing Slurm/Docker allocation.

Run inside each prepared container as the normal user. This helper blocks for
the lifetime of that node's Ray daemon; the parent Slurm step must stay alive.
It does not submit training, stop another Ray cluster, or allocate resources.

    python3 /work/tools/rubin_node.py --head-ip <head fabric IP> --interface <fabric NIC>

The interface must carry the peer-reachable NVL72 compute-fabric IP on every
node. The head uses the first address in numeric IP order, matching Miles's
worker-placement convention.
"""

import argparse
import fcntl
import os
import socket
import struct
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--interface", required=True)
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        address = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", args.interface.encode()))
        node_ip = socket.inet_ntoa(address[20:24])

    os.environ.update(
        GLOO_SOCKET_IFNAME=args.interface,
        NCCL_SOCKET_IFNAME="=" + args.interface,
        NCCL_CUMEM_ENABLE="1",
        RAY_TMPDIR="/work/ray",
        RAY_USAGE_STATS_ENABLED="0",
        RAY_DEDUP_LOGS="0",
        MASTER_ADDR=args.head_ip,
        MILES_NODE_EXTERNAL_IP=node_ip,
    )
    command = ["ray", "start", "--node-ip-address=" + node_ip, "--num-gpus=4", "--disable-usage-stats", "--block"]
    if node_ip == args.head_ip:
        command += ["--head", "--port=6379", "--dashboard-host=" + node_ip, "--dashboard-port=8265"]
    else:
        deadline = time.monotonic() + 300
        while True:
            try:
                with socket.create_connection((args.head_ip, 6379), timeout=3):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Ray head did not become reachable within 300 seconds") from None
                time.sleep(2)
        command += ["--address=" + args.head_ip + ":6379"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
