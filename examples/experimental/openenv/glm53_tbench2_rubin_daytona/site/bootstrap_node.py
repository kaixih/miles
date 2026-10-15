#!/usr/bin/env python3
"""Create this node's idle run container for one Slurm allocation.

Run once per node as the ordinary UID, on the node itself:

    python3 bootstrap_node.py --spec /path/to/run-spec.json

Checks the allocation, the pinned image, GPU idleness, the IMEX channel owned by the
run UID, the route to the head node, the model metadata and a fresh node root, then
starts one container whose lifetime is bounded by the allocation end. Pulls nothing,
installs nothing, starts no Ray or training, and refuses to replace a same-name container.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["TZ"] = "UTC"  # scontrol prints cluster-local times; the cluster runs UTC and we parse EndTime as UTC

INSPECT = (
    '{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Config.Image}},'
    '"image_id":{{json .Image}},"user":{{json .Config.User}},'
    '"pid_mode":{{json .HostConfig.PidMode}},"network":{{json .HostConfig.NetworkMode}},'
    '"ipc":{{json .HostConfig.IpcMode}},"privileged":{{json .HostConfig.Privileged}},'
    '"ulimits":{{json .HostConfig.Ulimits}},"devices":{{json .HostConfig.Devices}},'
    '"mounts":{{json .Mounts}},"running":{{json .State.Running}}}'
)
MODEL_METADATA = {
    "config.json": "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
    "model.safetensors.index.json": "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(argv):
    return subprocess.check_output(argv, text=True, timeout=60).strip()


def imex_device(uid, gid):
    """The one IMEX channel device this UID owns; sites assign one channel per allocation."""
    owned = []
    for path in sorted(Path("/dev/nvidia-caps-imex-channels").glob("channel*")):
        info = path.lstat()
        if stat.S_ISCHR(info.st_mode) and (info.st_uid, info.st_gid) == (uid, gid):
            owned.append(path)
    require(len(owned) == 1, f"expected exactly one IMEX channel owned by {uid}:{gid}, found {owned}")
    device = owned[0]
    info = device.lstat()
    require(stat.S_IMODE(info.st_mode) == 0o600 and os.access(device, os.R_OK | os.W_OK), f"{device}: not 0600 rw")
    return device, {
        "path": str(device), "uid": info.st_uid, "gid": info.st_gid,
        "mode": oct(stat.S_IMODE(info.st_mode)), "major": os.major(info.st_rdev), "minor": os.minor(info.st_rdev),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    uid, gid = spec["uid"], spec["gid"]
    job = str(spec["job_id"])
    node = socket.gethostname().split(".")[0]
    nodes = {entry["node"]: entry["ip"] for entry in spec["nodes"]}
    model_paths = {entry["node"]: entry["model_path"] for entry in spec["nodes"]}

    require((os.getuid(), os.getgid()) == (uid, gid), f"run as {uid}:{gid}")
    require(os.environ.get("SLURM_JOB_ID") == job, "wrong or missing SLURM_JOB_ID")
    require(node in nodes, f"{node} is not in the run spec")
    require(os.uname().machine == "aarch64", "expected an arm64 node")
    info = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", read(["scontrol", "show", "job", "-o", job])))
    require(info.get("JobState") == "RUNNING" and info.get("JobName") == spec["job_name"], "allocation differs")
    require(set(read(["scontrol", "show", "hostnames", info["NodeList"]]).splitlines()) == set(nodes), "node set differs")
    deadline = datetime.fromisoformat(info["EndTime"]).replace(tzinfo=timezone.utc)
    require((deadline - datetime.now(timezone.utc)).total_seconds() > 600, "allocation nearly over")

    name = spec["container_name"]
    existing = read(["docker", "ps", "-aq", "--no-trunc", "--filter", f"name=^/{name}$"])
    if existing:
        for cid in existing.splitlines():
            print(read(["docker", "inspect", "--format", INSPECT, cid]), flush=True)
        raise RuntimeError("same-name container exists; preserved, refusing to replace it")
    image = json.loads(read(["docker", "image", "inspect", "--format",
                            '{"id":{{json .Id}},"os":{{json .Os}},"arch":{{json .Architecture}}}', spec["image_ref"]]))
    require(image == {"id": spec["image_id"], "os": "linux", "arch": "arm64"}, f"image identity differs: {image}")
    require(not read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]), "GPU processes present")
    gpus = read(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines()
    require(len(gpus) == 4, f"expected four GPUs, saw {len(gpus)}")
    device, device_info = imex_device(uid, gid)

    own_ip = nodes[node]
    peers = [ip for ip in nodes.values() if ip != own_ip]
    if peers:
        peer_ip = spec["head_ip"] if own_ip != spec["head_ip"] else peers[0]
        (route,) = json.loads(read(["ip", "-j", "-4", "route", "get", peer_ip]))
        nic = route["dev"]
        require(nic != "lo" and route.get("prefsrc") == own_ip, f"route to {peer_ip} is not via this node's run IP: {route}")
        if nic != spec["nic"]:
            print(f"note: spec names NIC {spec['nic']} but this node routes to its peer via {nic}; using {nic}", flush=True)
    else:  # single-node allocation: no peer to route to; the spec's interface must carry this node's run IP
        nic = spec["nic"]
        route = {"single_node": True, "dev": nic}
        addrs = json.loads(read(["ip", "-j", "-4", "addr", "show", "dev", nic]))
        require(any(a.get("local") == own_ip for i in addrs for a in i.get("addr_info", [])), f"{nic} does not carry {own_ip}")

    model = Path(model_paths[node])  # per node: each node verified its own cache (stage_model_node.py receipt)
    require(model.is_absolute() and model.resolve(strict=True) == model and model.is_dir(), "model path must be canonical")
    require("," not in str(model), "model path cannot contain a Docker mount delimiter")
    for filename, expected in MODEL_METADATA.items():
        require(hashlib.sha256((model / filename).read_bytes()).hexdigest() == expected, f"model metadata: {filename}")

    root = Path(spec["node_root"])
    require(root.parent.resolve(strict=True) == root.parent and not root.is_symlink(), "unsafe node root")
    if root.exists():
        require(root.is_dir() and root.stat().st_uid == uid and not any(root.iterdir()), "node root exists and is not empty")
    free = shutil.disk_usage(root.parent).free
    require(free >= spec.get("min_free_bytes", 2 * 1024**4), f"{root.parent}: {free} bytes free, below the run's minimum")
    os.umask(0o077)
    root.mkdir(mode=0o700, exist_ok=True)
    for child in ("work", "tmp", "work/home", "work/private"):
        (root / child).mkdir(mode=0o700)

    receipt_path = root / "bootstrap-node.json"
    receipt = {
        "job_id": int(job), "run_id": spec["run_id"], "node": node, "end_utc": info["EndTime"] + "Z",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared", "node_root": str(root),
        "image": image, "gpu_inventory": gpus, "imex_host": device_info, "route": route, "nic": nic,
        "model_path": str(model), "model_metadata_sha256": MODEL_METADATA,
        "model_weight_verification": "separate stage_model_node.py receipt",
        "filesystem": read(["findmnt", "-T", str(root), "-n", "-o", "TARGET,SOURCE,FSTYPE"]),
    }
    with receipt_path.open("x") as output:
        json.dump(receipt, output, indent=2)
    try:
        env = {
            "HOME": "/work/home", "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "SLURM_JOB_ID": job, "MASTER_ADDR": spec["head_ip"], "MILES_NODE_EXTERNAL_IP": own_ip,
            "NCCL_SOCKET_IFNAME": "=" + nic, "GLOO_SOCKET_IFNAME": nic, "NCCL_CUMEM_ENABLE": "1",
        }
        remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
        argv = [
            "docker", "run", "-d", f"--name={name}", f"--hostname={node}", "--pull=never", "--restart=no",
            f"--user={uid}:{gid}", "--gpus=all", "--network=host", "--ipc=host", "--workdir=/opt/miles",
            "--ulimit=memlock=-1:-1", "--ulimit=stack=67108864:67108864", "--ulimit=nofile=65536:524288",
            f"--label=miles.run={spec['run_id']}", f"--device={device}:{device}:rwm",
            f"--mount=type=bind,src={root / 'work'},dst=/work", f"--mount=type=bind,src={root / 'tmp'},dst=/tmp",
            f"--mount=type=bind,src={model},dst=/inputs/GLM-5.3-Flash,readonly",
            *[f"--env={key}={value}" for key, value in env.items()],
            "--entrypoint=/bin/sleep", spec["image_ref"], str(remaining),
        ]
        receipt["docker_argv"] = argv
        cid = read(argv)
        require(re.fullmatch(r"[0-9a-f]{64}", cid), "docker did not return one container ID")
        receipt["container_id"] = cid
        receipt["status"] = "container_created"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        inspection = json.loads(read(["docker", "inspect", "--format", INSPECT, cid]))
        receipt["container"] = inspection
        require(inspection["id"] == cid and inspection["image_id"] == spec["image_id"], "created container image differs")
        require(inspection["user"] == f"{uid}:{gid}" and inspection["pid_mode"] == "" and inspection["running"], "user/pid/running differ")
        require(inspection["network"] == "host" and inspection["ipc"] == "host" and not inspection["privileged"], "network/ipc/privilege differ")
        mounts = {m["Destination"]: (m["Source"], m["RW"]) for m in inspection["mounts"]}
        require(mounts == {"/work": (str(root / "work"), True), "/tmp": (str(root / "tmp"), True),
                           "/inputs/GLM-5.3-Flash": (str(model), False)}, f"mounts differ: {mounts}")
        require(any(u["Name"] == "nofile" and u["Soft"] == 65536 and u["Hard"] == 524288 for u in inspection["ulimits"]), "nofile differs")

        probes = []
        receipt["permission_probes"] = probes
        for host_base, container_base in ((root / "work", "/work"), (root / "tmp", "/tmp")):
            probe = Path(tempfile.mkdtemp(prefix=".bootstrap-probe-", dir=host_base))
            payload = f"{spec['run_id']}-{node}"
            code = (
                "import json,os,stat;from pathlib import Path;"
                f"assert (os.getuid(),os.getgid())==({uid},{gid});"
                f"d=Path({str(device)!r});s=d.lstat();assert stat.S_ISCHR(s.st_mode);"
                "fd=os.open(d,os.O_RDWR);os.close(fd);"
                f"p=Path({container_base!r})/{probe.name!r};(p/'nested').mkdir(mode=0o700);"
                f"(p/'nested/value').write_text({payload!r});"
                "print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'imex_open_rdwr':'passed'}))"
            )
            user = json.loads(read(["docker", "exec", f"--user={uid}:{gid}", cid, "python3", "-B", "-c", code]))
            value = probe / "nested/value"
            require((value.stat().st_uid, value.stat().st_gid) == (uid, gid) and value.read_text() == payload, "probe failed")
            value.unlink()
            value.parent.rmdir()
            probe.rmdir()
            probes.append({"host_base": str(host_base), "container_base": container_base, "container_user": user,
                           "host_read": "passed", "host_delete": "passed"})
        receipt["status"] = "idle_container_ready_no_install_no_training"
    except Exception as error:
        receipt["status"] = "failed_preserved_for_inspection"
        receipt["error"] = str(error)
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: receipt[k] for k in ("node", "status", "node_root", "nic") if k in receipt}
                         | {"container_id": receipt.get("container_id"), "error": receipt.get("error")}), flush=True)


if __name__ == "__main__":
    main()
