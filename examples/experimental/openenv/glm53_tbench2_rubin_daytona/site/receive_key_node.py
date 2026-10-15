#!/usr/bin/env python3
"""Write one credential from stdin into the head container's private key file; never locate a key.

Run on the head node as the ordinary UID through the allocation, e.g. from the login node:

    srun --jobid=<job> --overlap --nodes=1 --ntasks=1 -w <head> \
        python3 receive_key_node.py --spec run-spec.json --execute < /path/to/staged/key

Without --execute it verifies the destination identity and reads no stdin. With --execute it pipes
stdin into `docker exec -i` where a receiver creates /work/private/daytona.key exclusively, 0600.
The credential never appears in argv, the environment, logs or this file.
"""

import argparse
import json
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path

RECEIVER = r'''
import json, os, stat, sys
from pathlib import Path
if not __debug__:
    raise RuntimeError("assertions must stay enabled")
uid, gid = int(sys.argv[1]), int(sys.argv[2])
assert (os.getuid(), os.getgid()) == (uid, gid)
private = Path("/work/private")
assert private.resolve(strict=True) == private
info = private.lstat()
assert stat.S_ISDIR(info.st_mode) and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (uid, gid, 0o700)
assert not sys.stdin.isatty()
key = sys.stdin.buffer.read(4097).rstrip(b"\r\n")
assert 0 < len(key) < 4096 and all(33 <= b <= 126 for b in key), "invalid single-line credential input"
dir_fd = os.open(private, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    fd = os.open("daytona.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    with os.fdopen(fd, "wb") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(key + b"\n")
        out.flush()
        os.fsync(out.fileno())
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
del key
info = (private / "daytona.key").lstat()
assert stat.S_ISREG(info.st_mode) and (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (uid, gid, 0o600)
print(json.dumps({"status": "PRIVATE_CREDENTIAL_CREATED", "path": "/work/private/daytona.key", "mode": "0600",
                  "uid": uid, "gid": gid, "bytes": info.st_size, "credential_logged": False}))
'''


def read(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def main():
    if not __debug__:
        raise RuntimeError("the safety checks are assertions; run without -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    uid, gid, job = spec["uid"], spec["gid"], str(spec["job_id"])
    head = next(n["node"] for n in spec["nodes"] if n["ip"] == spec["head_ip"])
    assert (os.getuid(), os.getgid()) == (uid, gid)
    assert os.environ.get("SLURM_JOB_ID") == job and socket.gethostname().split(".")[0] == head
    info = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", read(["scontrol", "show", "job", "-o", job])))
    assert info["JobState"] == "RUNNING" and info["JobName"] == spec["job_name"]
    receipt = json.loads((Path(spec["node_root"]) / "bootstrap-node.json").read_text())
    assert receipt["node"] == head and receipt["status"] == "idle_container_ready_no_install_no_training"
    cid = receipt["container_id"]
    fmt = '{"id":{{json .Id}},"name":{{json .Name}},"image_id":{{json .Image}},"user":{{json .Config.User}},"running":{{json .State.Running}},"mounts":{{json .Mounts}}}'
    container = json.loads(read(["docker", "inspect", "--format", fmt, cid]))
    assert (container["id"], container["name"], container["image_id"], container["user"], container["running"]) == (
        cid, "/" + spec["container_name"], spec["image_id"], f"{uid}:{gid}", True)
    assert any(m["Destination"] == "/work" and m["Source"] == spec["node_root"] + "/work" and m["RW"] for m in container["mounts"])
    private = Path(spec["node_root"]) / "work/private"
    meta = private.lstat()
    assert stat.S_ISDIR(meta.st_mode) and (meta.st_uid, meta.st_gid, stat.S_IMODE(meta.st_mode)) == (uid, gid, 0o700)
    assert not os.path.lexists(private / "daytona.key"), "refusing to overwrite an existing credential"
    if not args.execute:
        print(json.dumps({"status": "READ_ONLY_IDENTITY_VERIFIED_STDIN_NOT_READ", "node": head, "container_id": cid}))
        return
    assert not sys.stdin.isatty(), "pipe the credential; no interactive entry"
    result = subprocess.run(["docker", "exec", "-i", f"--user={uid}:{gid}", cid, "python3", "-B", "-c", RECEIVER, str(uid), str(gid)],
                            stdin=sys.stdin.buffer, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, f"receiver failed: {result.stderr.decode(errors='replace')[-400:]}"
    out = json.loads(result.stdout)
    out.update(node=head, container_id=cid)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
