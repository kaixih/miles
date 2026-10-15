#!/usr/bin/env python3
"""Provide a fully verified node-local copy of the original GLM-5.3-Flash HF checkpoint.

Run on the node as the ordinary UID. Looks for an existing node-local cache whose file set,
sizes and SHA256 all match the reference manifest; if none exists, copies the read-only
source (--source, or MODEL_SOURCE) into <dest> (a reusable cache location outside any run root) while hashing, then
verifies; a failed copy is removed. Never modifies or deletes an existing cache.

  python3 stage_model_node.py --manifest /path/model-reference-manifest.json --source <read-only checkpoint dir> \
      --dest /raid/dldata/<user>-models/GLM-5.3-Flash/hf-3f1971b_orig --receipt /path/receipt.json
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path

CHUNK = 16 * 1024 * 1024


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_dir(root, files, workers):
    def one(name):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != files[name]["size"]:
            return name, False
        return name, sha256_of(path) == files[name]["sha256"]

    if {p.name for p in root.iterdir()} != set(files):
        return False, ["file set differs"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        bad = [name for name, ok in pool.map(one, sorted(files)) if not ok]
    return not bad, bad


def copy_hashed(src, dst):
    digest = hashlib.sha256()
    with src.open("rb") as reader, dst.open("xb") as writer:
        for block in iter(lambda: reader.read(CHUNK), b""):
            digest.update(block)
            writer.write(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=os.environ.get("MODEL_SOURCE") or None,
                        help="read-only directory holding the original checkpoint (default: $MODEL_SOURCE)")
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate", action="append", default=[], help="existing cache dir to try first")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = manifest["files"]
    node = socket.gethostname().split(".")[0]
    started = time.monotonic()
    receipt = {"node": node, "uid": os.getuid(), "gid": os.getgid(), "revision": manifest["revision"],
               "total_bytes": manifest["total_bytes"], "file_count": len(files), "candidates": []}

    candidates = [Path(c) for c in args.candidate] + sorted(Path("/raid/dldata").glob("*/inputs/GLM-5.3-Flash"))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_dir() or candidate.stat().st_uid != os.getuid():
            continue
        config = candidate / "config.json"
        if not config.is_file() or sha256_of(config) != files["config.json"]["sha256"]:
            receipt["candidates"].append({"path": str(candidate), "result": "config differs"})
            continue
        ok, bad = verify_dir(candidate, files, args.workers)
        receipt["candidates"].append({"path": str(candidate), "result": "FULL_SHA_MATCH" if ok else f"mismatch: {bad[:5]}"})
        if ok:
            receipt.update(status="EXISTING_CACHE_VERIFIED", model_path=str(candidate), elapsed_s=round(time.monotonic() - started, 1))
            break
    else:
        if args.source is None or not (args.source / "config.json").is_file():
            raise RuntimeError(f"no verified cache and the source {args.source} is not readable on this node")
        if args.dest.exists():
            raise RuntimeError(f"{args.dest} exists but did not verify; refusing to overwrite")
        partial = args.dest.with_name(args.dest.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)  # leftover of an earlier failed copy of this same destination
        os.umask(0o022)
        partial.mkdir(parents=True)
        free = shutil.disk_usage(partial).free
        if free < manifest["total_bytes"] + 50 * 1024**3:
            raise RuntimeError(f"insufficient space under {partial}: {free} bytes free")

        def copy_one(name):
            digest = copy_hashed(args.source / name, partial / name)
            if digest != files[name]["sha256"] or (partial / name).stat().st_size != files[name]["size"]:
                raise RuntimeError(f"copied {name} does not match the manifest")
            return name

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for _ in pool.map(copy_one, sorted(files, key=lambda n: -files[n]["size"])):
                    pass
            ok, bad = verify_dir(partial, files, args.workers)
            if not ok:
                raise RuntimeError(f"post-copy verification failed: {bad}")
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        partial.rename(args.dest)
        receipt.update(status="COPIED_FROM_SOURCE_AND_VERIFIED", model_path=str(args.dest), source=str(args.source),
                       elapsed_s=round(time.monotonic() - started, 1))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("node", "status", "model_path", "elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()
