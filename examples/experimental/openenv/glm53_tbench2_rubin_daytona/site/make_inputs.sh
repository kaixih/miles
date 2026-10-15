#!/usr/bin/env bash
# Build the login-node run root that prepare_run.sh stages onto the nodes.
#
#   bash make_inputs.sh files <run root>                      # login node
#   bash make_inputs.sh manifest <run root> <checkpoint dir>  # where the checkpoint is on local disk
#
# files     <run root>/tools/   flat copy of this recipe's scripts (top level and site/)
#           <run root>/inputs/  data/*.jsonl (the frozen task split), initial-snapshot-retry.patch,
#                               known-versions.json
# manifest  <run root>/inputs/model-reference-manifest.json: size and SHA256 of every file in the
#           checkpoint directory; stage_model_node.py verifies each node's copy against it. The checkpoint
#           is about 330 GB, so point this at a local copy (for example a node's cache, through srun) and,
#           if that machine cannot write to the run root, give a local path here and copy the file over.
#
# The run root is created world-readable: the nodes read it through a read-only CIFS view under a
# service identity (prepare_run.sh stage-node), and it holds no secrets (the key goes through srun).
# The OpenEnv and Terminal-Bench-2 checkouts are cloned on the nodes by prepare_run.sh stage-tools.
# MODEL_ID (default zai-org/GLM-5.3-Flash) is recorded in the manifest.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)   # .../site
RECIPE=$(dirname "$HERE")
MODEL_ID=${MODEL_ID:-zai-org/GLM-5.3-Flash}

MODE=${1:-}; ROOT=${2:-}
case "$MODE" in
  files)
    test -n "$ROOT" || { echo "usage: make_inputs.sh files <run root>" >&2; exit 2; }
    umask 022
    mkdir -p "$ROOT/tools" "$ROOT/inputs/data" "$ROOT/receipts"
    cp "$RECIPE"/*.py "$RECIPE"/*.sh "$HERE"/*.py "$HERE"/*.sh "$ROOT/tools/"
    cp "$RECIPE/initial-snapshot-retry.patch" "$RECIPE/known-versions.json" "$ROOT/inputs/"
    cp "$RECIPE"/data/*.jsonl "$ROOT/inputs/data/"
    echo "tools: $(ls "$ROOT/tools" | wc -l) files; inputs:"
    (cd "$ROOT/inputs" && sha256sum ./*.patch ./*.json data/*.jsonl | cut -c1-16,66-)
    test -s "$ROOT/inputs/model-reference-manifest.json" || echo "next: bash make_inputs.sh manifest $ROOT <checkpoint dir>"
    ;;
  manifest)
    CKPT=${3:-}
    { test -n "$ROOT" && test -d "$CKPT"; } || { echo "usage: make_inputs.sh manifest <run root> <checkpoint dir>" >&2; exit 2; }
    mkdir -p "$ROOT/inputs"
    python3 - "$CKPT" "$ROOT/inputs/model-reference-manifest.json" "$MODEL_ID" <<'PY'
import concurrent.futures, hashlib, json, os, sys
src, out, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
names = sorted(os.listdir(src))
bad = [n for n in names if os.path.islink(os.path.join(src, n)) or not os.path.isfile(os.path.join(src, n))]
if bad:
    sys.exit(f"the checkpoint directory must hold regular files only (stage_model_node.py compares the whole directory); found: {bad}")

def one(name):
    digest = hashlib.sha256()
    with open(os.path.join(src, name), "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return name, {"size": os.path.getsize(os.path.join(src, name)), "sha256": digest.hexdigest()}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    files = dict(pool.map(one, names))
manifest = {"model_id": model_id, "revision": os.path.basename(os.path.normpath(src)),
            "total_bytes": sum(v["size"] for v in files.values()), "files": files}
with open(out, "w") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(f"wrote: {out} files={len(files)} total_bytes={manifest['total_bytes']}")
PY
    ;;
  *) echo "usage: make_inputs.sh files <run root> | manifest <run root> <checkpoint dir>" >&2; exit 2 ;;
esac
