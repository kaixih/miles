#!/usr/bin/env bash
# Build the login-node run root that prepare_run.sh stages onto the nodes.
#
#   bash make_inputs.sh files <run root>                      # on the login node; clones from GitHub
#   bash make_inputs.sh manifest <run root> <checkpoint dir>  # where the checkpoint reads at disk speed
#
# files     <run root>/tools/   flat copy of this recipe's scripts (top level and site/)
#           <run root>/inputs/  OpenEnv.bundle and terminal-bench-2.bundle (git bundles whose HEAD is the
#                               pinned commit; the nodes clone from the bundle, not from the network),
#                               data/*.jsonl (the frozen task split), initial-snapshot-retry.patch,
#                               known-versions.json
# manifest  <run root>/inputs/model-reference-manifest.json: size and SHA256 of every file in the
#           checkpoint directory; stage_model_node.py verifies each node's copy against it. The checkpoint
#           is about 330 GB, so point this at a local copy (for example a node's cache, through srun) and,
#           if that machine cannot write to the run root, give a local path here and copy the file over.
#
# Pinned revisions, overridable through the environment variables of the same name:
#   OPENENV_REPO / OPENENV_COMMIT, TB2_REPO / TB2_COMMIT, MODEL_ID
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)   # .../site
RECIPE=$(dirname "$HERE")
OPENENV_REPO=${OPENENV_REPO:-https://github.com/meta-pytorch/OpenEnv.git}
OPENENV_COMMIT=${OPENENV_COMMIT:-38b2a31354d1d4d627894412b9d469e92f4a2c61}
TB2_REPO=${TB2_REPO:-https://github.com/laude-institute/terminal-bench-2.git}
TB2_COMMIT=${TB2_COMMIT:-2fd12b88aafdd04a52c298e3940bcb189f9766d6}
MODEL_ID=${MODEL_ID:-zai-org/GLM-5.3-Flash}

bundle() {  # bundle <repo url> <commit> <out>: a git bundle whose HEAD is exactly <commit>
  local url=$1 commit=$2 out=$3 tmp
  if [ -s "$out" ] && [ "$(git bundle list-heads "$out" HEAD | cut -d' ' -f1)" = "$commit" ]; then echo "exists: $out"; return; fi
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/make-inputs.XXXXXX")
  git clone --quiet --no-checkout "$url" "$tmp/repo"
  git -C "$tmp/repo" -c advice.detachedHead=false checkout --quiet "$commit"
  git -C "$tmp/repo" bundle create "$out.tmp" HEAD >/dev/null 2>&1
  git bundle verify "$out.tmp" >/dev/null
  mv "$out.tmp" "$out"; rm -rf "$tmp"
  echo "wrote: $out HEAD=$(git bundle list-heads "$out" HEAD | cut -d' ' -f1)"
}

MODE=${1:-}; ROOT=${2:-}
case "$MODE" in
  files)
    test -n "$ROOT" || { echo "usage: make_inputs.sh files <run root>" >&2; exit 2; }
    umask 077
    mkdir -p "$ROOT/tools" "$ROOT/inputs/data" "$ROOT/receipts"
    cp "$RECIPE"/*.py "$RECIPE"/*.sh "$HERE"/*.py "$HERE"/*.sh "$ROOT/tools/"
    cp "$RECIPE/initial-snapshot-retry.patch" "$RECIPE/known-versions.json" "$ROOT/inputs/"
    cp "$RECIPE"/data/*.jsonl "$ROOT/inputs/data/"
    bundle "$OPENENV_REPO" "$OPENENV_COMMIT" "$ROOT/inputs/OpenEnv.bundle"
    bundle "$TB2_REPO" "$TB2_COMMIT" "$ROOT/inputs/terminal-bench-2.bundle"
    echo "tools: $(ls "$ROOT/tools" | wc -l) files; inputs:"
    (cd "$ROOT/inputs" && sha256sum ./*.bundle ./*.patch ./*.json data/*.jsonl | cut -c1-16,66-)
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
