#!/usr/bin/env bash
# Login-node side orchestration for one allocation, one explicit step per invocation.
# Site settings come from the environment:
#   SCRATCH_ROOT  durable experiment root on the login node (run root = $SCRATCH_ROOT/<run id>)
#   RUN_PREFIX    run id prefix; run id = <RUN_PREFIX>-j<jobid> (default glm53-tb2-daytona)
#   KEY_STAGING   0600 file holding the Daytona key on the login node (step `key` only)
#   IMAGE_REF / IMAGE_ID   pinned container image reference and image id (step `spec`)
#   MODEL_SOURCE  read-only directory holding the original HF checkpoint (step `stage-model`)
#   MODEL_CACHE / KNOWN_GOOD_CACHE   node-local checkpoint cache to create / an existing verified cache to reuse
#   NODE_ROOT     node-local run root (default /raid/dldata/$USER-<run id>); CIFS_VIEW optional compute-node view of SCRATCH_ROOT
#
#   bash prepare_run.sh <step> <jobid> [step args]
#
#   stage-node   tar tools+inputs from the run's scratch root and pipe them through srun stdin into a
#                node-local staging dir on every node (compute nodes need not mount the login node's scratch)
#   stage-model  stage_model_node.py on every node (parallel); receipts fetched to $R/receipts/model/<node>.json
#   spec <nic>   make_run_spec.py on the login node from the model receipts, then push run-spec.json to every node
#   bootstrap    bootstrap_node.py on every node (fresh node root + idle container)
#   stage-tools  copy tools/inputs/data into each node root and clone the pinned OpenEnv/TB2 checkouts there
#   install      install_deps.py plan in every container (root); prints each plan SHA for review
#   apply [sha]  install_deps.py apply (root) in every container, then verify as the run UID; the optional
#                sha256 of the reviewed planned_versions JSON skips any container whose plan differs
#   key          pipe the login-node staging key file into the head container's /work/private/daytona.key
#   nccl [probe] run_nccl_node.py on every node at once
#   ray-ready    ray_ready.py inside the head container (Ray daemons must already run)
#   fetch <node> <path> <name>   copy one node file into $R/receipts/<name>
#
# Every step runs as the ordinary user through `srun --jobid=<jobid> --overlap`; nothing here cancels Slurm,
# removes containers or touches directories of other runs.
set -euo pipefail
STEP=$1; JOB=$2; shift 2
RUN_ID=${RUN_PREFIX:-glm53-tb2-daytona}-j${JOB}
R=${SCRATCH_ROOT:?set SCRATCH_ROOT to the durable experiment root on the login node}/$RUN_ID
STAGE=/raid/dldata/${USER}-${RUN_ID}-stage
SPEC_LOCAL=$R/run-spec.json
SPEC=$STAGE/run-spec.json
MODEL_CACHE=${MODEL_CACHE:-/raid/dldata/${USER}-models/GLM-5.3-Flash/hf-3f1971b_orig}
KNOWN_GOOD_CACHE=${KNOWN_GOOD_CACHE:-$MODEL_CACHE}
KEY_STAGING=${KEY_STAGING:?set KEY_STAGING to the 0600 key file on the login node}

hosts() { scontrol show hostnames "$(scontrol show job -o "$JOB" | tr ' ' '\n' | grep '^NodeList=' | cut -d= -f2)"; }
on_node() { local node=$1; shift; srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpu-bind=none --chdir=/tmp -w "$node" "$@"; }
spec_get() { python3 -c "import json; print(json.load(open('$SPEC_LOCAL'))$1)"; }
node_root() { spec_get "['node_root']"; }
head_node() { spec_get "['nodes'][0]['node']"; }
cid_of() { on_node "$1" cat "$(node_root)/bootstrap-node.json" | python3 -c "import json,sys; print(json.load(sys.stdin)['container_id'])"; }
mkdir -p "$R/receipts"

case "$STEP" in
  stage-node)
    # Compute nodes may see the scratch root through a CIFS view; copy from there into node-local staging.
    # Falls back to piping a tarball through srun stdin when the CIFS view cannot read the run root.
    CIFS_R=${CIFS_VIEW:-/nonexistent}/$RUN_ID
    VERIFY="echo tools=\$(ls $STAGE/tools | wc -l) inputs=\$(ls $STAGE/inputs | wc -l); sha256sum $STAGE/inputs/OpenEnv.bundle $STAGE/inputs/terminal-bench-2.bundle $STAGE/inputs/data/*.jsonl | cut -c1-16,66-"
    for n in $(hosts); do
      echo "== $n"
      if on_node "$n" test -r "$CIFS_R/tools/known-versions.json"; then
        on_node "$n" bash -c "set -e; umask 077; test ! -e $STAGE; mkdir -p $STAGE/receipts; cp -r $CIFS_R/tools $CIFS_R/inputs $STAGE/; echo via=cifs; $VERIFY"
      else
        tar -C "$R" -czf - tools inputs | on_node "$n" bash -c "set -e; umask 077; test ! -e $STAGE; mkdir -p $STAGE/receipts; tar -xzf - -C $STAGE; echo via=srun-stdin; $VERIFY"
      fi
    done ;;
  stage-model)
    mkdir -p "$R/receipts/model"
    for n in $(hosts); do
      on_node "$n" python3 -B "$STAGE/tools/stage_model_node.py" --manifest "$STAGE/inputs/model-reference-manifest.json" \
        --source "${MODEL_SOURCE:?set MODEL_SOURCE to the read-only checkpoint source directory}" --dest "$MODEL_CACHE" --receipt "$STAGE/receipts/model.json" --candidate "$KNOWN_GOOD_CACHE" > "$R/receipts/model/$n.log" 2>&1 &
    done; wait
    for n in $(hosts); do on_node "$n" cat "$STAGE/receipts/model.json" > "$R/receipts/model/$n.json"; echo "$n: $(tail -1 "$R/receipts/model/$n.log")"; done ;;
  spec)
    NIC=${1:?fabric interface name required}
    python3 "$R/tools/make_run_spec.py" --job-id "$JOB" --run-id "$RUN_ID" --nic "$NIC" --model-receipts "$R/receipts/model" --out "$SPEC_LOCAL" --image-ref "${IMAGE_REF:?}" --image-id "${IMAGE_ID:?}"
    for n in $(hosts); do on_node "$n" bash -c "cat > $SPEC" < "$SPEC_LOCAL"; done ;;
  bootstrap)
    test -f "$SPEC_LOCAL"
    for n in $(hosts); do on_node "$n" python3 -B "$STAGE/tools/bootstrap_node.py" --spec "$SPEC" > "$R/receipts/bootstrap-$n.summary.json" 2>&1 & done; wait
    for n in $(hosts); do echo "$n: $(tail -1 "$R/receipts/bootstrap-$n.summary.json" | cut -c1-300)"; on_node "$n" cat "$(node_root)/bootstrap-node.json" > "$R/receipts/bootstrap-$n.json" 2>/dev/null || true; done ;;
  stage-tools)
    ROOT=$(node_root)
    for n in $(hosts); do
      on_node "$n" bash -c "set -euo pipefail; umask 022; W=$ROOT/work; test -d \$W/home
        mkdir -p \$W/tools \$W/inputs \$W/data \$W/output
        cp $STAGE/tools/*.py $STAGE/tools/*.sh $STAGE/tools/known-versions.json \$W/tools/; cp $SPEC \$W/tools/run-spec.json
        cp $STAGE/inputs/initial-snapshot-retry.patch $STAGE/inputs/model-reference-manifest.json $STAGE/tools/known-versions.json \$W/inputs/
        cp $STAGE/inputs/data/tbench2_train69.jsonl $STAGE/inputs/data/tbench2_eval.jsonl \$W/data/
        git -c advice.detachedHead=false clone --quiet $STAGE/inputs/OpenEnv.bundle \$W/OpenEnv && echo openenv=\$(git -C \$W/OpenEnv rev-parse HEAD)
        git -c advice.detachedHead=false clone --quiet $STAGE/inputs/terminal-bench-2.bundle \$W/terminal-bench-2 && echo tb2=\$(git -C \$W/terminal-bench-2 rev-parse HEAD)
        git -C \$W/terminal-bench-2 remote set-url origin https://github.com/laude-institute/terminal-bench-2.git
        echo origin=\$(git -C \$W/terminal-bench-2 remote get-url origin); sha256sum \$W/data/*.jsonl | cut -c1-16,66-" > "$R/receipts/stage-tools-$n.log" 2>&1 &
    done; wait
    for n in $(hosts); do echo "== $n: $(grep -E '^(openenv|tb2|origin)=' "$R/receipts/stage-tools-$n.log" | paste -sd' ')"; done ;;
  install)
    for n in $(hosts); do
      ( CID=$(cid_of "$n"); on_node "$n" docker exec --user 0 "$CID" python3 -B /work/tools/install_deps.py plan > "$R/receipts/install-plan-$n.log" 2>&1 || true ) &
    done; wait
    for n in $(hosts); do
      echo "== $n: $(python3 - "$R/receipts/install-plan-$n.log" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
i = text.find('{\n  "status"')
if i < 0: print("PLAN FAILED: " + text[-300:].replace("\n", " ")); raise SystemExit
d = json.loads(text[i:])
print(d["status"], "new=%d" % len(d["new_packages"]), "replacements=%s" % sorted(d["replacements"]))
PY
)"
    done ;;
  apply)
    # Optional: sha256 of the reviewed planned_versions JSON (identical across nodes). When given, a container
    # whose plan differs is skipped; without it, apply installs what plan wrote in that container.
    VERSIONS_SHA=${1:-}
    for n in $(hosts); do
      ( CID=$(cid_of "$n")
        if [ -n "$VERSIONS_SHA" ]; then
          on_node "$n" docker exec "$CID" python3 -c "import json,hashlib,sys
d=json.load(open('/opt/tb2-daytona-install/installation-plan.json'))
v=hashlib.sha256(json.dumps(d['planned_versions'],sort_keys=True).encode()).hexdigest()
sys.exit('planned_versions differ from the reviewed plan: '+v) if v!='$VERSIONS_SHA' else None" || { echo "$n: plan mismatch, apply skipped" > "$R/receipts/install-apply-$n.log"; exit 0; }
        fi
        on_node "$n" docker exec --user 0 "$CID" python3 -B /work/tools/install_deps.py apply > "$R/receipts/install-apply-$n.log" 2>&1 || true
        on_node "$n" docker exec "$CID" python3 -B /work/tools/install_deps.py verify > "$R/receipts/install-verify-$n.log" 2>&1 || true ) &
    done; wait
    for n in $(hosts); do
      echo "== $n apply: $(grep -o '"status": "[A-Z_]*"' "$R/receipts/install-apply-$n.log" | tail -1 || tail -c 200 "$R/receipts/install-apply-$n.log")"
      echo "== $n verify: $(grep -o 'VERIFY_JSON=.*"status": "[A-Z_]*"' "$R/receipts/install-verify-$n.log" | grep -o '"status": "[A-Z_]*"' | tail -1 || tail -c 300 "$R/receipts/install-verify-$n.log" | tr '\n' ' ')"
    done ;;
  key)
    H=$(head_node); test -s "$KEY_STAGING"
    on_node "$H" python3 -B "$STAGE/tools/receive_key_node.py" --spec "$SPEC"
    on_node "$H" python3 -B "$STAGE/tools/receive_key_node.py" --spec "$SPEC" --execute < "$KEY_STAGING" | tee "$R/receipts/key-transfer.json" ;;
  nccl)
    PROBE=${1:-nccl-smoke-1}
    for n in $(hosts); do on_node "$n" python3 -B "$STAGE/tools/run_nccl_node.py" --spec "$SPEC" --probe "$PROBE" > "$R/receipts/$PROBE-$n.log" 2>&1 & done; wait
    for n in $(hosts); do echo "$n: $(tail -1 "$R/receipts/$PROBE-$n.log")"; done ;;
  ray-ready)
    H=$(head_node); OUT=${1:-ray-ready-1}; CID=$(cid_of "$H")
    on_node "$H" docker exec "$CID" python3 -B /work/tools/ray_ready.py --spec /work/tools/run-spec.json --output "/work/$OUT.json" > "$R/receipts/$OUT.log" 2>&1 || true; tail -2 "$R/receipts/$OUT.log" ;;
  restage-tools)
    # Re-send only the tools after a local edit (node-local staging copy, then the node root's /work/tools if it exists).
    ROOT=$(node_root 2>/dev/null || true)
    for n in $(hosts); do
      echo "== $n"
      tar -C "$R" -czf - tools | on_node "$n" bash -c "set -e; umask 077; rm -rf $STAGE/tools; tar -xzf - -C $STAGE
        if [ -n '$ROOT' ] && [ -d $ROOT/work/tools ]; then umask 022; cp $STAGE/tools/*.py $STAGE/tools/*.sh $STAGE/tools/known-versions.json $ROOT/work/tools/; echo refreshed=$ROOT/work/tools; fi
        sha256sum $STAGE/tools/*.py | cut -c1-16,66- | sed 's#$STAGE/tools/##'"
    done ;;
  cleanup-key)
    # Only after this run's sandboxes are reconciled: remove the head node's private key file (and the login-node staging copy on request).
    H=$(head_node); ROOT=$(node_root)
    on_node "$H" bash -c "set -e; f=$ROOT/work/private/daytona.key; stat -c 'before: %n mode=%a uid=%u' \$f; shred -u \$f; test ! -e \$f && echo removed=\$f"
    if [ "${1:-}" = "--also-staging" ]; then shred -u "$KEY_STAGING" && echo "removed=$KEY_STAGING"; fi ;;
  cleanup-containers)
    # Remove exactly this run's containers (label miles.run=<run id>); never other containers, never Slurm.
    for n in $(hosts); do
      on_node "$n" bash -c "set -e; for c in \$(docker ps -aq --filter label=miles.run=$RUN_ID); do echo \"$n: removing \$c \$(docker inspect --format '{{.Name}}' \$c)\"; docker rm -f \$c >/dev/null; done; docker ps -a --filter label=miles.run=$RUN_ID -q | wc -l"
    done ;;
  evidence)
    # m3_evidence.py in every container over one run's dump_details; results fetched to $R/evidence/<train-run>/<node>.json
    TRAIN_RUN=${1:?training run id required}; ROOT=$(node_root); mkdir -p "$R/evidence/$TRAIN_RUN"
    for n in $(hosts); do
      CID=$(cid_of "$n")
      on_node "$n" docker exec "$CID" bash -c "test -d /work/output/$TRAIN_RUN/dump_details && python3 -B /work/tools/m3_evidence.py --dump /work/output/$TRAIN_RUN/dump_details --out /work/output/$TRAIN_RUN/evidence-$n.json || echo 'no dump_details on this node'" 2>&1 | tail -2
      on_node "$n" cat "$ROOT/work/output/$TRAIN_RUN/evidence-$n.json" > "$R/evidence/$TRAIN_RUN/$n.json" 2>/dev/null || true
    done; ls -la "$R/evidence/$TRAIN_RUN" ;;
  fetch)
    on_node "$1" cat "$2" > "$R/receipts/$3"; echo "fetched $R/receipts/$3 ($(wc -c < "$R/receipts/$3") bytes)" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
