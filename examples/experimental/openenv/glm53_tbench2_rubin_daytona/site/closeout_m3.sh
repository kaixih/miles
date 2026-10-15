#!/usr/bin/env bash
# Close out one training run on the 8-node allocation. Run on the login node after the driver has returned
# (SCRATCH_ROOT / RUN_PREFIX / KEY_STAGING as for prepare_run.sh):
#
#   bash closeout_m3.sh <jobid> <train-run-id> [--remove-key] [--remove-containers]
#
# 1. per-node evidence (m3_evidence.py) and archive of each node's dump_details (trajectory, rollout_data,
#    events, dashboard metrics) into the run's durable root
# 2. sandbox reconcile under the run label with --delete (every leftover deleted and re-fetched -> 404)
# 3. --remove-key: shred the head node's private key file and the login-node staging copy
# 4. --remove-containers: remove exactly this run's containers (label miles.run=<run id>)
# Slurm is never cancelled here.
set -uo pipefail
JOB=$1; TRAIN_RUN=$2; shift 2
RUN_ID=${RUN_PREFIX:-glm53-tb2-daytona}-j${JOB}
R=${SCRATCH_ROOT:?set SCRATCH_ROOT}/$RUN_ID
P="bash $R/tools/prepare_run.sh"
ROOT=$(python3 -c "import json; print(json.load(open('$R/run-spec.json'))['node_root'])")
HEAD=$(python3 -c "import json; print(json.load(open('$R/run-spec.json'))['nodes'][0]['node'])")
hosts() { scontrol show hostnames "$(scontrol show job -o "$JOB" | tr ' ' '\n' | grep '^NodeList=' | cut -d= -f2)"; }
on_node() { local node=$1; shift; srun --jobid="$JOB" --overlap --nodes=1 --ntasks=1 --cpu-bind=none --chdir=/tmp -w "$node" "$@"; }
cid_of() { on_node "$1" cat "$ROOT/bootstrap-node.json" | python3 -c "import json,sys; print(json.load(sys.stdin)['container_id'])"; }

echo "### evidence $(date -u +%T)"
$P evidence "$JOB" "$TRAIN_RUN" 2>&1 | grep -v "Exited with exit code"
echo "### archive dump_details per node $(date -u +%T)"
mkdir -p "$R/m3"
for n in $(hosts); do
  mkdir -p "$R/m3/$n"
  on_node "$n" bash -c "cd $ROOT/work/output/$TRAIN_RUN 2>/dev/null && tar -cf - \$(ls -d dump_details/trajectory dump_details/rollout_data dump_details/events dump_details/dashboard/metrics.jsonl dump_details/dashboard/meta.json dump_details/dashboard/trajectories evidence-*.json 2>/dev/null)" 2>/dev/null | tar -C "$R/m3/$n" -xf - 2>/dev/null
  echo "$n: $(du -sh "$R/m3/$n" 2>/dev/null | cut -f1) $(find "$R/m3/$n" -type f | wc -l) files"
done
echo "### reconcile sandboxes (--delete) $(date -u +%T)"
CID=$(cid_of "$HEAD")
on_node "$HEAD" docker exec --env DAYTONA_API_KEY_FILE=/work/private/daytona.key --env DAYTONA_API_URL=https://app.daytona.io/api --env PYTHONPATH=/opt/Megatron-LM:/opt/miles/examples/experimental/openenv:/opt/miles "$CID" python3 -B /work/tools/reconcile_sandboxes.py --run-id "$TRAIN_RUN" --out "/work/output/reconcile-$TRAIN_RUN.json" --delete 2>&1 | grep -v "Exited with exit code" | tail -2
on_node "$HEAD" cat "$ROOT/work/output/reconcile-$TRAIN_RUN.json" > "$R/receipts/reconcile-$TRAIN_RUN.json" 2>/dev/null
python3 -c "import json; d=json.load(open('$R/receipts/reconcile-$TRAIN_RUN.json')); print('clean:', d['clean'], 'deleted:', len(d['deleted']), 'remaining_after:', d['remaining_after'])"
echo "### account-wide sandbox listing (ids/states only)"
on_node "$HEAD" docker exec -i --env DAYTONA_API_KEY_FILE=/work/private/daytona.key --env DAYTONA_API_URL=https://app.daytona.io/api --env PYTHONPATH=/opt/Megatron-LM:/opt/miles/examples/experimental/openenv:/opt/miles "$CID" python3 -B - <<'PY' 2>&1 | grep -v "Exited with exit code"
import sys; sys.path.insert(0, "/opt/miles/examples/experimental/openenv")
import tb2_sandbox_daytona as p
from daytona import ListSandboxesQuery
rows = [(s.id[:8], str(s.state).replace("SandboxState.",""), (s.labels or {}).get("openenv-run-id")) for s in p.make_daytona().list(ListSandboxesQuery(limit=200))]
print("total visible:", len(rows)); [print("  ", r) for r in rows]
PY
for arg in "$@"; do
  case "$arg" in
    --remove-key) echo "### cleanup-key $(date -u +%T)"; $P cleanup-key "$JOB" --also-staging 2>&1 | grep -v "Exited with exit code" ;;
    --remove-containers) echo "### cleanup-containers $(date -u +%T)"; $P cleanup-containers "$JOB" 2>&1 | grep -v "Exited with exit code" ;;
  esac
done
echo "### CLOSEOUT_DONE $(date -u +%T)"
