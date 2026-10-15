#!/usr/bin/env bash
# Submit a sampling check or a training run from inside the HEAD run container, as the run UID.
#
#   bash launch_train.sh sample <run-id> <prompt-jsonl> [launcher args...]
#       small sampling batch, no optimizer (two nodes); defaults SAMPLE_NUM_ROLLOUT=1 SAMPLE_PROMPTS=1
#       SAMPLE_SAMPLES=1 SAMPLE_CONCURRENCY=1 (env overrides)
#   bash launch_train.sh train <run-id> <train-jsonl> <eval-jsonl> [launcher args...]
#       training run (eight nodes); evaluation stays at --eval-interval 10, which with a short --num-rollout means a baseline
#       eval before training and the forced final eval, no mid-run eval. Pass --eval-interval 0 to skip both.
#
# Expects the container env from bootstrap (MASTER_ADDR = head compute IP), the Ray cluster started by
# rubin_node.py on every node, the install receipt from install_deps.py, and /work/private/daytona.key.
# Only the key file PATH and the API URL reach the rollout workers.
set -euo pipefail
MODE=$1; RUN_ID=$2; shift 2
: "${MASTER_ADDR:?MASTER_ADDR (head compute IP) must be set in the container environment}"
export DAYTONA_API_KEY_FILE="${DAYTONA_API_KEY_FILE:-/work/private/daytona.key}"
export DAYTONA_API_URL="${DAYTONA_API_URL:-https://app.daytona.io/api}"
export OPENENV_LAUNCHER="${OPENENV_LAUNCHER:-${USER:-miles}}"
export MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS="http://$MASTER_ADDR:8265" RAY_TMPDIR=/work/ray
MEGATRON_PATH=$(python3 -c "import json; print(json.load(open('/opt/tb2-daytona-install/receipt.json'))['megatron_path'])")
export PYTHONPATH="$MEGATRON_PATH:/opt/miles/examples/experimental/openenv:/opt/miles${PYTHONPATH:+:$PYTHONPATH}"
# Keep per-node inherited NIC settings; do not forward a head-specific NIC as a job-wide override.
unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME WANDB_API_KEY WANDB_KEY
test -r "$DAYTONA_API_KEY_FILE" || { echo "key file not readable: $DAYTONA_API_KEY_FILE" >&2; exit 1; }
test "$(stat -c '%a %u' "$DAYTONA_API_KEY_FILE")" = "600 $(id -u)" || { echo "key file must be 0600 and owned by $(id -u)" >&2; exit 1; }
test -d /work/output/"$RUN_ID" && { echo "output for $RUN_ID already exists" >&2; exit 1; }
cd /opt/miles
LAUNCHER=/work/tools/run_glm5_3_flash_tb2_daytona.py
COMMON=(--run-id "$RUN_ID" --hf-checkpoint /inputs/GLM-5.3-Flash --tasks-dir /work/terminal-bench-2
        --output-dir /work/output --megatron-path "$MEGATRON_PATH"
        --offload-train-disk-dir "/work/offload/$RUN_ID" --cache-dir "/work/cache/$RUN_ID"
        --cluster-backend ray --deploy-component all)
sha() { sha256sum "$1" | cut -c1-64; }
case "$MODE" in
  sample)
    PROMPT=$1; shift
    exec python3 "$LAUNCHER" train "${COMMON[@]}" --num-nodes 2 --debug-rollout-only --eval-interval 0 \
      --num-rollout "${SAMPLE_NUM_ROLLOUT:-1}" --rollout-batch-size "${SAMPLE_PROMPTS:-1}" \
      --n-samples-per-prompt "${SAMPLE_SAMPLES:-1}" --async-max-concurrent-samples "${SAMPLE_CONCURRENCY:-1}" \
      --prompt-data "$PROMPT" --prompt-data-sha256 "$(sha "$PROMPT")" "$@" ;;
  train)
    TRAIN=$1; EVAL=$2; shift 2
    exec python3 "$LAUNCHER" train "${COMMON[@]}" --num-nodes 8 \
      --prompt-data "$TRAIN" --prompt-data-sha256 "$(sha "$TRAIN")" \
      --eval-prompt-data "$EVAL" --eval-prompt-data-sha256 "$(sha "$EVAL")" "$@" ;;
  *) echo "mode must be sample or train" >&2; exit 2 ;;
esac
