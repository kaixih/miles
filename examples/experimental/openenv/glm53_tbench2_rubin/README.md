# GLM-5.3-Flash — OpenSandbox TB2 on eight Rubin nodes

This is the Rubin counterpart of the [GLM-5.2 TB2 recipe](../glm52_tbench2/README.md):
fully asynchronous GRPO, with four training nodes and four rollout nodes, each
with four GPUs. Miles owns the agent loop; OpenEnv's unmodified TB2 server runs
inside a remote OpenSandbox sandbox for each episode.

The reference experiment completed **8 optimizer updates on all 16 training
ranks**, followed by weight broadcasts to both rollout engines. This establishes
functionality, not a learning improvement or a promise that 20 updates fit in
one allocation. The files here consolidate that configuration; the image-build
and site commands below have not themselves been rerun end to end.

## Container and changes

Base container: **`miles-rubin-dev:sgfix-20261011-j2460554` (internal)**.
Resolve its registry address locally; no internal registry address is included here.
Use this exact image for this recipe, rather than replacing it with a moving tag.

| Component | Reference version / change |
|---|---|
| Miles in the base image | `0d9d726e0f9b9d487c36a0c342012b917820fb04` |
| SGLang in the base image | `6b5f3dfc5a395d6fd56d105c34c8e3542846ca4c` |
| Megatron in the base image | `fd15ee20a4f03b03680529baf4b8f7eeed64df1d`; unchanged source, copied from `/root/Megatron-LM` to readable `/opt/Megatron-LM` |
| OpenSandbox provider | This branch's `3bddb624` + `439f80b2` changes only; do not overlay the entire older branch onto the image |
| Initialization retry | [PR #4029](https://github.com/radixark/miles/pull/4029); bundled `initial-snapshot-retry.patch` preserves the first successful weight snapshot |
| OpenEnv, including `tbench2_env` | Unmodified `38b2a31354d1d4d627894412b9d469e92f4a2c61` |
| TB2 task checkout | Unmodified `2fd12b88aafdd04a52c298e3940bcb189f9766d6` |

The base image already contains the Rubin/model support. Do not reapply historical
GLM KDA, parser, template, timeout, strict-filter, or direct-HF patch bundles.
There are two source patches above: the provider and the snapshot retry fix.
The remaining setup is dependencies, native flags, and site configuration.

`requirements-rubin.txt` records the observed package delta, not a new training
stack: 32 new dependency packages plus the separately installed OpenEnv and TB2
packages (34 additions total). It also changes `openai` 2.6.1 → 2.7.2 and
`mcp` 2.3.0 → 1.30.0 to satisfy the pinned OpenEnv/FastMCP requirements.
Installation uses `--no-deps` to preserve the image's other packages; this lock
must not be reused against arbitrary images. Gradio and server-only Camel/Docker
extras are not installed in the training container.

## 1. Prepare a derived container

On an **ARM64** machine with Docker and access to the internal base image, check
out `kaixih/miles`, branch `kaixi/openenv-opensandbox`. Build from a temporary
context so generated patches do not alter the checkout:

```bash
export MILES_ROOT=/path/to/miles
export BASE_IMAGE='<locally resolved reference for the base container above>'
export PREPARED_IMAGE=miles-glm53-tb2:rubin
RECIPE_DIR="$MILES_ROOT/examples/experimental/openenv/glm53_tbench2_rubin"
BUILD_CONTEXT=$(mktemp -d)
cp "$RECIPE_DIR"/* "$BUILD_CONTEXT/"

# Freeze the provider diff; later README/recipe commits are intentionally excluded.
git -C "$MILES_ROOT" diff 3d388872 439f80b2 -- > "$BUILD_CONTEXT/provider.patch"
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$PREPARED_IMAGE" "$BUILD_CONTEXT"
```

The Dockerfile applies both patches to the image's Miles checkout, installs the
pinned client dependencies, relocates Megatron without changing its source, and
prepares the original 69/20 task split. The recipe runs from `/opt/miles` inside
the image. **Do not bind-mount this branch over `/opt/miles`**: its base predates
the GLM-5.3 support used by this run.

Distribute this prepared image to all eight nodes through your normal image
registry or `docker save` / `docker load`. Verify the same image ID on every node.
The runtime steps below use Docker; they are not Pyxis/Enroot commands.

## 2. Allocate nodes and prepare storage

Obtain **eight exclusive Rubin nodes in the same NVL72 domain**, with four GPUs
per node (approximately 280 GiB each). Partition/account/constraint names are
site-specific. From a persistent shell inside that Slurm allocation:

```bash
export RUN_ID="glm53-tb2-${SLURM_JOB_ID}"
export CONTAINER_NAME="$RUN_ID"
export PREPARED_IMAGE=miles-glm53-tb2:rubin
export HOST_RUN_ROOT="/raid/dldata/$USER/$RUN_ID"
export HF_MODEL_DIR=/path/on/every/node/to/original/GLM-5.3-Flash
export FABRIC_IFACE='<peer-routable compute interface>'
export IMEX_DEVICE='<site-assigned /dev/nvidia-caps-imex-channels/channel...>'
export RUN_UID=$(id -u)
export RUN_GID=$(id -g)
```

The model path must hold the **complete original HF FP8 model**, including its
config, tokenizer, index, scales and all 62 weight shards (71 files in the
reference snapshot, 45 layers). Check file hashes against the original model
manifest on all nodes. Do not point it at an offline `torch_dist` conversion or
a reduced-layer model. Mount this directory read-only.

`HOST_RUN_ROOT` must be a fresh directory on each node's local disk. Reserve at
least 2 TiB free on each training node for streamed optimizer state, plus room
for logs and rollout dumps. Verify normal-user write/read/delete access on the
actual mount. The allocated IMEX device must be usable by that same UID/GID;
do not solve this by running training as root or making the host device public.

Resolve compute IPs (do not select a management IP or `127.0.0.1`). Sort them
numerically, matching Miles placement; the first four nodes train, the last
four run the two TP8 rollout engines:

```bash
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpu-bind=none bash -c '
  ip -4 -o addr show dev "$FABRIC_IFACE" | awk -v n="$(hostname -s)" \
    "{split(\$4,a,\"/\"); print a[1],n}"
' | sort -V > "$RUN_ID.nodes"
export HEAD_IP=$(awk 'NR==1 {print $1}' "$RUN_ID.nodes")
export HEAD_NODE=$(awk 'NR==1 {print $2}' "$RUN_ID.nodes")
cat "$RUN_ID.nodes"
```

There must be exactly eight unique nodes/IPs and four allocated GPUs per node.
Check the actual Slurm end time and leave at least 20 minutes for collection and
cleanup. This launcher does not extend allocations or add a custom soft-final
timer. Choose the update target to fit the remaining time.

## 3. Start fresh containers

These commands require exclusive access to the allocated GPUs. A same-name
container is an error; inspect it instead of deleting/replacing it blindly.

```bash
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpu-bind=none bash -c '
  set -eu
  test ! -e "$HOST_RUN_ROOT"
  umask 077
  mkdir -p "$HOST_RUN_ROOT/work/home" "$HOST_RUN_ROOT/tmp" \
    "$HOST_RUN_ROOT/work/cache" "$HOST_RUN_ROOT/work/offload"
  docker run -d --name "$CONTAINER_NAME" --hostname "$(hostname -s)" \
    --pull=never --restart=no --user "$RUN_UID:$RUN_GID" \
    --gpus all --network host --ipc host \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    --ulimit nofile=65536:524288 --device "$IMEX_DEVICE:$IMEX_DEVICE:rwm" \
    --label "miles.run=$RUN_ID" --workdir /opt/miles \
    --mount "type=bind,src=$HOST_RUN_ROOT/work,dst=/work" \
    --mount "type=bind,src=$HOST_RUN_ROOT/tmp,dst=/tmp" \
    --mount "type=bind,src=$HF_MODEL_DIR,dst=/inputs/GLM-5.3-Flash,readonly" \
    --env HOME=/work/home --env PYTHONUNBUFFERED=1 --env PYTHONNOUSERSITE=1 \
    --env SLURM_JOB_ID --env MASTER_ADDR="$HEAD_IP" \
    --env GLOO_SOCKET_IFNAME="$FABRIC_IFACE" \
    --env NCCL_SOCKET_IFNAME="=$FABRIC_IFACE" --env NCCL_CUMEM_ENABLE=1 \
    --entrypoint /bin/sleep "$PREPARED_IMAGE" infinity \
    > "$HOST_RUN_ROOT/container.id"
  docker exec "$CONTAINER_NAME" python3 -c \
    "from pathlib import Path; p=Path(\"/work/write-probe\"); p.write_text(\"ok\"); assert p.read_text()==\"ok\"; p.unlink()"
  docker exec "$CONTAINER_NAME" python3 -c \
    "import openenv, tbench2_env, opensandbox, megatron.core; import openenv_opensandbox_agent_function"
'
```

Keep the default private PID namespace; do not add `--pid=host`. Before loading
the model, run your site's **real 32-rank NCCL P2P/MNNVL smoke test inside these
containers**, using the same devices/interfaces, and verify all ranks exit
successfully. Seeing eight nodes in Ray or passing a CPU test does not verify
cross-node GPU communication.

## 4. Start Ray and keep its Slurm step alive

```bash
srun --overlap --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpu-bind=none \
  bash -c 'exec docker exec "$CONTAINER_NAME" python3 /opt/tb2-tools/rubin_node.py \
    --head-ip "$HEAD_IP" --interface "$FABRIC_IFACE"' \
  > "$RUN_ID.ray.log" 2>&1 &
RAY_STEP_PID=$!
```

Keep this parent shell/tmux pane alive through the run. `rubin_node.py` is a
small site adapter: it starts one Ray daemon per container and blocks, while
the training recipe uses Miles's existing `execute_train` with external Ray.

Before submitting training, verify all eight Ray nodes are alive and the
cluster has 32 GPUs. Run on the head through `srun --overlap`:

```bash
srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none -w "$HEAD_NODE" \
  docker exec --env MASTER_ADDR="$HEAD_IP" "$CONTAINER_NAME" python3 -c '
import os, ray
ray.init(address=os.environ["MASTER_ADDR"] + ":6379")
nodes = [node for node in ray.nodes() if node["Alive"]]
assert len(nodes) == 8, nodes
assert ray.cluster_resources()["GPU"] == 32
print([(node["NodeManagerAddress"], node["Resources"].get("GPU")) for node in nodes])
ray.shutdown()
'
```

The head dashboard/API binds to the **head compute IP**, not localhost. Verify
that another node can reach `http://$HEAD_IP:8265/api/version`. A remote Miles
worker uses this API; a working browser on the head alone is insufficient.

## 5. Configure OpenSandbox on the head

Set the service origin and, when required, a CA file path:

```bash
export OPEN_SANDBOX_API_URL='<your HTTPS OpenSandbox service origin>'
export OPEN_SANDBOX_API_KEY_FILE=/work/private/opensandbox.key
# Optional: copy your service CA onto the head at this path first.
export OPEN_SANDBOX_CA_FILE=/work/private/opensandbox-ca.pem
```

Provision the API key using your secret manager or a **non-echoing stdin
transfer**, into the head's `/work/private/opensandbox.key`, owned by the runtime
UID with mode `0600` (directory `0700`). Do not put the key in command arguments,
shell history, this repository, or archives. Only the head needs it:
`--pin-rollout-manager-to-head` places the provider-calling agent processes
there; it does not move the rollout GPUs onto the head. Unset
`OPEN_SANDBOX_CA_FILE` if the service uses the default trusted CA chain.

The service must support the provider API and authenticated OpenEnv HTTP/WS
forwarding used by this branch. CPU sandboxes need network/package access to
start the unmodified TB2 environment server. Allow quota for the configured
128 in-flight training trajectories and evaluation activity; this is separate
from the 64-sample training batch.

## 6. Run the recipe

Start with `UPDATES=3` to check functionality, then use a fresh run/container
for a longer target. `UPDATES=20` is the normal recipe default, not a claim about
wall-clock duration. Do not reuse the same output/offload/cache directories.

```bash
export UPDATES=3
srun --overlap --nodes=1 --ntasks=1 --cpu-bind=none -w "$HEAD_NODE" \
  docker exec \
    --env MILES_SCRIPT_EXTERNAL_RAY=1 --env MASTER_ADDR="$HEAD_IP" \
    --env RAY_ADDRESS="http://$HEAD_IP:8265" --env RAY_TMPDIR=/work/ray \
    --env RUN_ID --env UPDATES --env OPEN_SANDBOX_API_URL \
    --env OPEN_SANDBOX_API_KEY_FILE --env OPEN_SANDBOX_CA_FILE \
    "$CONTAINER_NAME" bash -c '
  set -eu
  cd /opt/miles
  # Keep per-node inherited NIC settings; do not forward a head-specific NIC
  # as a job-wide override. This matches the reference submitter environment.
  unset NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
  read -r TRAIN_SHA _ < <(sha256sum /opt/tb2-data/train69.jsonl)
  read -r EVAL_SHA _ < <(sha256sum /opt/tb2-data/eval20.jsonl)
  python3 examples/experimental/openenv/glm53_tbench2_rubin/run_glm5_3_flash_tb2.py train \
    --run-id "$RUN_ID" --num-rollout "$UPDATES" \
    --hf-checkpoint /inputs/GLM-5.3-Flash \
    --prompt-data /opt/tb2-data/train69.jsonl --eval-prompt-data /opt/tb2-data/eval20.jsonl \
    --prompt-data-sha256 "$TRAIN_SHA" --eval-prompt-data-sha256 "$EVAL_SHA" \
    --tasks-dir /opt/terminal-bench-2 --output-dir /work/output \
    --megatron-path /opt/Megatron-LM --offload-train-disk-dir /work/offload \
    --cache-dir /work/cache --cluster-backend ray --deploy-component all
' > "$RUN_ID.driver.log" 2>&1
```

The Python file contains the complete training arguments; the table below is
the intended configuration, not a separate set of overrides:

| Setting | Rubin recipe |
|---|---|
| Training | Full 45-layer GLM-5.3-Flash, BF16, 4 × 4 GPUs; TP4 / PP4 / EP4 / CP1; pipeline layers 11/11/11/12 |
| Initialization | Direct original HF load; no offline `torch_dist` checkpoint |
| Rollout | 4 × 4 GPUs; two TP8 / EP8 engines; FP8 weights, BF16 KV; Triton MoE, TileLang DSA; no speculative decoding |
| FP8 refit | Native `--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}'`; match the original HF scale rule |
| Async sampling | 128 in-flight trajectories, `sample` submission granularity |
| Update batch | 8 prompts × 8 samples = 64; native `check_no_aborted` filter |
| GRPO / Adam | LR 1e-6, WD 0.1, betas 0.9/0.98; KL/entropy 0; clip 0.2/0.28; TIS 0..2 |
| Episode limits | 8,192 generated tokens per turn, 30 turns, 3,600 seconds per episode, 65,536 assembled sequence limit |
| Evaluation | Fixed 69 train / 20 held-out tasks; baseline, every 10 updates, native final eval; 2 samples/task, temperature 0.8 |
| TITO / prompt | Native `glm53` TITO and session server v1; no dataset-level `--apply-chat-template` (agent/session renders messages) |
| Memory | Full recomputation, dynamic batching at 8,192 tokens/GPU, optimizer state streamed to local disk |
| Weight validation | Native strict initial comparison, existing `visual.` exclusion; compare both engines and observe each update broadcast |
| Output | `dump_details` and sandbox lifecycle logs; no model/optimizer checkpoints (`--save` omitted) |

Compared with the GLM-5.2 example, the intentional differences are the model,
provider, 8-node topology/backends, direct-HF loading, explicit scale setting,
single prompt rendering, head placement, and no checkpoint saving. The recipe
does not carry the old extra TITO mismatch filter, retry policy, or custom final
timer. `sample` is retained from the successful GLM run; the earlier Qwen `group`
experiment is not silently incorporated.

## 7. Verify results and collect before cleanup

The first weight broadcast is **initialization, not an optimizer update**.
Require successful initialization of all 16 trainer ranks, both engines'
strict comparisons, then actual optimizer events/loss/grad norm and subsequent
weight broadcasts. A submitted Ray job or completed sampling batch alone is
not training success.

Collect `$RUN_ID.driver.log`, `$RUN_ID.ray.log`, and each node's `work/output`
and `work/ray` directories to durable storage. The head's output includes
`dump_details` and `sandbox/*/lifecycle.json`; rank logs also live on other nodes.
Archive only those explicit paths, **not all of `work`**: `work/private` contains
the key. Keep model/optimizer scratch out of diagnostic archives.

For evaluation report successes / 40 and successes / officially scored samples
separately. Preserve unscored official rewards as `null`; native RM fallback
zeros are not evidence that the task was scored. A flat/noisy short curve does
not invalidate the functionality check, nor establish learning improvement.

On normal completion the provider closes episode sandboxes. Check lifecycle
records for `cleanup.deletion_confirmed` and DELETE/GET-404 evidence. On failure
or interruption, reconcile **this run's** sandbox IDs and `openenv-create-id`
metadata with the service, including creates whose response was lost, before
deleting the key. Do not assume killing Ray closes remote sandboxes.

After the driver is terminal, archives are durable, and all this run's sandboxes
are accounted for, remove only the containers recorded for this run:

```bash
srun --overlap --nodes=8 --ntasks=8 --ntasks-per-node=1 --cpu-bind=none bash -c '
  set -eu
  cid=$(cat "$HOST_RUN_ROOT/container.id")
  test "$(docker inspect --format "{{index .Config.Labels \"miles.run\"}}" "$cid")" = "$RUN_ID"
  rm -f "$HOST_RUN_ROOT/work/private/opensandbox.key"
  docker rm -f "$cid"
'
wait "$RAY_STEP_PID" || true
```

Keep bind-mounted results and the original model. The commands do not cancel
Slurm or remove unrelated containers. If the allocation expires before cleanup,
use the durable lifecycle records and your service credentials to reconcile
remaining remote sandboxes separately.

## Known bring-up pitfalls

- `Too many open files`: use the documented container `nofile` limit; do not
  patch Ray or lower the training batch to work around it.
- State API connection refused on another node: dashboard must advertise the
  head compute IP, not localhost. This was a site-script error.
- Ray starter disappears: keep the parent Slurm step alive while `--block`
  processes run. This was also a site-script lifecycle error.
- Initial strict comparison fails after a reset timeout: verify the snapshot
  retry patch. Do not disable the checker.
- Initial FP8 values differ despite correct transport: verify the trainer
  runtime receives `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`. This does not enable
  FP32 training; it selects the refit scale rule for this BF16/FP8 configuration.
- Credentials missing on a worker: verify head placement and the head's private
  file. Distributing a file path does not distribute its contents.
