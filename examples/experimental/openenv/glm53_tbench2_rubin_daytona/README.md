# GLM-5.3-Flash on Terminal-Bench-2 with Daytona sandboxes (Rubin)

This directory holds the recipe and the results of a fully asynchronous GRPO run of
GLM-5.3-Flash on OpenEnv's Terminal-Bench-2 environment, on Rubin nodes, with every episode
executed in its own Daytona sandbox through Miles' native Daytona provider. The run trained
for 20 optimizer steps and collects the three per-step series of Figure 5 in the Miles v0.1
report (§9): step wall-clock, trainer-vs-rollout log-probability divergence, and raw task
reward. The launcher and helper scripts used for the run are in this directory.

Setup:

1. **GPUs**: 8 Rubin nodes, 4 GPUs each (32 in total): 16 for training (TP4 / PP4 / EP4,
   layers 11/11/11/12) and 16 for rollout (two TP8 / EP8 SGLang engines).
2. **Model**: GLM-5.3-Flash, full 45 layers, original FP8 HF checkpoint, served and trained
   as-is (no checkpoint conversion).
3. **Data**: Terminal-Bench-2 via OpenEnv `tbench2_env`, 69 training tasks; 8 prompts × 8
   samples = 64 trajectories per step, up to 30 turns and 3600 s per episode, one Daytona
   sandbox per episode built from the task's official image. Evaluation was off for this run.
4. **Steps**: 20 optimizer updates, fully async (128 trajectories in flight, in-place pause
   for weight sync), GRPO with Adam at lr 1e-6, TIS clipped to [0, 2].
5. **Container**: `miles-rubin-dev:sgfix-20261011-j2460554` (internal), built from
   `docker/Dockerfile.rubin` in radixark/miles#4011. On top of the image: the Daytona SDK and
   OpenEnv added by `install_deps.py`, and the one-file fix from radixark/miles#4029.
6. **Results**: all 20 updates completed (16/16 trainer ranks healthy, weights broadcast to
   both engines on every step). The three Figure 5 series are below.

![Figure 5-style series from the 20-step run](assets/fig5.png)

(a) wall-clock per training step: median 367 s over steps 1–19, 355–385 s from step 5 on
(step 0 includes the first rollout, 2213 s, and is clipped). (b) mean |trainer − rollout|
log-probability at sampled tokens: 0.0787 on average, 0.0875 → 0.0627. (c) raw task reward per
step: mean 0.188, 9-step moving average 0.219 → 0.189, flat as expected for 20 steps.
Per-step values are in [`assets/fig5-series.csv`](assets/fig5-series.csv).

## Container

Use **`miles-rubin-dev:sgfix-20261011-j2460554` (internal)**, built from
`docker/Dockerfile.rubin` in [radixark/miles#4011](https://github.com/radixark/miles/pull/4011).
It already contains Miles, SGLang and Megatron-LM with the Rubin / GLM-5.3-Flash support. The
Daytona SDK and OpenEnv are not in the image; `install_deps.py` (next section) adds them. Where
each piece comes from, with the revisions this run used:

| Component | Revision | Where |
|---|---|---|
| Miles | `0d9d726e0f9b9d487c36a0c342012b917820fb04` | in the image, `/opt/miles` |
| SGLang | `6b5f3dfc5a395d6fd56d105c34c8e3542846ca4c` | in the image |
| Megatron-LM | `fd15ee20a4f03b03680529baf4b8f7eeed64df1d` | in the image, `/root/Megatron-LM`; `install_deps.py` relocates it to `/opt/Megatron-LM` when `/root` is unreadable for the run user, which is the normal case |
| Daytona SDK | `0.224.1` | added by `install_deps.py` |
| OpenEnv incl. `tbench2_env` | `38b2a31354d1d4d627894412b9d469e92f4a2c61` | checkout staged at `/work/OpenEnv`, installed editable by `install_deps.py` |
| Terminal-Bench-2 tasks | `2fd12b88aafdd04a52c298e3940bcb189f9766d6` | checkout staged at `/work/terminal-bench-2`, `origin` set to GitHub: the sandbox build downloads each task directory from that origin at this commit |

Do not overlay a newer Miles, SGLang or Megatron tree over the preinstalled ones, and do not
reapply historical GLM-5.3 patch bundles (parser, template, timeouts, strict filter): the image
is complete as built, and the only addition is what `install_deps.py` does.

## What has to be added to the image

### 1. Dependencies — [`install_deps.py`](install_deps.py)

The image has no Daytona SDK and no OpenEnv. `install_deps.py` adds them in three steps and
leaves the rest of the image's Python stack as it is:

- `plan` (as root): works out what is missing while pinning every package the image already
  has, and writes a hash-locked requirements file under `/opt/tb2-daytona-install/`. Nothing
  is installed. Two existing pins have to move because OpenEnv needs them (`openai` 2.6.1 →
  2.7.2, `mcp` 2.3.0 → 1.30.0); everything else stays.
- `apply` (as root): installs exactly that lock with `pip install --no-deps --require-hashes`,
  installs OpenEnv and `tbench2_env` editable from `/work/OpenEnv` (copied to `/opt/OpenEnv`),
  relocates Megatron-LM to `/opt/Megatron-LM` when `/root` is unreadable for the run user (the
  launcher takes the resulting path from the install receipt), and applies the patch from
  section 2. It refuses to run if the installer, the lock or the installed packages changed
  after `plan`. The run user is the owner of the bind-mounted `/work`; pass `--uid`/`--gid` if
  that is not the user the training runs as.
- `verify` (as the run user): imports everything the launcher needs and prints a receipt.

Run the three steps in every container:

```bash
# as root
python3 install_deps.py plan      # resolves and locks; shows what will change; installs nothing
python3 install_deps.py apply     # installs exactly what plan wrote
# as the run user
python3 install_deps.py verify    # imports everything the launcher needs
```

### 2. One source change — radixark/miles#4029

The image predates [radixark/miles#4029](https://github.com/radixark/miles/pull/4029) (keep the
first inference-engine weight snapshot across reset retries). It is bundled here as
[`initial-snapshot-retry.patch`](initial-snapshot-retry.patch), and `install_deps.py apply`
applies it to `/opt/miles`; there is nothing to do by hand. The patch matches the pinned image
revision only; on a newer Miles tree use the PR itself.

## Files in this directory

| File | Role |
|---|---|
| [`run_glm5_3_flash_tb2_daytona.py`](run_glm5_3_flash_tb2_daytona.py) | The launcher: builds the `train_async.py` command (model, topology, GRPO settings, the Daytona agent function) and submits it to the Ray cluster. |
| [`launch_train.sh`](launch_train.sh) | Head-container wrapper around the launcher: environment, dataset hashes; `sample` runs a small sampling check without an optimizer, `train` runs training. |
| [`install_deps.py`](install_deps.py), [`known-versions.json`](known-versions.json) | The three-step dependency install above. |
| [`initial-snapshot-retry.patch`](initial-snapshot-retry.patch) | radixark/miles#4029 as a patch file; applied by `install_deps.py`. |
| [`fig5_collect.py`](fig5_collect.py) | Builds the Figure 5 series (JSON, CSV, PNG) from the driver log. |
| [`reconcile_sandboxes.py`](reconcile_sandboxes.py) | Lists and deletes the Daytona sandboxes a run leaves behind (see "After a run"). |
| [`assets/`](assets/) | The figure and its per-step CSV. |
| [`data/`](data/) | The frozen Terminal-Bench-2 task split: 69 training tasks and 20 held-out tasks, one row per task in the format [`../make_tbench2_data.py`](../make_tbench2_data.py) produces. |
| [`site/`](site/) | **Site-specific.** The preparation between "Slurm gave us N nodes" and "`launch_train.sh` can run in the head container", as we did it on our cluster; kept as a reference, assumptions below. |

What `site/` does, in order:

1. `make_inputs.sh`: builds the login-node run root once. `files <run root>` copies these
   scripts, the task split from `data/`, the patch and `known-versions.json` into it;
   `manifest <run root> <checkpoint dir>` writes the size and SHA256 of every checkpoint file
   to `inputs/model-reference-manifest.json` (run it where the checkpoint is on local disk; it
   is about 330 GB).
2. `prepare_run.sh`: the login-node entry point, one sub-command per step; each step reaches
   the nodes through `srun` and runs one of the scripts below.
3. `stage_model_node.py`: on every node, verify the node-local copy of the checkpoint against
   the reference manifest (file set, sizes, SHA256); copy it from the read-only source if absent.
4. `make_run_spec.py`: ask Slurm for the node list, read each node's fabric IP, order the nodes
   by IP (the lowest is the Ray head) and write `run-spec.json`: image, node root, per-node
   verified model path.
5. `bootstrap_node.py`: on every node, `docker run` one idle container from the spec
   (`--user uid:gid --gpus all --network host --ipc host --ulimit nofile=65536:524288`, the IMEX
   channel device, `/work`, `/tmp` and the read-only model mounts, lifetime bounded by the Slurm
   end time, label `miles.run=<run id>`, and the environment the launcher relies on:
   `MASTER_ADDR=<head fabric IP>`, `MILES_NODE_EXTERNAL_IP=<this node's fabric IP>`,
   `NCCL_SOCKET_IFNAME==<fabric NIC>`, `GLOO_SOCKET_IFNAME=<fabric NIC>`, `NCCL_CUMEM_ENABLE=1`,
   `HOME=/work/home`, `TMPDIR=/tmp`).
6. `receive_key_node.py`: write the Daytona key from stdin into the head container's
   `/work/private/daytona.key` (exclusive create, `0600`, never echoed).
7. `rubin_node.py`: start Ray in each container (head or worker), pinned to the fabric interface.
8. `ray_ready.py`: wait until Ray reports every node and GPU.

`site/` is written for our cluster and assumes: Slurm, with `srun --overlap` into an existing
allocation; Docker on every node, usable by the ordinary user; the image already pulled on every
node; a node-local NVMe directory (`/raid/dldata`) with room for a 330 GB checkpoint copy and the
run; an NVL72 IMEX channel device owned by the run user; one fabric interface per node that
carries the IP the other nodes route to; outbound HTTPS from the nodes (GitHub, PyPI and the
Daytona API are reachable directly); and no mount of the login node's scratch directory on the
nodes except a read-only CIFS view under a service identity, so the run root has to be
world-readable to be read from there (`CIFS_VIEW`), otherwise `stage-node` pipes it through
`srun` stdin. Another cluster needs its own version of these steps; what it has to reproduce is
the container below.

### Layout

| On the node | In the container | Holds |
|---|---|---|
| `<node root>/work/` | `/work` | `tools/` (these scripts), `data/` (the task split), `inputs/` (patch, pins, manifest), `OpenEnv/` and `terminal-bench-2/` (checkouts at the pinned commits, cloned from GitHub by `stage-tools`), `output/<run id>/` (`dump_details`, logs), `offload/`, `cache/`, `ray/`, `home/`, and on the head `private/daytona.key` |
| `<node root>/tmp/` | `/tmp` | scratch |
| `<model cache>/` | `/inputs/GLM-5.3-Flash` | the original HF FP8 checkpoint, read-only, verified on every node against the manifest |
| `/opt/OpenEnv`, `/opt/Megatron-LM` | the same paths | copies made inside the container by `install_deps.py apply` |

The node root defaults to `/raid/dldata/<user>-<run id>` (`NODE_ROOT`). The Terminal-Bench-2
checkout keeps its GitHub `origin` because the Daytona sandbox build downloads each task
directory from that origin at the pinned commit.

### The container

What `bootstrap_node.py` starts on every node, written out as the equivalent `docker run`. This
is the contract another cluster has to reproduce:

```bash
docker run -d --name <run id> --hostname <node> --pull never --restart no \
  --user <uid>:<gid> --gpus all --network host --ipc host --workdir /opt/miles \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 --ulimit nofile=65536:524288 \
  --label miles.run=<run id> --device /dev/nvidia-caps-imex-channels/channel<N> \
  --mount type=bind,src=<node root>/work,dst=/work --mount type=bind,src=<node root>/tmp,dst=/tmp \
  --mount type=bind,src=<model cache>,dst=/inputs/GLM-5.3-Flash,readonly \
  --env HOME=/work/home --env TMPDIR=/tmp --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env SLURM_JOB_ID=<job> --env MASTER_ADDR=<head fabric IP> --env MILES_NODE_EXTERNAL_IP=<this node's fabric IP> \
  --env NCCL_SOCKET_IFNAME==<fabric NIC> --env GLOO_SOCKET_IFNAME=<fabric NIC> --env NCCL_CUMEM_ENABLE=1 \
  --entrypoint /bin/sleep <image> <seconds until the allocation ends>
```

The container runs as the ordinary user, not root; `install_deps.py plan` and `apply` are the
only steps run with `docker exec --user 0`. Ray is then started in every container by
`rubin_node.py` (`ray start --head` on the node with the lowest fabric IP, the others join it),
and `launch_train.sh` submits to that cluster from the head container.

## The Daytona key

Only the head container ever holds the key, as a file; it is never put in a command line, an
environment value or a log.

1. On the login node, keep the key in a `0600` file and point `KEY_STAGING` at it.
2. `bash site/prepare_run.sh key <jobid>` pipes that file into the head container, where
   `receive_key_node.py` creates `/work/private/daytona.key` (`0700` directory, `0600` file,
   owned by the run user) from stdin and prints nothing but a receipt.
3. In the head container, `launch_train.sh` exports `DAYTONA_API_KEY_FILE=/work/private/daytona.key`
   and `DAYTONA_API_URL=https://app.daytona.io/api` (its defaults). The launcher forwards the
   file **path** and the URL to the Ray job (`provision_provider` in
   `miles.rollout.agentic.credentials`), and `--pin-rollout-manager-to-head` keeps the processes
   that read the file on the head.
4. After the run's sandboxes are reconciled, `bash site/prepare_run.sh cleanup-key <jobid>
   [--also-staging]` shreds the file (and the login-node copy).

## Launch

Everything below runs in the head container, as the run user. Prerequisites: Ray is up on all
nodes (`site/ray_ready.py` confirms every node and GPU), the key file is in place, `/work/data`
holds `tbench2_train69.jsonl` and `tbench2_eval.jsonl`, and the model is mounted at
`/inputs/GLM-5.3-Flash`.

**The 20-step run** (eight nodes, evaluation off; the run in the figure above):

```bash
bash /work/tools/launch_train.sh train <run>-fig5 /work/data/tbench2_train69.jsonl /work/data/tbench2_eval.jsonl \
    --num-rollout 20 --eval-interval 0 2>&1 | tee /work/output/<run>-fig5.log
python3 /work/tools/fig5_collect.py --log /work/output/<run>-fig5.log \
    --out fig5-series.json --csv fig5-series.csv --png fig5.png
```

`launch_train.sh` checks the data file hashes, submits the Ray job and streams its log; that
log is what `fig5_collect.py` reads (the PNG needs `matplotlib`; JSON and CSV do not).

The launcher's training arguments: TP4 / PP4 / EP4 / CP1 with
layers 11/11/11/12, full uniform recompute, dynamic batching at 8192 tokens per GPU, optimizer
state streamed to local disk in FP32, two TP8 / EP8 FP8 rollout engines with Triton MoE,
TileLang DSA and BF16 KV cache, 8 prompts × 8 samples = 64 per update, 128 trajectories in
flight with `sample` submission granularity, temperature 0.8, 8192 tokens per turn, 65536
max sequence, 30 turns and 3600 s per episode, GRPO with Adam at 1e-6, clip 0.2/0.28, TIS 0–2,
`check_no_aborted`, strict weight comparison with the `visual.` exclusion, and
`NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` for the trainers. No checkpoints are saved.

**Variants and checks**

- *Training with evaluation*: drop `--eval-interval 0`. With the default `--eval-interval 10`
  and a short `--num-rollout` (we used 3), this runs the held-out evaluation before training and
  after the last update, and nothing in between.
- *Sampling check, no weight update* (two nodes, one TP8 engine): `launch_train.sh sample <run>
  <prompts.jsonl>` with `SAMPLE_NUM_ROLLOUT=1 SAMPLE_PROMPTS=1 SAMPLE_SAMPLES=1 SAMPLE_CONCURRENCY=1`
  (one episode) or `SAMPLE_PROMPTS=4 SAMPLE_SAMPLES=2 SAMPLE_CONCURRENCY=8`; the prompt file is a
  subset of the training file. `--debug-rollout-only` at this Miles revision still initializes
  Megatron's parallel state on the engine's eight GPUs, so the launcher states the stub layout
  explicitly (`--actor-num-nodes 1 --actor-num-gpus-per-node 8`, TP8/PP1 so a one-sample batch is
  valid). Inspect `dump_details/rollout_data/0.pt`: `response_length == len(loss_mask) ==
  len(rollout_log_probs)`, `metadata.accumulated_token_ids == tokens`,
  `metadata.tito_session_mismatch` empty or only `assistant_text` entries.
- *Sandbox check, no model* (head container; no GPUs or Ray needed): run one task's reference
  solution in a Daytona sandbox through the upstream `scan_golden.py`; the reward must be 1.0:

  ```bash
  mkdir -p /work/output/<run>-check
  OPENENV_SANDBOX_BACKEND=daytona DAYTONA_API_URL=https://app.daytona.io/api \
  DAYTONA_API_KEY_FILE=/work/private/daytona.key OPENENV_TB2_TASKS_DIR=/work/terminal-bench-2 \
  OPENENV_LAUNCHER=<you> OPENENV_RUN_ID=<run>-check \
  python3 /opt/miles/examples/experimental/openenv/scan_golden.py --tasks fix-git --concurrency 1 \
      --out /work/output/<run>-check/golden.jsonl
  ```

## After a run

1. Collect `dump_details` from every node (trainer ranks log on their own nodes).
2. Reconcile sandboxes: `python3 reconcile_sandboxes.py --run-id <run> --out <reconcile.json> --delete`. When the Ray
   job ends, the fully-async producer still has prefetched episodes in flight; their agent
   processes are killed before the provider can delete their sandboxes (98 were left after the
   20-step run, 5 after an 8-trajectory sampling run). Daytona's auto-stop / auto-delete
   backstop would reclaim them later, but reconcile first.
3. Remove the key file(s), then this run's containers (label `miles.run=<run-id>`).

## Known behaviour

- The TB2 tasks `qemu-startup` and `qemu-alpine-ssh` fail to build on Daytona at the recipe's
  first server layer (`apt-get update && apt-get install curl ca-certificates bash` does not
  complete in those task images). The provider raises, the episode aborts, `check_no_aborted`
  drops that prompt's group and the producer samples another prompt; the run continues. Their
  `BUILD_FAILED` sandboxes stay listed until deleted.
- At engine start SGLang logs a traceback from its post-warmup `freeze_gc` self-call and the
  trainers' first `destroy_weights_update_group` returns 400 before any group exists. Both are
  harmless.
- `Daytona.delete()` returns as soon as the request is accepted; a 404 check needs a short
  poll.
