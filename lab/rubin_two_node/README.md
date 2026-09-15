# Qwen3.5 on two Rubin nodes

This is a short adaptation of Miles `scripts/run_qwen3_5_35b_a3b_mtp.py` for
two nodes with four visible Rubin GPUs each. It configures the ordinary synchronous
`train.py` driver for two complete rollout / Megatron optimizer / weight-update
iterations. The second rollout therefore consumes weights from the first update.

Validation is in progress on job `2179787`. Registry recovery on both nodes and
the eight-GPU MNNVL communication check passed. Attempt 1 was intentionally
stopped during initial weight synchronization after Miles' default cuMem setting
selected Socket transport (about 4.1 seconds per weight bucket, with a 512 MiB
bucket limit). Its status is `STOPPED`; neither success nor a workload failure
was established.

Attempt 2 explicitly enables cuMem in both containers and the Ray job runtime.
The training NCCL communicators now report cross-node `P2P/MNNVL`, with no Socket
transport edges or NCCL warnings in the inspected logs. Megatron loaded the
checkpoint successfully. The head SGLang engine loaded its HF weights and
returned HTTP 200 from its health endpoint; the worker engine is still loading
from cold storage. Rollout and optimizer execution have not started.
See [RESULTS.md](RESULTS.md) for the verified results and durable evidence paths.

## Preserved images

The original core image and its Qwen3.5 dependency extension have both been
pushed to [Kaixi's GitLab registry](https://gitlab-master.nvidia.com/kaixih/my_docker_hub/container_registry).
Registry digests and image IDs were independently checked through the GitLab
API and are recorded in `registry-images.json`. Both images are `linux/arm64`.

```bash
docker pull gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:cu134-20260914-f187f76
docker pull gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:qwen35-cu134-20260914-f187f76
```

`Dockerfile.qwen35` uses the preserved core manifest digest as its `FROM` image,
so building it on a new node adds only the FLA dependency layer. It does not
rebuild Torch, TE, Apex, or FlashAttention. The repository's launcher is mounted
into the runtime containers by `orchestrate_rubin.py`.

```bash
docker build -f lab/rubin_two_node/Dockerfile.qwen35 -t miles-rubin:qwen35-incremental .
```

The two-node communication check must pass before launching training. Both
containers must use the same image and source tree, expose identical model/data
paths, and already belong to the same Ray cluster (eight GPUs total). Prepare
writable Ray, Triton, FlashInfer, temporary, and output directories on both nodes.

## Incremental image dependency

Keep the native cu134 Torch, Triton, CUTLASS, FlashInfer, and TE stack. Add both
published FLA packages; PyPI splits the kernels/modules into `fla-core`, although
the repository root pyproject does not show that release-time split:

```dockerfile
RUN python3 -m pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
RUN FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton python3 -c \
    'from fla.modules import FusedRMSNormGated, ShortConvolution; from fla.ops.gated_delta_rule import chunk_gated_delta_rule'
```

These are Python/Triton packages. FLA's CUDA extra declares Torch >=2.7 and
Triton >=3.3; the cu134 image satisfies those version requirements. That is not
an SM107 execution guarantee. No extra TileLang, FlashQLA, causal-conv1d,
mamba-ssm, or Megatron-Bridge install is needed for the selected path. Existing
TileLang remains installed; `FLA_DISABLE_BACKEND_DISPATCH=1` selects FLA's default
Triton implementations, and `FLA_CONV_BACKEND=triton` selects its bundled convolution.

Published wheel SHA256 values:

- `fla_core-0.5.2-py3-none-any.whl`: `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761`
- `flash_linear_attention-0.5.2-py3-none-any.whl`: `dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400`

Sources: [FLA wheel metadata](https://pypi.org/pypi/flash-linear-attention/0.5.2/json),
[core wheel metadata](https://pypi.org/pypi/fla-core/0.5.2/json),
[Triton convolution default](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/conv/short_conv.py),
[backend dispatch switch](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/backends/__init__.py).

## Existing inputs

The following paths were found under `/home/scratch.kaixih_ent/models` on dl3:

- `Qwen3.5-35B-A3B/`: HF config, tokenizer, index, and fourteen weight shards.
- `Qwen3.5-35B-A3B_torch_dist/`: existing `release` checkpoint and tracker.
- `dapo-math-17k/dapo-math-17k.jsonl`: existing rollout data.

Mount that directory read-only at the same path on both nodes. The launcher
does not call the original example's unconditional download/convert `prepare`.
It loads the existing converted checkpoint with the raw Miles model plugin.

## Inspect and run

From the Miles checkout in the prepared head container, inspect the command first:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python3 lab/rubin_two_node/run_qwen3_5_35b_a3b_rubin.py \
  --model-dir /home/scratch.kaixih_ent/models \
  --data-dir /home/scratch.kaixih_ent/models \
  --output-dir /run-output --megatron-path /opt/Megatron-LM --print-only
```

After the communication check and external Ray setup succeed, run the same command
without `--print-only`. Set `MASTER_ADDR` to the head IP, `RAY_ADDRESS` to its GCS
address (`IP:26379`), and `RAY_API_SERVER_ADDRESS` to its dashboard URL
(`http://IP:28265`). Paths such as `/run-output` are mount points chosen by the cluster
launcher, not directories this script prepares. Network/NCCL settings should
match the successful communication check and be present in both Ray containers;
use `--extra-env-vars` when worker runtime overrides are needed.

Preserve `NCCL_CUMEM_ENABLE=1` and `NCCL_NVLS_ENABLE=0` in **both the container
environment and the Ray job runtime environment**, as the current orchestrator
and training launcher do. The standalone communication check alone does not
establish the transport used by Miles: its trainer defaults cuMem to zero when
unset, while the two single-node TP4 SGLang engines do not trigger SGLang's
automatic multi-node cuMem setup. [NCCL requires cuMem for MNNVL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-mnnvl-enable).

This setting keeps the existing colocate/offload behavior and requires no TMS
patch. The pinned TMS preload intercepts `cudaMalloc/cudaFree`, so NCCL's
`cuMem*` allocations bypass its allocation tracking; Miles also destroys its
reloadable NCCL process groups before pausing training memory.
See the [TMS hooks](https://github.com/fzyzcjy/torch_memory_saver/blob/f05a8754daf68238d54e4cf31cb3ba866684bbaf/csrc/entrypoint.cpp#L55).
Complete offload/onload and training continuity still require the full run.

## Changes from the existing example

| Area | Short-run setting |
| --- | --- |
| Training | TP2, PP1, CP1, EP8, expert TP1; eight GPUs total |
| Rollout | Two TP4 engines, each within one node; SGLang EP1 |
| Work | Two rollouts, eight prompts × two samples, global batch 16 |
| Token budget | Prompt <=1024, response <=256, 2048 training tokens/GPU |
| Precision | BF16; TE fused/cuDNN softmax attention; FLA Triton GDN |
| SGLang | Triton softmax attention, GDN prefill/decode and MoE; Torch BF16 GEMM |
| Communication | Standard all-to-all MoE, cuMem enabled for MNNVL, NVLS disabled, Ray object store, Miles Python router |
| Optimizer | GPU Adam; no CPU optimizer offload or precision-aware optimizer |
| Optional paths | MTP layers/training/speculation, CP, DeepEP/flex, CUDA graphs, eval, saves, reference KL pass disabled |

`--mtp-num-layers 0` overrides the existing model registry's one-layer MTP
setting. The converted checkpoint successfully loaded and resharded
from TP1/PP8 to TP2/PP1. The launcher does not change checkpoint structure or
re-convert it. The original actor/rollout memory offload
schedule under `--colocate` is retained, as are real backward/optimizer and weight
synchronization operations. W&B is off by default and can be enabled with
`--enable-wandb` using the existing Miles helper.

Short responses may receive zero math reward; this run establishes execution,
not learning quality. Success requires both rollout IDs to finish generation,
finite training metrics and optimizer steps, and weight updates after each step,
with the Ray job exiting successfully. An import test or a successful first
generation alone is insufficient.

The pinned SGLang hybrid-GDN guard treats SM major 10 as Blackwell and rejects
the `flashinfer` full-attention backend for this model. This recipe therefore
uses `triton`; `--sglang-bf16-gemm-backend torch` also avoids its automatic SM100
CuTe GEMM selection on Rubin. TP4 is allowed with two KV heads: SGLang replicates
each KV head across two ranks.

## One container per node

`orchestrate_rubin.py` is a standard-library host utility run on dl3, outside the
Miles environment. It verifies the existing Slurm allocation, uses the normal
SSH user's UID/GID, probes container write access and normal-user deletion, and
starts Ray on two persistent containers. It uses the MNNVL device mounts and
privileged Docker flags from the separate communication setup. It never creates
an allocation or starts training as part of `bootstrap`.

Defaults target job 2179787, head/worker IPs 10.102.74.84/.85, Ray port 26379,
dashboard port 28265, the pinned Qwen3.5 manifest digest, the selected shared run root,
and node-local `/tmp/miles-rubin-j2179787/qwen35` caches. These nodes have writable
local `/tmp`; the script does not require changing `/raid` permissions. Supply
the complete allocated node names, as reported by `scontrol show hostnames`:

```bash
python3 lab/rubin_two_node/orchestrate_rubin.py plan --nodes NODE_C17 NODE_C18
python3 lab/rubin_two_node/orchestrate_rubin.py bootstrap --nodes NODE_C17 NODE_C18
python3 lab/rubin_two_node/orchestrate_rubin.py status --nodes NODE_C17 NODE_C18
python3 lab/rubin_two_node/orchestrate_rubin.py train --nodes NODE_C17 NODE_C18 \
  --gate-log /path/to/successful-eight-rank-mnnvl.log
```

Replace both node placeholders and the log path. `train` requires `world=8`,
`P2P/MNNVL`, and `ALLREDUCE_OK` in the supplied communication log, rechecks the
allocation and Ray's two four-GPU nodes, and streams output into the shared
run's `logs/qwen35_train.log`. It refuses to overwrite an existing training log.
Bootstrap refuses to replace existing named containers. It retains
`bootstrap.json` and `train_exit.json`; node-local caches are disposable and no
automatic cleanup is scheduled. Preserve communication-test NCCL settings using
`--env KEY=VALUE` when its settings differ from the defaults.

For another allocation, explicitly override `--job-id`, `--nodes`, `--node-ips`,
`--run-dir`, `--cache-dir`, and `--container-prefix` on every command. These
defaults are independent: changing the job ID does not update the other paths
or container names. Use a fresh output directory and a communication log from
the intended nodes; the script checks the three markers but does not match the
log's allocation or image identity. The source tree is a read-only bind mount,
so preserve its Git commit and any uncommitted recipe changes with the run's
evidence, in addition to the container digest.

Ray advertises 32 logical CPUs per node (`--num-cpus`, configurable), and worker
ports span 26400–26999. The first bootstrap used 352 detected CPUs and only 200
worker ports; its eagerly started workers exhausted that range and reported
`No available ports`. Each status probe now has a 40-second in-container timeout
(plus a five-second termination grace period); bootstrap retains its overall
120-second retry deadline.

Local validation: Python syntax checks and an isolated recording of the resolved
training argv passed; the recording used stubbed command execution and did not
start Ray or GPUs. The image also passed `train.py --help` and the launcher's
`--print-only` check. GPU execution is now in progress as described above; the
full Miles launcher test suite has not been run for this recipe.

`launch_plan.json`, if present in a working copy, is an early review artifact,
not the execution record: it omits the cluster environment and source identity.
Regenerate the launcher's `--print-only` output in the pinned image for each
final configuration and retain it with the run logs. Its output starts with a
human-readable argument table before the JSON, so do not parse the entire
stdout as a JSON document. The source launcher and orchestrator are the recipe;
`bootstrap.json`, the actual launch log, and `train_exit.json` are run evidence.
