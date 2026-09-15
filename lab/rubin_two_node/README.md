# Qwen3.5 on two Rubin nodes

This is a short adaptation of Miles `scripts/run_qwen3_5_35b_a3b_mtp.py` for
two nodes with four visible Rubin GPUs each. It configures the ordinary synchronous
`train.py` driver for two complete rollout / Megatron optimizer / weight-update
iterations. The second rollout therefore consumes weights from the first update.

**PASS:** attempt 3 completed both iterations on eight SM10.7 GPUs across two
nodes, confirmed at 2026-09-15 01:12:12 UTC. Both rollouts generated 16 samples; all eight
ranks completed a valid optimizer step in each iteration and all three weight
synchronizations. Ray reported `SUCCEEDED`, and both launcher exit records were
zero. This validates BF16 training with TP2/PP1/CP1/EP8, two TP4 SGLang engines,
and the existing colocate/offload schedule over cross-node MNNVL.

The passing recipe uses FlashAttention 2 through TE with the scoped SM107 patch
below. Rollouts 0 and 1 took 23.8681 and 10.7653 seconds, using weight versions 1
and 2. All responses hit the 256-token limit; rewards and gradient norms were
zero. This establishes execution, without claiming learning or parameter-value
changes. See [validation-summary.json](validation-summary.json) for the compact
result and [RESULTS.md](RESULTS.md) for proof, runtime source/image identities,
and the retained earlier failures.

## Preserved images

The core image, original FLA extension, and current FLA/FA2 extension are preserved
in [Kaixi's GitLab registry](https://gitlab-master.nvidia.com/kaixih/my_docker_hub/container_registry).
Registry digests and image IDs were independently checked through the GitLab
API and are recorded in `registry-images.json`. All images are `linux/arm64`;
the current image was pulled successfully on both nodes.

```bash
docker pull gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:cu134-20260914-f187f76
docker pull gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:qwen35-fa2-cu134-20260915@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c
```

`Dockerfile.qwen35` defaults to the preserved core digest, adding the two FLA
packages and the TE Python backend-selection patch. To reproduce the latest
incremental build, reuse the preserved FLA-only image as `BASE_IMAGE`:

```bash
docker build -f lab/rubin_two_node/Dockerfile.qwen35 \
  --build-arg BASE_IMAGE=gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:676354cc71c6f5d4fbb4fd5bf246cfd4a33925932a5d4171b6f6969bf7a6886a \
  -t miles-rubin:qwen35-fa2-cu134-20260915 .
```

Omit `--build-arg BASE_IMAGE=...` to build from the core image instead. Neither
route recompiles native libraries; all 40 protected native package versions and
the Torch ABI remain unchanged. `orchestrate_rubin.py` defaults to the current
manifest digest and mounts the repository launcher into the runtime containers.

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

## TE FlashAttention 2 compatibility

`patch_te_sm107_fa2.py` adjusts TE 2.19's Python head-dimension allowlist only for
SM10.7, BF16, Q/K/V head dimension 256, and zero attention dropout. It checks the
TE version and exact source anchor and leaves all other backend checks intact.
The native FA2 2.8.3.post1 kernels were already built for SM107. The launcher uses
`--attention-backend flash`, which makes Megatron enable TE FlashAttention and
disable its fused/cuDNN and unfused alternatives; the flag alone cannot bypass
the original allowlist.

Direct FA2 and TE-wrapped causal THD attention passed output and Q/K/V gradient
comparisons against Torch FP32 math for GQA 8:1, d256, and sequence lengths
`[512]`, `[256, 512]`, and `[257, 385, 129]`. Relative L2 errors were below 0.31%.
The same checks passed in the newly built image.

The current CP1 Miles pack has no gaps between sequences: `*_padded` boundaries
are `None`, and trailing alignment padding is represented as a separate dummy
sequence in `cu_seqlens`. TE infers `pad_between_seqs=False`, so FA2 remains
eligible. An explicit gap-padding probe still returns `NoBackend`; that boundary
is intentionally unsupported by this recipe and does not broaden the patch's
validated scope.

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
The completed two-iteration run exercised this offload/onload schedule.

## Changes from the existing example

| Area | Short-run setting |
| --- | --- |
| Training | TP2, PP1, CP1, EP8, expert TP1; eight GPUs total |
| Rollout | Two TP4 engines, each within one node; SGLang EP1 |
| Work | Two rollouts, eight prompts × two samples, global batch 16 |
| Token budget | Prompt <=1024, response <=256, 2048 training tokens/GPU |
| Precision | BF16; TE FlashAttention 2 with the scoped SM107 patch; FLA Triton GDN |
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

For a rerun, require both rollout IDs to finish generation, finite training
metrics and valid optimizer steps on every rank, all three weight
synchronizations, and Ray `SUCCEEDED` plus launcher exit 0. Weight version
increments alone do not establish that parameter values changed.

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
dashboard port 28265, the pinned Qwen3.5 FA2 manifest digest, the selected shared run root,
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
ports span 26400–26999 to accommodate worker startup. Each status probe has a
40-second in-container timeout
(plus a five-second termination grace period); bootstrap retains its overall
120-second retry deadline.

Local validation: Python syntax checks and an isolated recording of the resolved
training argv passed; the recording used stubbed command execution and did not
start Ray or GPUs. The image also passed `train.py --help` and the launcher's
`--print-only` check. The full two-node GPU run passed as recorded in RESULTS;
the full Miles launcher test suite has not been run for this recipe.

`launch_plan.json`, if present in a working copy, is an early review artifact,
not the execution record: it omits the cluster environment and source identity.
Regenerate the launcher's `--print-only` output in the pinned image for each
final configuration and retain it with the run logs. Its output starts with a
human-readable argument table before the JSON, so do not parse the entire
stdout as a JSON document. The source launcher and orchestrator are the recipe;
`bootstrap.json`, the actual launch log, and `train_exit.json` are run evidence.
