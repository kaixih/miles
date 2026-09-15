# Qwen3.5 on two Rubin nodes

This is a short adaptation of Miles `scripts/run_qwen3_5_35b_a3b_mtp.py` for
two nodes with four visible Rubin GPUs each. It configures the ordinary synchronous
`train.py` driver for two complete rollout / Megatron optimizer / weight-update
iterations. The second rollout therefore consumes weights from the first update.

**Learning validation PASS:** on 2026-09-15, the two-node/eight-SM10.7-GPU run
completed two real DAPO GRPO updates. The filtered batches had raw reward means
0.75 and 0.50; gradient norms were 0.5861834 and 0.6526616. Both updates changed
served parameter values, both TP4 rollout engines had identical weights after
every synchronization, and rollout 1 used the updated weights. All eight ranks
completed both optimizer steps and three weight synchronizations. Ray and the
independent launcher exits confirmed success.

See [learning-validation-summary.json](learning-validation-summary.json) and
[RESULTS.md](RESULTS.md) for the 32/32 independent reward regrades, source/image
identities, parameter hashes and retained earlier failures. This validates the
BF16 TP2/PP1/CP1/EP8 FA2/Triton recipe with colocate/offload over cross-node MNNVL.
The accepted samples were 50% and 75% truncated at 4096 response tokens; this is
a two-update functional check, not a model-quality or convergence result.

The launcher's default 256-token configuration remains an earlier **execution
smoke PASS** with zero rewards and gradients. Its separate result is preserved
in [validation-summary.json](validation-summary.json). Use the non-thinking,
`math`-scored configuration below to reproduce the nonzero training signal.

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

| Area | Default execution-smoke setting |
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

### Reproducing the validated nonzero training signal

The default 256-token smoke run does not establish learning. A later 4096-token
thinking run also exhausted its sampling budget without accepting a mixed-reward
group. Full auxiliary responses showed coherent reasoning, but were truncated;
this did not establish a numerical correctness failure. The existing GB200/GB300
self-distillation example uses a much longer DAPO reasoning budget.

For a short check with the same DAPO data, disable thinking and use Miles' existing
`math` scorer, which compares the boxed answer with the original label:

```bash
--rollout-batch-size 4 --n-samples-per-prompt 4 \
--rollout-max-response-len 4096 --rollout-max-prompt-len 1024 \
--no-enable-thinking --rm-type math \
--rollout-temperature 0.7 --rollout-top-p 0.8 --rollout-top-k 20 \
--max-tokens-per-gpu 8192 \
--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter \
--save-debug-event-data /run-output/events --save-local-weight-checksum \
--save-debug-rollout-data '/run-output/rollout_data/{rollout_id}.pt' \
--save-debug-trajectory-data '/run-output/trajectories/{rollout_id}.jsonl'
```

Append these arguments to the in-container launcher, or pass their shell-quoted
form through the host orchestrator's `--launcher-args`. This keeps a global batch
of 16 while accepting only prompt groups whose actual rewards differ. The filter
can resample indefinitely; use a bounded run and stop its specific Ray job if the
budget expires. The selected scorer uses actual answer correctness; no labels or
rewards are injected. The default `deepscaler` scorer requires `</think>` within
the generated response, so it is incompatible with this non-thinking check.
These temperature/top-p/top-k values follow the model card's non-thinking general
preset; Miles does not expose its presence-penalty setting in this rollout path,
so this is not an exact copy of every sampling parameter. This mode validates
short-run training mechanics and does not validate long-context thinking.

For a GSM8K comparison, keep the non-thinking recipe above and override only
the dataset and response limit, while explicitly preserving its 5120-token
rollout/SGLang context capacity:

```bash
--prompt-data-path /path/to/gsm8k.jsonl \
--rollout-max-response-len 1024 --rollout-max-context-len 5120
```

The dataset must retain the same `prompt` and `label` fields. An empty
`--prompt-data-path` keeps the DAPO path under `--data-dir`. The context option
defaults to 0, which derives prompt plus response limits; an explicit value
must cover their sum. Other model, sampling, filter and iteration settings
remain those supplied in the original command.

Prepare the audit directories with the container writer's UID/GID before launch.
The `.pt` rollout dumps are the authoritative single-turn sample records; this
run did not emit separate trajectory JSONL files. They contain samples, not model
checkpoints. Local checksum collection requires the event directory and adds
tensor copies to CPU. It hashes local model parameters and optimizer state; it
does not directly audit FP32 master-parameter values.

Use runtime commit `e05d2549cc1027a9c229b40253e1a7b059e38968` or a descendant
containing its native-FP32 checksum fix. Qwen3.5's FP32 `linear_attn.A_log` has
no separate master parameter in the pinned distributed optimizer; the diagnostic
must map its actual owned optimizer shard. This fixes the checksum hook without
changing training math or native libraries. All 26 checksum tests passed in the
preserved image. The audited run used a recording wrapper around the unchanged
original filter; its exact hash and all 39 recorded decisions are referenced in
the learning summary and retained experiment directory.

Require finite nonzero gradient norms, mixed rewards recomputed from the saved
samples, and matching parameter checksums across both rollout engines after each
synchronization. Compare the initial served parameters with both post-training
versions to establish actual BF16 parameter changes. A centered GRPO loss can be
near zero even with nonzero gradients. These checks complement the execution
criteria above; two iterations alone do not establish convergence or model quality.

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
the longer sampling/audit flags also passed the real CLI check in the same image.
The full launcher suite was subsequently run on 2026-09-15: all 43 fast tests
passed. A CPU-only rerun with a writable temporary `/root` produced 488 passed,
12 failed, and 4 errors. Remaining failures concern unchanged AMD/Kimi snapshots,
AMD launchers calling the absent `U.exec_command`, and legacy shell recipes
requiring unavailable `envsubst`. The first run also had 36 permission errors
from test launchers writing `/root/models`; those disappeared in the isolated
rerun. No framework packages or snapshots were changed to make these tests pass.

`launch_plan.json`, if present in a working copy, is an early review artifact,
not the execution record: it omits the cluster environment and source identity.
Regenerate the launcher's `--print-only` output in the pinned image for each
final configuration and retain it with the run logs. Its output starts with a
human-readable argument table before the JSON, so do not parse the entire
stdout as a JSON document. The source launcher and orchestrator are the recipe;
`bootstrap.json`, the actual launch log, and `train_exit.json` are run evidence.
