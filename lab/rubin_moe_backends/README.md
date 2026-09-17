# Rubin BF16 MoE: two-rollout validation

See [completed results](RUN_RESULTS.md) for generation gains, the full-step
timing limitation, and the verified internal container reference.

This experiment runs Qwen3-30B-A3B/GSM8K with `flashinfer_cutlass`, then
`flashinfer_trtllm` plus [SGLang PR #33743](https://github.com/sgl-project/sglang/pull/33743).
Each backend starts from the original HF/release checkpoint with a fresh optimizer.
The second rollout exercises generation after a training weight update. The
experiment checks execution and early numerical sanity and collects preliminary
generation timing; it does not establish convergence or an isolated kernel speedup.

## Configuration and pinned sources

Both candidates use one four-GPU SM107 node, four TP1/EP1 rollout engines, BF16,
Triton attention, Torch BF16 GEMM, decode CUDA Graph ON and prefill graph OFF.
Each of two rollouts generates 256 prompts × 8 responses and performs four
optimizer updates at global batch size 512. Prompt/response/context limits are
512/1,024/1,536 tokens; training and log-prob budgets are 4,096 tokens/GPU.
Initial evaluation uses the same fixed 256-question subset. Checkpoint saving,
debug tensor dumps and W&B are disabled. See [BASELINE.md](BASELINE.md) for the
sampling, memory settings, seeds and historical Triton measurements.

- Base image for CUTLASS and the TRTLLM derivative: `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c` (internal; FlashInfer 0.6.18).
- Existing SGLang source: `aea7fb92c047c9c096eae66460acb25dec9ae5a9`.
- PR #33743 head: `89f25a0b9772576fbd427a495a1144f544c90928`.
- Pinned patch SHA256: `2137ce6f3ec431e70f0ffd4e7e876070255a3e9f1488ece1c5346b1111fa908d`.
- Miles run source comes from the retained baseline at `e73d8d68d65b467410113a0d91a879b77f6d1351`, with the launcher extended to select either MoE backend. The run's `source-manifest.json` records the staged files.

The derivative changes only the Python hot-update layout handling: restore
canonical expert data, copy bucketed updates, then repack the TRTLLM layout.
The build verifies the original source blobs, applies the pinned patch, runs
all 11 upstream CPU tests and checks that package metadata, native binaries and
Torch ABI remain unchanged. It does not rebuild CUDA kernels. Build instructions
and retained in-image provenance are in [README.trtllm-refit.md](README.trtllm-refit.md);
patch provenance is in [pr33743/README.md](pr33743/README.md).

The exact upstream PR alone failed at the first Miles weight synchronization:
the final post-load hook packed already blocked weights a second time. The
separate [idempotent post-load follow-up](followup_idempotent_postload/README.md)
fixes that lifecycle case. Its build reproduces the original failure with real
FlashInfer layout functions, then requires four full-hook regression tests and
the original 11 tests to pass, with native libraries unchanged.

## Run both backends

Use an existing single-node/four-GPU allocation requested with `salloc_node`.
The driver runs on dl3 as UID 28644/GID 30; it does not allocate or cancel jobs.
Prepare a fresh run directory containing `source/`, `build-context/`, the three
driver/summary scripts, `triton-baseline-first2.json`, and the two input JSONL
files under `inputs/`. Models must already exist under
`/home/scratch.kaixih_ent/models`. Input hashes are enforced by the runner.

The recorded invocation for allocation 2211281 is:

```bash
MILES_MOE_ROOT=/home/scratch.kaixih_ent/repro/miles-rubin-moe-backends/20260916-j2211281
python3 -u "$MILES_MOE_ROOT/run_and_summarize.py" \
  --root "$MILES_MOE_ROOT" --job-id 2211281
```

This is the existing run's invocation, not a resume command: its lock prevents a
second driver. A repeat requires a new prepared directory and its own allocation
ID. To invoke the sequential runner directly, the equivalent arguments are:

```bash
python3 -u "$MILES_MOE_ROOT/run_two_backends.py" \
  --job-id 2211281 --root "$MILES_MOE_ROOT" \
  --source "$MILES_MOE_ROOT/source" \
  --build-context "$MILES_MOE_ROOT/build-context"
```

The runner stages models on verified local disk and finishes bulk copies before
timed work. It starts a fresh Docker/Ray environment and checks four-rank NCCL
for each candidate. Both use the existing Qwen3 launcher with
`--num-rollout 2 --sglang-enable-cuda-graph --max-tokens-per-gpu 4096
--save-interval 0 --save-retain-interval 0`; the backend selector is
`--sglang-moe-runner-backend flashinfer_cutlass` for the base image and
`--sglang-moe-runner-backend flashinfer_trtllm` for the derivative image.
Other baseline settings are supplied by the runner and launcher.

Containers and caches are separate, and the first container is stopped and
removed before the second starts. The driver stops work 15 minutes before the
allocation ends, with bounded commands and identity-checked cleanup. A TRTLLM
image that passes the run checks is pushed to an experimental tag in the same
internal GitLab registry; `registry-push.json` records success or failure.

`retry_trtllm.py` runs the corrected image in a separate output/cache namespace
after the original attempt is terminal. It verifies in-image regression
provenance, reuses the checked local model files, and preserves the original
allocation deadline. The original failed attempt is retained. Supply the
immutable corrected image ID, a fresh `--root`, the original `--parent-root`,
and a unique `--registry-tag` under the same registry repository.

## Results and validation

Each backend directory retains `logs/qwen3_train.log`, `runtime-provenance.json`,
`allreduce.log`, `ray-terminal.json`, `metrics.json` and `result.json`. Success
requires both rollout/stage records, Ray success, weight versions 1 then 2 with
no mixed versions, eight finite-loss/nonzero-gradient updates, all 32 normal
rank-update outcomes, the requested backend in SGLang's effective configuration,
and decode graph capture/replay log evidence.

The run root retains `state.json`, hardware/storage/input/source provenance,
the derivative build log/image identity, and `ops/commands.jsonl`. The outer
driver produces `comparison.json`, `RESULTS.md` and `driver-exit.json`. The
offline summary can also be regenerated without touching GPUs:

```bash
python3 -B "$MILES_MOE_ROOT/summarize_results.py" \
  --root "$MILES_MOE_ROOT" --source "$MILES_MOE_ROOT/source" \
  --baseline "$MILES_MOE_ROOT/triton-baseline-first2.json"
```

Compare against the historical Triton run's first two rollouts, prioritizing
rollout 1 after warmup. Report generated tokens, tokens/GPU/s, reward and
truncation alongside generation time: sampling can change the amount of work.
The Triton reference used a different allocation; two samples and healthy
training signals do not prove identical numerics, convergence, or a causal
kernel-level performance benefit. Final candidate performance is intentionally
left to the completed run receipts and summary.

When summarizing the corrected attempt, add `--trtllm-retry-root` to the summary
command. This reports the original PR failure and the additional fix separately.
