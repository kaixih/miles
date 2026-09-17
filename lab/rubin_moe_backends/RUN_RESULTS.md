# Rubin BF16 MoE: two-rollout results

Qwen3-30B-A3B/GSM8K, one four-GPU Rubin SM107 node, BF16, four TP1 rollout engines. Decode CUDA Graph ON; prefill graph OFF. Each backend starts from the original weights: 256 prompts × 8 responses per rollout, global training batch 512, four optimizer updates per rollout. Other settings match the retained Triton run.

**CUTLASS and corrected TRTLLM both completed two rollouts and eight optimizer updates. Generation improved; the corrected TRTLLM full training step did not.**

| Backend | Generation 0 (s) | Generation 1 (s) | Rollout 1 response tokens/GPU/s | Rollout 1 reward |
| --- | ---: | ---: | ---: | ---: |
| Triton (historical) | 63.90 | 61.63 | 6,567.12 | 50.88% |
| FlashInfer CUTLASS | 57.26 | 53.23 | 7,625.08 | 51.03% |
| FlashInfer TRTLLM + both fixes | 48.48 | 47.38 | 8,568.60 | 50.73% |

For rollout 1, CUTLASS reduces generation time by **13.6%** and increases token throughput by **16.1%** versus Triton. Corrected TRTLLM reduces generation time by **23.1%** and increases throughput by **30.5%** versus Triton; versus CUTLASS, it reduces generation time by **11.0%**. These are complete generation measurements, not isolated MoE kernel timings.

## Full-step limitation

| Backend, rollout 1 | Actor update (s) | Weight sync (s) | Full step (s) |
| --- | ---: | ---: | ---: |
| Triton (historical) | 83.62 | 3.67 | 210.96 |
| FlashInfer CUTLASS | 83.95 | 3.70 | 204.10 |
| FlashInfer TRTLLM + both fixes | 114.32 | 7.20 | 233.10 |

The TRTLLM actor slowdown is present in all four updates (~27.6–29.7 s each versus ~20–21.7 s). All three runs scheduled the same microbatch counts (27/29/27/29); CUTLASS and TRTLLM total mean sequence lengths differ by only 0.029%. Neither logs a new compilation warning in rollout 1. Existing logs cannot identify the CPU, GPU, communication, or routing cause. No full-training speedup is claimed.

## Fix and validation

- Applying [PR #33743](https://github.com/sgl-project/sglang/pull/33743) alone failed during the initial weight sync. Its bucket hook already repacked the weights; Miles then invoked final post-load and packed them again, causing the FlashInfer 2D/3D assertion.
- The [separate post-load patch](followup_idempotent_postload/idempotent-postload.patch) reuses the existing shape-gated repack method. Its build reproduces the original failure, passes four tests using real FlashInfer layout functions plus all 11 upstream tests, and verifies unchanged native libraries and Torch ABI.
- Both successful GPU runs have Ray SUCCEEDED, eight finite-loss/nonzero-gradient updates, all 32 rank-step outcomes NORMAL, clean rollout weight versions 1/2 and final synchronization version 3. Four decode graph captures and replay are verified. Training/rollout log-probability differences remain about 0.02.
- Recovered startup connection errors, one first-rollout generation retry per candidate, and teardown Inductor warnings are retained. Prioritize rollout 1. Triton is historical on another Rubin node; two rollouts check execution and early numerical sanity, not convergence equivalence.

## Preserved container

[GitLab registry (internal)](https://gitlab-master.nvidia.com/kaixih/my_docker_hub/container_registry)

```bash
docker pull gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:experimental-moe-pr33743-cu134-20260916-j2211281-postload-r1
```

Registry manifest digest: `sha256:405315ba3add773be16cfe176a4dcd15a1071cbf17f3be255ab2c47324c65f34`. The remote manifest was read back and its config matches the tested image `sha256:448a37a80616c92c5674489585ae4c3a3edd981d6d442bea121676e0dec3f4ad`.

See [build instructions](followup_idempotent_postload/README.md), [experiment runner](README.md), and [machine-readable validation](validation-summary.json). HTML slides were not modified.
