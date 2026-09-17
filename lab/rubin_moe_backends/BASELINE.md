# Rubin BF16 MoE: two-rollout baseline

Compare each candidate from the original Qwen3-30B-A3B HF/release checkpoint,
using the same GSM8K data and an independent fresh optimizer. Keep initial
evaluation enabled, as in the retained Triton run. This is an execution and
weight-update check plus preliminary timing, not a convergence comparison.

The baseline image is
`gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`
(FlashInfer 0.6.18). See the retained [launch command](../../outputs/rubin-gb300-qwen3-cudagraph/rubin/train-launch.json)
and [resolved metadata](../../outputs/rubin-gb300-qwen3-cudagraph/rubin/metadata.json).

Preserve one four-GPU node, four TP1/EP1 rollout engines, BF16, Triton attention,
Torch BF16 GEMM, decode CUDA Graph ON, prefill graph OFF, static memory fraction
0.55 and 128 running requests per engine. Each rollout has 256 prompts × 8
responses, global training batch size 512, and four optimizer updates. Training
and log-prob token budgets are 4,096/GPU; prompt/response/context limits are
512/1,024/1,536 tokens. Train sampling is T=1, top-p=1, top-k=-1; initial
evaluation uses the same fixed 256 examples with T=1, top-p=0.7, top-k=-1.
Training seed is 1234; rollout seed is 42.

Use `--num-rollout 2 --save-interval 0 --save-retain-interval 0` and omit the
save sentinel. A positive save interval also saves the last rollout, even in a
short run. Use `--sglang-moe-runner-backend flashinfer_cutlass` or
`--sglang-moe-runner-backend flashinfer_trtllm`; the default remains `triton`.

| Recorded Triton metric | Rollout 0 | Rollout 1 |
| --- | ---: | ---: |
| Generated response tokens | 1,624,386 | 1,618,889 |
| Generation time (s) | 63.898965 | 61.628560 |
| Response tokens/GPU/s | 6,355.291 | 6,567.122 |
| Mean response length | 793.1572 | 790.4731 |
| Raw reward | 44.8242% | 50.8789% |
| Truncation rate | 36.1816% | 36.9141% |
| Weight version | 1 | 2 |
| Mixed weight-version ratio | 0 | 0 |
| Actor train time (s) | 112.9649 | 83.6206 |
| Total step time (s) | 351.3220 | 210.9610 |

Source: retained [qwen3_train.log](../../outputs/rubin-gb300-qwen3-cudagraph/rubin/logs/qwen3_train.log),
generation records at lines 15788 and 29567, stage timers at lines 18527 and
31037. Total generated tokens are the exact logged mean response length ×
2,048 responses. Rollout 0 includes warmup; prioritize rollout 1 generation
time and token-normalized throughput, and report output lengths, rewards and
truncation alongside them. The full 50-rollout average is not the baseline for
these initial two rollouts. Identical seeds do not guarantee identical outputs
across kernels or scheduling. Successful weight-version advancement, no mixed
versions, finite nonzero gradients and a completed second rollout are required
execution evidence; none alone proves full training correctness.

Container model paths are
`/home/scratch.kaixih_ent/models/Qwen3-30B-A3B` and
`/home/scratch.kaixih_ent/models/Qwen3-30B-A3B_torch_dist`.
Data is mounted as `/run-output/inputs/train.jsonl` and
`/run-output/inputs/test-fixed-256.jsonl`; verify their SHA256 values against the
retained metadata before launching. Node-local source locations are allocation
specific and must be checked again.
