# Qwen3 GSM8K: one Rubin node versus one GB300 node

The comparison uses **Miles/SGLang on both platforms**, one node and four GPUs
per run. The historical GB300 **verl/vLLM** run is a learning-curve reference,
not the Miles hardware-performance baseline. New training results are pending.

## Common learning recipe

| Setting | Value |
| --- | --- |
| Model | Qwen3-30B-A3B, standard post-trained model, not Base |
| HF revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Data | Original GSM8K train prompts/labels; single user message requesting `####` |
| Training | 50 rollouts, 256 prompts × 8 responses each |
| Optimizer batch | 512 responses; 4 optimizer updates per rollout, 200 total |
| Length | Prompt ≤512, response ≤1024; default Qwen3 thinking template |
| Sampling | Temperature 1, top-p 1, top-k disabled; no reward-group filtering |
| Algorithm | GRPO, token-mean loss, clip 0.2/0.28, dual clip 10 |
| Optimizer | Adam lr 1e-6, betas 0.9/0.999, epsilon 1e-8, weight decay 0.1, grad clip 1 |
| Reference KL | Loss coefficient 0.001, low-var estimator; no KL reward penalty |
| Training layout | TP1 / PP1 / CP1 / EP4, expert TP1, BF16, TE FA2 |
| Rollout layout | Four TP1 engines, BF16, Triton attention/MoE, Torch GEMM |
| Runtime | Colocate/offload, 8192 training tokens/GPU, CUDA graphs disabled |
| Evaluation | Fixed 256 GSM8K test questions, seed 42, before training and every 10 rollouts |
| Eval sampling | Temperature 1, top-p 0.7, top-k disabled, one response/question |
| Checkpoint | Final complete optimizer checkpoint; deadline sentinel may request an earlier save |

The 7,473 training prompts are 35–235 tokens; the fixed test prompts are 49–185
tokens with the actual tokenizer. No examples are removed by the 512-token cap.
Train/test question overlap is zero. `prepare_qwen3_gsm8k.py` records source and
output hashes, original indices, and an immutable test selection. Labels and
prompt messages are preserved.

`gsm8k_verl_reward.py` uses the strict GSM8K scorer from a recorded official verl
source revision. The historical internal image's exact source commit could not
be retrieved; byte-for-byte parity with that historical image is unverified.
The adapter reports answer correctness and formatting separately and does not
force truncated answers to zero. See its module docstring for source identities.

## Historical reference, not the Miles performance baseline

`qwen3-gb300-baseline.json` identifies its framework as verl/vLLM and retains all
50 original steps, source-log hashes, timer definitions, and aggregation rules.
It must not be presented as a Miles GB300 result.

| Historical step | Training reward | Length-cap fraction |
| --- | ---: | ---: |
| 1 | 47.314% | 37.646% |
| 10 | 81.250% | 15.527% |
| 25 | 92.480% | 5.762% |
| 50 | 93.652% | 3.662% |

The last ten historical steps averaged 55.683 seconds/rollout and 5191.875
tokens/second/GPU. That throughput counts input plus output tokens, divided by
rollout-step elapsed time and four GPUs. Its actor MFU is always zero and is not
a usable utilization measurement. This reference used different MoE kernels,
microbatch token limits, optimizer implementation and memory scheduling, and
did not run held-out evaluation.

## Reproduce

Use the same Miles checkout and learning arguments on both platforms. Native
CUDA extensions must contain code for the actual GPU architecture; the preserved
Rubin image contains SM107-only FA2/Apex kernels and needs a matching-version
GB300 extension build before it can serve as the GB300 runtime.

Inside an externally prepared four-GPU container:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python3 lab/rubin_two_node/run_qwen3_30b_a3b_gsm8k_rubin.py \
  --model-dir /home/scratch.kaixih_ent/models \
  --prompt-data-path /run-output/inputs/train.jsonl \
  --eval-prompt-data-path /run-output/inputs/test-fixed-256.jsonl \
  --output-dir /run-output --save-trigger-sentinel /run-output/checkpoint-now \
  --extra-env-vars RUBIN_RUN_ID=UNIQUE_RUN_ID --print-only
```

Remove `--print-only` after validating the resolved arguments, GPU layout,
storage, and bounded run monitor. `orchestrate_rubin.py --recipe qwen3-gsm8k`
supports one node and passes the node count to the launcher. Existing two-node
Qwen3.5 recipes remain available with `--recipe qwen35-smoke`.

Compare rewards by rollout and cumulative samples; compare throughput with the
same token/time definition, separating evaluation/checkpoint/startup time.
Collect short profiles after the main measurements, then identify time spent in
generation, log-prob calculation, training, weight synchronization, memory
transfer and communication. This task diagnoses differences; it does not tune
or optimize either platform.
