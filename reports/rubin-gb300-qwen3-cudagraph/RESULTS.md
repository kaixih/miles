# Qwen3 with decode CUDA Graph — 2026-09-16

Both Miles runs completed **50 rollouts and 200 optimizer updates** on one
four-GPU node per platform. Decode CUDA Graph was enabled; prefill graphs were
disabled. The previous eager experiment and its HTML slides remain separate.

Slide 11's OFF/ON control refers specifically to rollout decode CUDA Graph.
Rubin's ON replay is verified, but its wrapper failed during post-ON cleanup
before collecting OFF. This does not indicate a failure of the main run.
The two retained actor microbatch traces contain no graph launch or setup
events; see [the trace scan](actor-evidence/actor-graph-mode.json), bound to their
SHA256 values. The actor gap therefore needs a separate host/software explanation.

## Matched main runs

Both used Qwen3-30B-A3B standard, GSM8K, 256 prompts × 8 responses per rollout,
global batch 512, four updates per rollout, prompt limit 512 and response limit
1024. Training and log-prob microbatch budgets were both 4096 tokens/GPU.
Rollout used four TP1 engines with Triton attention/MoE and Torch BF16 GEMM.

The timing cohort is the same 48 completed, unprofiled rollouts on both systems:
IDs 1–48. Warmup rollout 0 and the configured final checkpoint round 49 are excluded.
Checkpoint writing has its own timer; excluding round 49 does not imply its step
timer includes the later checkpoint write.

| Measurement | Rubin | GB300 |
| --- | ---: | ---: |
| Mean Miles step | 170.745 s | 247.913 s |
| Mean generation | 47.071 s | 54.910 s |
| Mean actor training | 66.596 s | 118.095 s |
| Mean reference log-prob | 19.737 s | 29.296 s |
| Mean actor log-prob | 16.167 s | 25.843 s |
| Mean weight sync | 3.982 s | 4.276 s |
| Weighted output tokens/GPU/s | 5,729.321 | 4,929.029 |
| Training reward, rollout 0 → 49 | 44.824% → 96.143% | 46.338% → 95.703% |
| Training truncation, rollout 0 → 49 | 36.182% → 1.953% | 35.547% → 2.344% |
| Held-out reward, initial → final | 120/256 → 247/256 | 118/256 → 245/256 |

Weighted throughput is total output tokens divided by total generation seconds
and four GPUs. It is not the mean of per-rollout rates. Held-out evaluation uses
the same fixed 256 questions, one sampled response per question, temperature 1
and top-p 0.7. These are recorded scorer results, without independent regrading.

Both runs recorded all 200 finite, positive gradient norms, finite losses and
800 NORMAL rank-update outcomes; Ray succeeded and both driver exit codes were
zero. Gradient-norm ranges were 0.004766–0.893772 on Rubin and
0.003696–0.249636 on GB300.

The generation difference is 7.839 seconds per rollout. Actor training is the
largest measured substage difference, at 51.499 seconds. Nested timers are not
additive: generation must not be added again to the Miles step timer. The step
difference decomposes into the recorded train and wait timers. This diagnostic
profiles generation; it does not identify the kernel-level cause of the actor
training difference.

## Runtime and historical comparison

GB300 used the upstream image unchanged:
`radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986`.
The normal-user launch requires only traversal permission on the image's `/root`
directory to import its existing editable Megatron installation. No package or
model-code change is made by that permission preparation.

Rubin used the existing image saved in the user's GitLab registry:
`gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`.
Its build changes are documented in the preserved original presentation.

The software stacks differ, including Torch/CUDA and Transformer Engine. These
measurements compare the recorded systems and do not isolate GPU hardware.
Both main runs used frozen Miles source `e73d8d68d65b467410113a0d91a879b77f6d1351`.

The old GB300 run used 8192-token training **and log-prob** microbatch budgets;
this run uses 4096. The old runs also had different checkpoint schedules. A
historical comparison must preserve these changes and use the common eligible
rollout IDs. Faster generation alongside slower GB300 actor training therefore
cannot be attributed solely to turning CUDA Graph on.

## Separate generation traces

The bounded diagnostic uses one TP1 engine, the initial policy, 128 frozen
GSM8K prompts and exactly 64 output tokens per request. Every batch starts after
a verified radix-cache flush. Each mode has one warmup, three unprofiled HTTP
measurements and a separate short CPU/GPU profile. This workload differs from
the four-engine main run. HTTP duration includes prefill, decode, queueing and
response delivery.

Rubin's ON capture retained one prefill forward with batch 128 and 10,883 input
tokens, plus four decode forwards with batch 128. Each decode forward has one
same-thread CUDA graph launch and 868 matching graph kernels identified by
launch correlation. Prefill uses eager launches. The earliest decode GPU
annotation spans 10.038 ms; the four spans are 10.038, 9.672, 8.709 and 9.011 ms.
The prefill annotation spans 122.379 ms. These are instrumented trace durations.

The screenshots render real retained trace events. Kernel tables describe
device activity overlapping the selected annotation window; asynchronous
scheduler work can overlap it. Cumulative duration, interval-union coverage and
uncovered time are separate quantities and do not measure physical utilization.

Rubin's ON files are valid, but its subsequent cleanup failed before OFF ran.
No Rubin OFF/ON speedup is claimed. The wrapper issues were fixed and covered by
native Linux CPU regressions for process-title changes and nondumpable process
exit. A later port preflight fix also distinguishes retired TIME_WAIT connections
from actual listening servers; a native Linux socket regression covers both.

GB300 completed a matched OFF/ON diagnostic using the same frozen requests,
initial policy, image and cache policy. The three unprofiled HTTP batches were
3.976579, 4.001848 and 4.018212 seconds with graph OFF, versus 0.936774, 0.933128
and 0.932162 seconds with graph ON. Means are 3.998880 and 0.934021 seconds,
respectively: **4.28× higher batch throughput** for this short workload. Each
batch returned 128 × 64 output tokens from 10,883 input tokens; the generated
64-token sequences matched exactly across OFF/ON for all 128 requests in each
of the three corresponding measurements. This is complete
request time, including prefill, decode, queueing and response delivery; it is
not a pure decode timer or an end-to-end Miles speedup. Three sequential samples
per mode provide a bounded diagnostic, not a broad statistical benchmark.

The four GB300 ON decode GPU annotations span 9.131, 8.760, 7.652 and 7.854 ms;
the corresponding captured OFF sequence spans 101.037, 86.741, 84.099 and 83.908 ms.
The prefill spans remain close: 125.742 ms OFF and 126.622 ms ON, with prefill
graphs disabled in both. These instrumented windows show the launch pattern
change; profiler overhead prevents using their ratio as an unprofiled speedup.

Both GB300 modes retained full prefill and decode traces. OFF shows native
kernel launches with no graph launch; all four ON decode forwards show actual
CUDA graph replay. Both modes finished their workload and trace export before
bounded process cleanup. The wrapper exited 0 and the exact container was stopped
before the stable 28-file retention check. The two cleanup sequences needed a
KILL after the configured TERM grace period; those recorded engine exit codes
are not unexpected workload failures.

## Evidence and reproduction

The frozen final comparison SHA256 is
`eb44d048e25c68e95fc32c2d8dbfb8ca973f3e23eaf061c09051205e1a44baaf`.
The independent main analysis SHA256 is
`af7833fe6466feb7157436e9a913180f8d53b2bed97afb8eb9cd220db8dbc61b`.
It verifies four source-log hashes and 1,424 metric records, retains raw metric
keys and exact cohorts, and independently recomputes means, medians and rates.

See [README.md](README.md) for the slide builder and evidence contracts,
[DIAGNOSTIC_ADAPTER.md](DIAGNOSTIC_ADAPTER.md) for converting retained captures,
and [the diagnostic runbook](../../lab/rubin_two_node/SGLANG_GRAPH_DIAGNOSTIC.md)
for the bounded runtime procedure. The offline HTML deck includes its input
hashes, raw observations and downloadable trace evidence.

## Actor supplement — one matching microbatch on each platform

The original main learning runs and their 1.7733× actor-stage ratio remain
unchanged. On the main cohort, GB300 processed only 0.309% more reported prompt
and response tokens and 0.474% more scheduled microbatches. That workload audit
is retained separately; the sampled trace below does not measure the whole
main actor stage.

Both systems completed an actor-only replay of the same newly generated,
frozen initial-policy batch: 256 prompts × 8 responses, four real updates of
512 samples, four GPUs, TP1/EP4 and a 4,096-token training budget per GPU. Both
used frozen Miles e73d8d68 and the same recorded initial HF/release construction
with a fresh optimizer. Every rank/update completed normally with finite,
positive gradients. All 16 packing fingerprints match across platforms;
the selected four sequences have identical ordered sample IDs and token
hashes (4,096 total tokens, including 3,644 response tokens).

Slide 16 uses only this one microbatch's time window:

| Same microbatch | Rubin | GB300 |
|---|---:|---:|
| Total elapsed | 0.785 s | 2.048 s |
| Forward | 0.244 s | 0.630 s |
| Backward, including recomputation | 0.540 s | 1.418 s |
| GPU kernel interval coverage inside that window | 0.432 s | 0.294 s |

The total and forward/backward values are elapsed CPU annotation ranges and
include dispatch and waits. The tiny remainder outside the F/B annotations
and rounding explain why the displayed components need not sum exactly to
the total. GPU kernel coverage is the union of recorded kernel intervals
within that same window; it is neither a second elapsed-time denominator nor
a measurement of physical GPU utilization.

Rubin's measured microbatch finishes sooner even though its GPU kernel
coverage is longer. The trace therefore does not support explaining this
sample's elapsed-time advantage by uniformly faster GPU kernels. Recorded GPU
activity, including memory operations, leaves about 0.352 s uncovered on Rubin
versus 1.753 s on GB300. Those intervals contain host work, waits and profiler
overhead that this capture does not fully attribute. Dispatch and scheduling
are candidates for further investigation; uncovered intervals are not proof
that the GPU was physically idle.

Longer Rubin communication-kernel intervals account for much of the difference
in GPU coverage. NCCL kernels can wait or spin, so their duration alone does
not measure network bandwidth or identify an interconnect problem. Kernel
families overlap across streams and must not be added as a critical-path
breakdown. The trace also shows different attention implementations: Rubin
uses the FA2 kernel family and GB300 uses the FlashAttention CuTe Sm100 family.
Their training stacks differ (Torch 2.15/cu134 and TE 2.19 on Rubin versus
Torch 2.13/cu130 and TE 2.17 on GB300).

This is a profiled rank-0 sample from a fresh initial-policy diagnostic.
Identical batch, recipe and initial construction do not establish bitwise
identical weights after warmup. Profiler overhead also differs with software.
The trace locates a large difference outside recorded GPU execution, but does
not establish a hardware-only cause or fully explain the main 1.77× ratio.
Optimizer and final gradient synchronization lie outside this microbatch;
pure recomputation is not separated from backward. Direct GPU attribution to
the outer backward scope is incomplete because autograd uses another thread.

Both exact diagnostic containers are stopped and both allocations released
after hash-verified retention. Complete traces, native receipts, independent
trace QC and the paired workload audit are downloadable from the deck. Full
update timings remain in those audits. Slide 10 keeps the original main-stage
chart; slide 16 contains both microbatch windows. The original rollout figures
and the 16-slide count are preserved.
