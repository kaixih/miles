# Final Qwen3 learning and paired timing

Snapshot: 2026-09-16T05:28:24.849459+00:00. SHA256: `89f46cb823c7702cc87604ad6ca83aaa28b75aba2def693af5a81cf6356594e6`.

**Both selected runs SUCCEEDED: 50 rollouts and 200 optimizer updates each.** Both outer/inner exits are0. Independently scanned raw logs confirm all200 aggregate gradient norms are finite and positive, losses finite, and all800 rank/update outcomes per run are NORMAL/valid. Duplicate metric prints agree exactly.

| Run | Training reward initial → final | Last10 reward | Truncation initial → final | Last10 truncation | Final held-out |
|---|---:|---:|---:|---:|---:|
| rubin | 45.61% → 94.58% | 95.34% | 35.89% → 3.81% | 2.36% | 96.48% (247/256) |
| gb300 | 46.63% → 95.65% | 95.28% | 36.87% → 1.42% | 2.17% | 96.09% (246/256) |

Each last10 window is rollouts40–49 with20,480 sampled responses. Final training reward is measured before its own update; held-out final evaluation follows all200 updates. The same256 evaluation questions are reused stochastically; the final one-question difference does not establish a quality gap.

| Optimizer updates | Rubin held-out accuracy / truncation | GB300 held-out accuracy / truncation |
|---:|---:|---:|
| 0 | 46.48% / 38.67% | 47.27% / 39.84% |
| 40 | 82.81% / 14.84% | 82.81% / 16.02% |
| 80 | 92.19% / 6.64% | 91.41% / 6.25% |
| 120 | 95.70% / 2.34% | 94.92% / 3.52% |
| 160 | 95.31% / 2.73% | 96.48% / 1.56% |
| 200 | 96.48% / 1.56% | 96.09% / 1.17% |

## Paired timing: N=17

Exact shared rollout IDs: `[1, 2, 3, 6, 7, 8, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48]`.

Both sides use the same completed, unprofiled, timing-eligible IDs, excluding warmup0 and existing save/I/O windows. Values are **mean / median**.

| Metric | Rubin ES A2 | GB300 A1 |
|---|---:|---:|
| Miles step (s) | 215.49 / 192.89 | 317.94 / 295.43 |
| Generation (s) | 95.63 / 84.92 | 190.01 / 180.09 |
| Actor train (s) | 63.96 / 56.26 | 72.17 / 63.94 |
| Reference log-prob (s) | 19.67 / 17.45 | 20.34 / 18.00 |
| Actor log-prob (s) | 16.31 / 14.26 | 16.60 / 14.52 |
| Weight sync (s) | 3.91 / 3.67 | 4.30 / 4.22 |
| Output tokens/GPU/s | 2863.29 / 2716.34 | 1468.75 / 1394.08 |

| Own eligible cohort | N | Step mean / median(s) | Output tokens/GPU/s mean / median |
|---|---:|---:|---:|
| rubin | 22 | 210.63 / 193.76 | 2810.57 / 2686.71 |
| gb300 | 27 | 326.72 / 328.85 | 1503.05 / 1518.23 |

**Scope:** Rubin ES A2 uses4096 training/logprob tokens/GPU; GB300 uses8192. Images/software differ. Timing reflects these two systems and workloads, not a hardware-only speedup. Nested stage timers are not additive. Actual SGLang decode evidence is summarized below; kernel-level causes remain unresolved. Historical VeRL and failed Rubin A1 are not mixed into this comparison.

## Evidence

- rubin: gradient range `0.004248159006237984`–`0.25029805302619934`; raw log SHA256 `c8044ee521522c96e81d191906d9183c4ce46500a240f3f44708d3666e64e1e9`.
- gb300: gradient range `0.0039013447239995003`–`0.5748487114906311`; raw log SHA256 `befcd670281765c535d56964606cbb790f7c1f3703ee6ba63b6e0e1f3b65a882`.
- Full statistics, exact paired raw values, all400 gradient/loss observations with physical line numbers, terminal file hashes and conditions: [final-metrics.json](evidence/final-metrics.json).
- Physical file newlines were used for log auditing; Unicode line-separator characters were not treated as new log lines.

## Actual decode capture

Each short diagnostic replay starts from the same initial model and uses the same4096 token/GPU cap. The original software images are preserved. The captures are separate from the complete learning run above.

Both SGLang captures contain actual CPU/CUPTI events and have matching source-before/source-after, durable NFS, and local SHA256 receipts. Select only `step[DECODE bs=128]`: Rubin has4 forwards; GB300 has1. GB300's other2 decode127 and1 extend1 forwards are excluded.

| Mean per selected forward | Rubin ES, N=4 | GB300, N=1 |
|---|---:|---:|
| GPU annotation elapsed span(ms) |36.053|85.220|
| Kernel interval union(ms) |14.832|13.270|
| Remaining annotation span(ms) |21.221|71.950|
| Kernel calls |875|875|

The recorded kernel intervals do not account for the sampled elapsed-time gap. This points to host/runtime/launch scheduling as areas for further isolation. It does not establish a particular CPU bottleneck or explain the whole production generation gap. Context lengths and expert routing are not proven equal; images, profiler overhead, and the unequal small samples constrain the comparison.

Kernel interval union is trace coverage, not production GPU utilization. The remaining interval includes gaps, copies and unknown time, not just CPU time. Runtime and driver launch APIs can nest and must not be added together. Their larger recorded GB300 durations do not account for the full forward-span gap.

[Raw per-forward values, scope and source hashes](evidence/sglang-decode-comparison.json). The HTML slides show original trace renderings and include the original short SGLang gzip traces for download. They are not Nsight UI screenshots.

## Trainer profiler limitation

Both separate diagnostic replays completed their eight optimizer updates, but neither yielded a verified complete trainer trace. Rubin reached its original profiling deadline during export. GB300's profiler finalization exceeded Ray's95% host-memory threshold (916.31/958.34GB reported), and Ray terminated workers. These are diagnostic replay outcomes; both original50-rollout learning runs succeeded. No deadline was extended and no optimization or new training attempt followed. The kernel analysis above uses only the independently completed, hash-verified SGLang captures.
