# Miles TRTLLM BF16: a separate HTML slide deck

This 12-slide report uses only the new full GB300/Rubin campaign. The previous
Triton/eager/CUDA Graph HTML reports stay untouched. The initial deck is explicitly
pending: no old trace, curve or timing value is presented as a new measurement.

Bound run IDs:

- Rubin: `20260917-rubin-j2212643-trtllm`
- GB300: `20260917-gb300-j2212644-trtllm`

## Build after collecting new evidence

The existing collector writes `comparison.json`, `health.json` and each
platform's original `logs/qwen3_train.log` under the new output root.
The builder rechecks source-log hashes and independently repeats the completion
audit against the raw rank-step lines. Final status requires 50 rollouts,
200 optimizer updates, 800 NORMAL rank outcomes, finite positive gradients,
finite losses, Ray SUCCEEDED and successful driver exits on each platform.

```bash
python3 -B lab/rubin_trtllm_full/build_report.py \
  --inputs outputs/rubin-gb300-qwen3-trtllm-full
```

Missing data produces pending slides; an in-progress collection produces clearly
interim observations. Main curve fields are not fabricated or smoothed. The
builder rejects old run IDs, wrong MoE/graph settings, changed dataset hashes,
stale health receipts and changed source logs. It writes only inside this new
report directory.

After running the separate capture and
`lab/rubin_trtllm_full/profiling/analyze_profile.py --png` on both platforms:

```bash
python3 -B lab/rubin_trtllm_full/build_report.py \
  --inputs outputs/rubin-gb300-qwen3-trtllm-full \
  --profile-rubin /ABSOLUTE/NEW/rubin-profile-evidence/profile-evidence.json \
  --profile-gb300 /ABSOLUTE/NEW/gb300-profile-evidence/profile-evidence.json \
  --final
```

`--final` additionally requires all four fresh screenshots and their exact
retained source traces, four actual correlated decode replays per platform,
matched frozen-request hashes and selected-forward batch/token fields.
Every supplied profile binds to the corresponding main run and its actual image
digest. PNG and raw-trace hashes are verified before copying into the offline
site. A missing partner or token mismatch cannot silently become a matched
comparison.

Open `site/index.html`. Arrow keys/Space/PageUp/PageDown navigate; F fullscreen;
O overview. Printing creates 16:9 pages. Plotly and all assets are local, so no
network is needed. `report-data.json` retains raw source values, timing selections,
validation, version provenance and profile measurements. `asset-manifest.json`
records the emitted file hashes.

## How to interpret the report

- The model is **Qwen3-30B-A3B standard**, not Qwen3.5. Training and evaluation
  rewards are separate charts. Evaluation uses the same 256 held-out questions.
- Correctness views include reward, held-out accuracy, response length,
  truncation, training/rollout log-probability difference and gradient norm.
  Observed curves are shown without asserting statistical equivalence.
- Whole-step, generation and actor timings all remain visible. The identical
  completed steady-round cohort excludes startup and checkpoint-related rounds.
  Its exact IDs remain in evidence JSON instead of unexplained slide labels.
  Timing lines retain gaps at excluded/missing rounds. Stage bars are independent
  timers, **not** an additive wall-time partition. An actor slowdown is retained
  even when generation improves.
- The generation throughput summary is weighted by generation time, using the
  actual per-round output-token rates. It is not input+output throughput or a
  simple average of rates.
- Four full-size trace screenshots show real prefill and decode windows on both
  platforms. The profile table uses **GPU annotation elapsed time**. It never
  substitutes CPU `cudaGraphLaunch` submission time. Kernel sums, interval unions
  and uncovered intervals remain distinguishable in downloadable evidence.
- The short profile uses one initial-policy TP1 engine with matched requests;
  main generation uses four engines and evolving weights. Instrumentation and
  software differences preclude a hardware-only explanation of whole-run speed.
- GB300 uses the upstream Miles base with scoped weight-refit fixes; Rubin uses
  its CUDA13.4 adaptation. Planned TE versions are 2.17/2.19 respectively; actual
  metadata versions render as soon as available.

## Browser QA

```bash
/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  reports/rubin-gb300-qwen3-trtllm-full/check_slides.mjs \
  reports/rubin-gb300-qwen3-trtllm-full/site \
  reports/rubin-gb300-qwen3-trtllm-full/qa-final
```

This checks 12 slides, overflow, navigation/mobile scaling, zero external requests,
source-value/cohort preservation, pending states and actual image bindings. It
renders every slide to PNG for visual inspection. Run it again after final data
and images arrive; pending QA cannot validate final-data layout.
