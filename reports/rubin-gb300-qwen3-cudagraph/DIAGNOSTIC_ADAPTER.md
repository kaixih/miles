# Retained diagnostic evidence → new report inputs

`prepare_diagnostic_report.py` is an offline adapter. It never queries a node,
launches a model, edits source evidence, or replaces an existing output directory.
It produces the existing `profiles.json` and `diagnostics.json` contracts plus
per-platform evidence receipts and actual trace renderings. The original report
is read only: its timeline renderer is reused, but none of its data is imported.

Create an explicit input list after a diagnostic is retained locally. These are
path/identity placeholders; use the exact actual retained capture ID:

```json
[
  {"platform":"rubin", "directory":"/absolute/local/retained/operator-root",
   "capture_run_id":"EXACT_CAPTURE_ID"}
]
```

The directory must contain `operator-plan.json` or `recovery-plan.json`, a
successful `diagnostic-retention.json`, and its `node-output/` artifacts. The
adapter verifies main run ID against `experiment.json`, capture/engine/Ray
submission identities, model inventory ID, wrapper source hashes, the actual
resolved graph policy, and size/SHA256 for every consumed retained artifact.
These are provenance checks; this utility does not independently revalidate
main training correctness or substitute for its 50-rollout/200-update audit.

```bash
python3 -B reports/rubin-gb300-qwen3-cudagraph/prepare_diagnostic_report.py \
  --experiment reports/rubin-gb300-qwen3-cudagraph/experiment.json \
  --inputs /absolute/local/diagnostic-inputs.json \
  --output /absolute/local/NEW-evidence-report-directory --png
```

Supply generated files explicitly to `generate.py --profiles .../profiles.json
--diagnostics .../diagnostics.json`. This adapter does not edit canonical inputs
or generate the presentation. A missing platform/mode/stage stays pending. A
retained identity/hash inconsistency fails loudly. A retained failed terminal is
preserved; an earlier complete stage does not make the whole diagnostic succeed.
Partial output is preserved if rendering fails; choose a fresh directory to retry.

`--png` uses local Chrome through the existing trace capture helper, bounded to
90 seconds per selected image. Without it, the adapter produces the real trace
HTML/selection receipts and verifies the trace, but the slide's screenshot slot
remains pending. No placeholder or synthetic image is created.

## Evidence semantics

- Three raw **HTTP generation durations** come from `measured-1/2/3.json`, checked
  against response token metadata and `summary.json`. They include prefill,
  decode, queueing and response delivery for one TP1 engine with128 frozen
  requests ×64 output tokens. Warmup, graph creation and trace capture timings
  are excluded. This is separate from four-engine main Miles timings.
- OFF/ON request IDs deliberately differ. The contract's `request_sha256` uses
  the wrapper's canonical `workload_sha256`, removing **only `rid`**; literal
  request hashes remain in the receipt. `context_sha256` hashes frozen initial
  token IDs, not per-forward KV lengths, generated tokens or expert routing.
- A verified pair also requires actual paired OFF eager-DECODE launch evidence
  and ON graph-replay evidence. Raw HTTP samples remain available in the receipt
  when capture proof is missing, but the comparison stays pending.
- The primary images are **ON-mode eager prefill** and **ON-mode graph decode**.
  The selection is the earliest complete paired EXTEND with direct kernel
  launches and no graph launch, and the earliest complete paired DECODE at
  `bs=128` with graph-launch proof. It never selects the fastest duration. If no
  qualifying stage exists, that image remains pending. Native file suffixes do
  not decide stage: an EXTEND in a DECODE-named file is valid evidence.
- The analyzer pairs CPU/GPU annotations using exact annotation name and
  `External id`. Replay proof requires a graph-launch API within the matched
  same-thread CPU DECODE span and kernels in its paired GPU span. Counts are
  recorded ON GPU-annotation counts, not full-main totals or a fallback rate.
  Absent proof is **unknown**, not an eager fallback. Mixed forward labels and
  the full original trace remain in the receipt.
- Renderer start time is converted from the analyzer's absolute microseconds to
  milliseconds relative to the first recorded GPU event on the selected device.
  Exact selected forward fields, index, window and original compressed SHA are
  preserved in `selection.json`. Kernel interval unions are recorded timestamp
  coverage, never physical utilization, occupancy, or a CPU-time attribution.
- All trace timings include profiler overhead. One-engine scope, scheduling,
  context distributions and routing prevent hardware-only causal claims.

Focused CPU tests (temporary synthetic inputs only):

```bash
python3 -B reports/rubin-gb300-qwen3-cudagraph/test_prepare_diagnostic_report.py
```

The real failed Rubin v2 retention was also checked locally: zero verified
pairs/images, terminal `FAILED` preserved, with no canonical report writes.
Actual successful trace compatibility must be checked again once retained.
