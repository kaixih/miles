# Qwen3 CUDA Graph: separate offline HTML slides

This directory belongs to the **new** decode-graph experiment. The original
`../rubin-gb300-qwen3` report stays unchanged. The initial14-slide deck contains
planned scope and explicit **PENDING / UNMEASURED** sections, with no old curves
or screenshots copied as new measurements.

```bash
python3 reports/rubin-gb300-qwen3-cudagraph/generate.py
```

Open `site/index.html`. Everything needed to display the deck is local. Arrow
keys/Space/PageUp/PageDown/Home/End navigate, F toggles fullscreen, O opens the
overview, and printing uses16:9 landscape pages. The optional original-report
link expects the two report directories to remain siblings. The new deck still
works by itself if that separate baseline is unavailable.

## Update after evidence arrives

1. Fill `experiment.json.run_bindings` with the new exact platform/run identities:

   ```json
   [{"label":"rubin","run_id":"ACTUAL_NEW_RUBIN_RUN_ID"},
    {"label":"gb300","run_id":"ACTUAL_NEW_GB300_RUN_ID"}]
   ```

2. Collect a new `miles-qwen3-comparison-v1` JSON using the existing metrics
   summarizer. Keep source logs, timestamps and metadata in a **new output root**.
   The builder rejects any supplied run that lacks an exact binding and refuses
   the explicitly listed old eager run IDs. An enable flag never creates replay
   evidence. Recipe fields display actual metadata when present and otherwise
   say `Planned`.
3. Regenerate explicitly. These input paths are placeholders, not existing runs:

   ```bash
   python3 reports/rubin-gb300-qwen3-cudagraph/generate.py \
     --experiment reports/rubin-gb300-qwen3-cudagraph/experiment.json \
     --runs outputs/NEW_EXPERIMENT/comparison.json \
     --profiles outputs/NEW_EXPERIMENT/profiles/profiles.json \
     --diagnostics outputs/NEW_EXPERIMENT/diagnostics.json \
     --run-health outputs/NEW_EXPERIMENT/snapshot-health.json
   ```

   Optional missing files stay pending. No current-task or remote discovery occurs.
   Supply the same comparison SHA in optional health metadata to reject staleness.
4. Run the browser checker below and inspect the changed slide screenshots.
   Preserve old versions of the new input evidence before replacing snapshots.

The builder stores all supplied data plus hashes in `site/report-data.json`.
It preserves raw per-run statistics and derives paired stage bars from the
intersection of completed, explicitly unprofiled, timing-eligible IDs. Warmup0,
configured save/I/O exclusions and incomplete observations are removed from
timing only. Excluded/missing rollout IDs retain null slots, so line segments
cannot span those gaps. Learning curves retain their actual values and IDs.

## Profile and graph evidence contract

Use a separate new profiles JSON with the exact `experiment_id`. A record's
`source_run_id` is the bound main run associated with this platform; the actual
diagnostic/capture job identity belongs in `capture_run_id` and the receipts.
Do not reuse a previous-run capture under a new name.

```json
{
  "schema":"qwen3-cudagraph-profiles-v1",
  "experiment_id":"qwen3-cudagraph-nightly-v1",
  "status":"pending",
  "graph_evidence":[],
  "profiles":[]
}
```

Each `graph_evidence` object supports:

```text
run_label, source_run_id, verified, scope, capture_status,
decode_forwards, decode_graph_replays, decode_fallbacks, decode_unknown,
prefill_forwards, prefill_graph_replays,
evidence_refs: [{path: local receipt, sha256: exact SHA256}]
```

Counts must be nonnegative integers; replay+fallback+unknown must equal observed
decode forwards when all counts are supplied. A `verified:true` record requires
referenced local receipts whose hashes the builder checks and copies. Counts
describe the recorded scope, not automatically the full main run. Runtime proof
should include actual replay API/event correlation and the exact batch/context
scope. Prefill remains explicitly separate from decode.

Each `profiles` object supports:

```text
run_label, source_run_id, capture_run_id, verified,
title, scope, observations: [up to three concise measured statements], caption,
image: local PNG/JPEG/WebP, trace: local JSON/JSON.gz/etc,
source_trace_sha256: exact compressed-file SHA256
```

The builder copies only explicit local attachments and checks the declared trace
hash. A verified profile requires the complete local trace. Unverified images
never render. Preserve export/retention failure evidence separately; completed
main training does not imply completed profiling. Actual screenshots or trace
renderings only, with profiler overhead and causal limitations in the caption.

## Separate OFF/ON generation diagnostic

`--diagnostics` accepts `qwen3-cudagraph-diagnostics-v1` with the matching
`experiment_id` and `pairs:[]`. Each pair has `platform`, bound `source_run_id`,
`verified`, `scope`, `evidence_refs`, and `off`/`on` condition objects. Each condition
records:

```text
image_digest, initial_model_id, request_sha256, context_sha256,
cache_policy, prefill_graph, decode_graph,
timing_scope: "http_request_prefill_decode_queue_response",
generation_seconds: [actual raw HTTP request durations in seconds]
```

For a verified pair, image/model/request/context/cache/prefill fields must agree,
decode modes must be false/true, the exact timing scope above must be supplied,
and duration samples must be finite and positive.
The builder derives mean/median/count directly from the raw samples and checks
local receipt hashes. The diagnostic stays pending otherwise. This panel is
generation-only; it is not a replacement for main learning or end-to-end Miles timings.

Populate each `generation_seconds` array directly from the corresponding wrapper
`off/summary.json` or `on/summary.json`:

```python
generation_seconds = [m["generation_request_seconds"] for m in summary["measurements"]]
```

This is complete HTTP request wall time, including prefill, decode, queueing and
response delivery for one TP1 engine. It excludes model loading, graph creation,
flush and warmup. Keep raw input/output token counts, cache observations and the
summary SHA in the attached receipts; equal input requests need not imply equal
sampled output lengths. The slide reports seconds, with derived
`generation_seconds_mean`, `generation_seconds_median` and `sample_count`.
`decode_ms` is rejected in verified diagnostic conditions. Actual scheduler
DECODE span durations (trace microseconds, converted explicitly if needed) remain
separate profile evidence and must never populate this HTTP timing array.

## Checks and source ownership

```bash
python3 -B reports/rubin-gb300-qwen3-cudagraph/test_generate.py
/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  reports/rubin-gb300-qwen3-cudagraph/check_slides.mjs \
  reports/rubin-gb300-qwen3-cudagraph/site \
  reports/rubin-gb300-qwen3-cudagraph/qa
```

The8 CPU tests exercise pending mode, exact new identity binding, rejection of
old runs, stale health, inconsistent replay counts and mismatched paired context.
Browser QA checks overflow, pending charts, source values, timing gaps, replay
status, local-only assets and keyboard/mobile navigation. Synthetic unit inputs
exist only in temporary test directories, never in the canonical pending deck.

Source files are `generate.py`, `experiment.json`, `template.html`, `assets/*`,
`test_generate.py`, `check_slides.mjs` and this README. `site/` and `qa/` are generated
artifacts. Plotly3.1.0 is bundled locally; the CSS/navigation and paired-timing
selection reuse the original report's design and evidence logic. No runtime,
collector, cluster, model, or original-report file is modified by this builder.
