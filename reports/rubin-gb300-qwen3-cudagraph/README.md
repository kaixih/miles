# Qwen3 CUDA Graph: separate offline HTML slides

This directory belongs to the **new** decode-graph experiment. The original
`../rubin-gb300-qwen3` report stays unchanged. The 16-slide deck contains the new learning and performance curves plus separate
prefill and decode evidence. See [RESULTS.md](RESULTS.md) for measured conclusions
and limitations. Missing optional evidence is explicitly marked; old curves and
screenshots are never copied as new measurements.

```bash
python3 reports/rubin-gb300-qwen3-cudagraph/generate.py
```

Open `site/index.html`. Everything needed to display the deck is local. Arrow
keys/Space/PageUp/PageDown/Home/End navigate, F toggles fullscreen, O opens the
overview, and printing uses16:9 landscape pages. The optional original-report
link expects the two report directories to remain siblings. The new deck still
works by itself if that separate baseline is unavailable.

## Rebuild the retained final experiment

When the task's local output directory is available:

```bash
python3 -B reports/rubin-gb300-qwen3-cudagraph/generate.py \
  --runs outputs/rubin-gb300-qwen3-cudagraph/comparison.final.json \
  --run-health outputs/rubin-gb300-qwen3-cudagraph/health.final.json \
  --profiles outputs/rubin-gb300-qwen3-cudagraph/profile-evidence-final-20260916/profiles.json \
  --diagnostics outputs/rubin-gb300-qwen3-cudagraph/profile-evidence-final-20260916/diagnostics.json
```

The final package contains both complete 50-rollout main curves and four verified
ON trace figures. Rubin's missing OFF comparison remains explicitly unavailable;
GB300 has a verified matched OFF/ON request-time diagnostic. Data and trace files
are distributed in the offline report package; the Git branch contains source
and the measured results summary, without model/checkpoint files.

## Actor supplement in the existing slides

The supplement contains verified Rubin and GB300 forward/backward traces of
the same microbatch. See [paired evidence and limits](actor-evidence/rubin-v1/README.md).
To include it when rebuilding the complete main experiment above, append:

```bash
--actor-profile outputs/rubin-gb300-qwen3-cudagraph/actor-update/actor-profile.paired-verified.json
```

This changes only slide16. Slide10 retains the original main stage bars; actor
timelines and two same-microbatch statements appear together on slide16. The deck still has
16 slides and preserves all main curves, timing values and rollout captures.

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
run_label, source_run_id, capture_run_id, stage: "prefill" | "decode", verified,
decode_graph: boolean, prefill_graph: boolean,
title, scope, observations: [up to three concise measured statements], caption,
image: local PNG/JPEG/WebP, trace: local JSON/JSON.gz/etc,
source_trace_sha256: exact compressed-file SHA256
```

The deck reserves four separate profile slides: Rubin prefill, Rubin decode,
GB300 prefill and GB300 decode (16 slides total). Each platform/stage pair accepts
one explicitly selected primary capture; missing stages remain pending. Optional
`decode_graph` and `prefill_graph` booleans visibly label the diagnostic condition
(ON/OFF); absent fields display unknown, never an inferred graph mode. A record
must name its stage, and duplicate platform/stage records are rejected rather
than silently showing only the first. Select the intended OFF or ON primary
capture explicitly; keep alternatives in linked receipts or trace downloads.
The stage label organizes the figure, not a claim that every recorded forward
belongs to that stage. The scope must state actual EXTEND/DECODE annotations,
batch/token counts, mixed windows and early-prefill export limitations.

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

The CPU tests exercise pending mode, exact new identity binding, rejection of
old runs, stale health, inconsistent replay counts and mismatched paired context.
Browser QA checks overflow, pending charts, source values, timing gaps, replay
status, local-only assets and keyboard/mobile navigation. Synthetic unit inputs
exist only in temporary test directories, never in the canonical pending deck.

Source files are `generate.py`, `experiment.json`, `template.html`, `assets/*`,
`test_generate.py`, `check_slides.mjs` and this README. `site/` and `qa/` are generated
artifacts. Plotly3.1.0 is bundled locally; the CSS/navigation and paired-timing
selection reuse the original report's design and evidence logic. No runtime,
collector, cluster, model, or original-report file is modified by this builder.

## Optional actor-update analysis

`--actor-profile /absolute/path/actor-profile.json` adds verified actor evidence
and actual timeline PNGs in existing slide16 only. Slide10 stays dedicated to
main stage timings. Optional concise findings need hashed audit receipts; missing partner
captures stay explicit. The deck remains16 slides and all generation
curves/figures remain unchanged. Missing input preserves the current
layout. Individually verified unmatched diagnostics are supported; matched
workload is a separate, explicit attestation. See [ACTOR_PROFILE.md](ACTOR_PROFILE.md)
for raw per-update category fields, timing semantics, identity and attachment
hash requirements. The current v2 capture is one rank0 forward/backward
microbatch (update1/index1); optimizer/final gradient sync stay unmeasured.
Visible actor findings use the same microbatch window and seconds throughout;
whole-update diagnostic times and main token-normalized ratios stay in the raw
audits, not on the actor slide. Kernel coverage is not GPU utilization.
Never substitute a main actor-timer ratio for measured
forward/backward/recompute or kernel-category evidence.
