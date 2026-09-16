# Optional actor-update evidence in the existing 16 slides

Pass `generate.py --actor-profile /absolute/path/actor-profile.json` only after
retaining a bounded actor diagnostic. Nothing is discovered or executed remotely.
Absent input leaves the stage slide and its main timing values unchanged. Present
verified input adds a compact table and actual screenshot/trace links **inside
slide10**. Existing slide16 displays supplied actor timeline PNGs inline for both
platforms, with explicit pending slots, audited findings and compact provenance. It never adds a slide or changes the
learning curves, generation figures, original report, or main stage statistics.

A missing image remains pending even if its trace is verified. Images are shown
without cropping and link to full resolution; prepare a focused timeline window
with readable labels for a roughly700px-wide panel. An absent second platform
never yields an inferred timing or ratio. At most two concise findings fit beside
the retained provenance; supplied observations must cite hashed audit evidence.
The builder validates this contract but does not establish causal explanations.
Do not state that a trace explains the main stage gap without separate evidence.

The actor table is instrumented diagnostic evidence, separate from the main
unprofiled actor timer. Individually verified captures are useful even when the
other platform is missing or workloads differ. Such data displays as
**Diagnostic / unmatched**, with no causal ratio. The observed main actor gap
is never relabeled as a measured forward/backward ratio.

## JSON contract

Top-level fields:

```text
schema: "qwen3-actor-profile-v1"
experiment_id: exact existing experiment ID
measurement_basis: "gpu_kernel_interval_union_ms" | "cpu_annotation_elapsed_ms"
capture_window: "whole_optimizer_update" | "single_forward_backward_microbatch"
scope: concise diagnostic workload description (1–180 characters)
runs: [zero, one or two platform records]
matched_workload: {verified: false}
findings: optional array of at most two observations, each {
  verified: true only after source attribution audit,
  text: single-line plain text, 1–120 characters (200 characters total across findings),
  run_labels: nonempty distinct subset of individually verified platforms,
  evidence_refs: [{path: local audit receipt, sha256: exact SHA256}]
}
interpretation_limits: optional single-line text, 1–140 characters
```

Each platform record:

```text
run_label: exact platform binding, e.g. "rubin"
source_run_id: exact bound main run
capture_run_id: actual independent actor capture identity
verified: true only after the source audit
scope: actual selected workload and trace/rank limits
rank_ids: explicit nonnegative rank IDs
samples: [{update_id: actual ID, durations_ms: {
  forward: measured value or null,
  backward: measured value or null,
  recompute: measured value or null,
  moe: measured value or null,
  communication: measured value or null,
  optimizer: measured value or null
}}]
unattributed_reasons: {category: explanation for each null category}
trace: local retained trace path
source_trace_sha256: SHA256 of that exact file
image: optional actual screenshot/rendering path
source_image_sha256: required if image is supplied
evidence_refs: [{path: local audit receipt, sha256: exact SHA256}]
input_identity: optional IDs described below
```

All six categories must be explicit. Unknown is `null`, never zero. A displayed
category mean requires values for every selected update; a missing observation
leaves the whole category unknown rather than silently shortening its cohort.
Use actual update IDs and an explicit capture_window; units follow that window. Values from
multiple ranks are not automatically wall time. Both platforms use the same
named timing basis; do not mix GPU kernels with CPU enqueue time in one table.

Forward/backward/recompute/MoE/communication can be nested or overlap. The table
does not add them or draw a partition/pie chart. Audit receipts must explain
range attribution, recomputation nesting, communication overlap, profiler
overhead and any unattributed events. A trace's existence alone does not prove
an operation-category assignment. Screenshots must render actual retained trace
events and preserve their scope; no decorative or synthetic runtime figures.

## Current v2 capture: one forward/backward microbatch

The new hook captures **rank0, update1, second microbatch (zero-based index1)**,
not the whole update. Use `capture_window="single_forward_backward_microbatch"`.
Each verified record must have `rank_ids:[0]`, exactly one sample with
`update_id:1`, `microbatch_index:1`, and `final_gradient_sync:null`.
Set `samples[0].durations_ms.optimizer=null` with an explicit
`unattributed_reasons.optimizer` saying it lies outside the captured window.
The generator refuses numeric optimizer/final-gradient-sync values for this
scope and labels table units **ms / microbatch**. Zero would be a false timing.

The `communication` field covers only attributed communication inside this
selected F/B window (such as MoE dispatch/combine); it is **not** the final
gradient synchronization cost. If attribution is unavailable, set it to null.
Likewise, a recompute-plus-backward range is not pure recompute: keep the latter
unknown unless a separate audited method isolates it. Do not infer microbatch
cost by dividing a whole-update timer, or extrapolate this trace into 1.77×.

Example concise scope: `Rank0, update1, microbatch1 (0-based): one F/B pair;
optimizer and final gradient sync excluded.` The trace filename may still be
`actor-update.json.gz`; its receipt's `capture_window`, selected packing and
observed schedule identify what was captured. Retain the COMPLETE receipt and
trace-analysis audit as evidence references. Screenshot the actual selected
microbatch; no optimizer lane should be invented. Both main runs' four-update
actor timers and their observed ratio remain separate.

For a **matched** microbatch claim, additionally provide the same
`input_identity.selected_microbatch_sha256` on both records, audited from the
ordered selected tokens/lengths/global identities in the native receipts. One
shared 2048-sample dump alone does not establish the same local packed
microbatch. If packing or relevant input identity differs, the verified traces
still display as diagnostic/unmatched. No bitwise post-warmup equivalence is
assumed by either mode.

## Optional matched-workload attestation

Trace verification does **not** require bitwise optimizer/model equivalence.
If a stronger matched-workload comparison is supported, set:

```text
matched_workload: {
  verified: true,
  identity_scope: exact meaning and limits of the common initial construction,
  evidence_refs: [{path, sha256}]
}
```

Then each of the two verified platform records must supply equal 64-character
SHA256 identity fields:

```text
input_identity: {
  initial_state_id: immutable release + fresh-optimizer construction identity,
  batch_sha256: exact frozen training batch artifact (tokens/labels/masks/order),
  training_recipe_sha256: common training-workload recipe identity
}
```

Selected rank and update IDs must also agree. Explicitly claiming a matched
workload with mismatched identities is rejected. A shared initial-construction
ID does not claim that post-warmup tensors are bitwise identical; document that
limit in `identity_scope`. Without the attestation, different or missing input
identities are accepted as individually verified, unmatched diagnostics.

The generator checks exact main bindings, raw sample validity, declared trace,
screenshot and audit hashes. It derives category means itself. This is an
artifact/provenance validator, not an independent trace-attribution algorithm;
the source audit must establish what the samples measure.

## Offline validation

```bash
python3 -B reports/rubin-gb300-qwen3-cudagraph/test_actor_profile.py
python3 -B reports/rubin-gb300-qwen3-cudagraph/test_generate.py
```

Tests use clearly marked temporary synthetic fixtures, never canonical report
inputs. Browser QA additionally checks table values/unknowns against the derived
samples, separate platform states, actual inline images and trace hashes, findings
and audit links, the fixed16-slide count and overflow. Run it
on a fresh temporary build before the root task rebuilds the real presentation.
