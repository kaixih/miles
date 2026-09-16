# Optional actor-update evidence in the existing 16 slides

Pass `generate.py --actor-profile /absolute/path/actor-profile.json` only after
retaining a bounded actor diagnostic. Nothing is discovered or executed remotely.
Absent input leaves the stage slide and its main timing values unchanged. Present
verified input adds a compact table and actual screenshot/trace links **inside
slide10**, plus the input hash on slide16. It never adds a slide or changes the
learning curves, generation figures, original report, or main stage statistics.

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
scope: concise diagnostic workload description (1–180 characters)
runs: [zero, one or two platform records]
matched_workload: {verified: false}
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
Use actual update IDs, and make the rank/window scope explicit. Values from
multiple ranks are not automatically wall time. Both platforms use the same
named timing basis; do not mix GPU kernels with CPU enqueue time in one table.

Forward/backward/recompute/MoE/communication can be nested or overlap. The table
does not add them or draw a partition/pie chart. Audit receipts must explain
range attribution, recomputation nesting, communication overlap, profiler
overhead and any unattributed events. A trace's existence alone does not prove
an operation-category assignment. Screenshots must render actual retained trace
events and preserve their scope; no decorative or synthetic runtime figures.

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
samples, separate platform states, the fixed16-slide count and overflow. Run it
on a fresh temporary build before the root task rebuilds the real presentation.
