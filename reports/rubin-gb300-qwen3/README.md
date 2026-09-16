# Qwen3 Rubin / GB300 HTML slides

An offline, 16:9 presentation. It reads collected evidence only and never contacts a
cluster, runs inference, or computes a replacement reward. All chart points come
from the supplied JSON. Missing inputs show **pending**. Partial runs retain their
status. The current Miles comparison and the historical VeRL reference are separate.

```bash
python3 reports/rubin-gb300-qwen3/generate.py \
  --runs /path/to/qwen3-comparison.json \
  --build lab/rubin_two_node/rubin-build-report.json \
  --profiles /path/to/profiles.json \
  --historical lab/rubin_two_node/qwen3-gb300-baseline.json \
  --run-health reports/rubin-gb300-qwen3/snapshot-health.json \
  --output reports/rubin-gb300-qwen3/site
```

Open `site/index.html` directly, including from `file://`. The directory is portable.
There are 13 main slides and two appendices, plus one optional measured decode
comparison slide when verified trace metrics are supplied. Arrow keys, Space, Page Up/Down,
Home/End navigate. **F** toggles fullscreen and **O** opens the overview. Printing
uses landscape pages. Chart toolbar export produces standalone PNG figures.

The historical argument is optional if `--runs` already contains
`historical_baseline`. A draft with no supplied inputs is valid:

```bash
python3 reports/rubin-gb300-qwen3/generate.py
```

## Inputs

For the recorded allocation pair, the desktop evidence collector is
`lab/rubin_two_node/collect_qwen3_snapshot.py`. It downloads exact incremental log
bytes through `dl3`, validates prefix and reconstructed-file hashes, and installs
a snapshot only after both platforms and the staged summary pass validation.
Its explicit `outputs/rubin-gb300-qwen3/io-overlap-windows.json` input records
background-copy windows. Timing exclusions combine warmup, observed checkpoint
saves, copying and unknown coverage, including the documented following-rollout
guard. These exclusions do not change reward, gradient or completion counts.

`--runs` uses `miles-qwen3-comparison-v1` from
`lab/rubin_two_node/summarize_qwen3_runs.py`. The deck uses `rows[].common`,
`train_steps`, `eval`, and `unprofiled_stage_statistics` directly. It does not smooth
curves, fill missing points, or infer completion from row count. Metadata should
include `status`, `image`, `versions`, `gpus`, `expected_rollouts`, and
`optimizer_steps_per_rollout`. Recipe aliases supported for concise slide display:

```json
{
  "recipe": {
    "model": "Qwen3-30B-A3B",
    "dataset_reward": "GSM8K / strict #### scorer",
    "batch_description": "256 prompts × 8 responses",
    "length_description": "1024 response / 512 prompt tokens",
    "learning_rate": "1e-6",
    "sampling_description": "T=1, top-p=1",
    "group_filter": "None"
  }
}
```

These strings are display metadata, not defaults. Actual missing values say
“Not recorded.” The full original metadata remains in `report-data.json`.

`--build` accepts `lab/rubin_two_node/rubin-build-report.json` with
`components[{name,version,rationale}]`, `changes[{title,detail,scope}]` and the
remaining source/validation fields. Keep displayed component reasons concise.
The build slide groups the recorded CUDA/Torch, SGLang, TE, FA2/Apex,
Megatron, mbridge/TMS and FLA/Triton entries. A separate compatibility slide
shows the native constraints, scoped Qwen3.5 d256 patch, and FA4 package limitation.
The full input remains downloadable and keeps every source reference.

`--profiles` uses the following contract. Relative paths resolve against the JSON
file. The generator copies only explicitly referenced local attachments, gives
them content hashes, and preserves their source paths. HTTP asset URLs are rejected.

```json
{
  "schema": "qwen3-profile-evidence-v1",
  "status": "partial",
  "summary": "Only write a measured gap observation supported by the traces.",
  "profiles": [
    {
      "id": "rubin-train",
      "run_label": "rubin",
      "title": "Actual capture title",
      "tool": "PyTorch profiler / Nsight Systems",
      "status": "complete",
      "scope": "Actual optimizer steps, ranks, and warmup coverage",
      "observations": ["A short measured observation, with units."],
      "image": "profile-rubin.png",
      "trace": "profile-rubin.json.gz",
      "caption": "Say whether this is an actual UI screenshot or a rendering of trace data.",
      "source_log": "/absolute/source/log",
      "source_log_sha256": "..."
    }
  ]
}
```

The two profile slides select the first record whose `run_label` or title contains
`Rubin` / `GB300`. All records remain in the downloadable JSON. Up to three concise
observations appear with each capture. The image opens at original resolution.

### Optional actual decode comparison

To add the measured interval slide immediately after those two profile slides,
embed the complete independently checked analysis object (not manually entered
means) in the existing profile input:

```python
profiles["decode_comparison"] = json.loads(
    Path("outputs/rubin-gb300-qwen3/profiles/sglang-decode-comparison.json").read_text()
)
```

This object's schema is `sglang-same-nominal-decode-bs128-v1`. Both
`provenance.<rubin|gb300>.verified` fields must be true and contain exact
`trace_sha256` values. `selection` records the exact annotation, per-platform
counts, indices and all captured annotation counts. `forward_statistics` retains
the raw per-forward `gpu_annotation_span_ms.values` and `kernel_union_ms.values`,
each with `n` and `mean`. The renderer independently calculates means from those
arrays, validates counts/hashes/means, and rejects a union exceeding its forward
span. Missing or unverified evidence leaves the original 15-slide layout intact;
inconsistent evidence claiming verification fails visibly. Verified input adds
slide13 and shifts conclusions/appendices, giving16 slides.

The stacked bars show kernel interval union and the remaining GPU annotation
span. The remainder includes gaps, copies and unknown time; it is neither CPU
time nor production utilization. For the retained captures, the actual subset
is Rubin N=4 versus GB300 N=1 at `DECODE bs=128`. GB's other decode/extend batches
stay outside this comparison. Context lengths and expert routing are not proven
matched. `cpu_api_statistics` optionally supports the qualified launch-API note;
runtime/driver totals may nest and are never added into the stacked bars.

The full input, source hashes and raw forward arrays remain in `report-data.json`.
`window.DECODE_GAP_DATA` exposes the checked derived stack for browser inspection.
`check_slides.mjs` verifies both stack components against raw values, displayed
unequal counts, dynamic15/16 slide count, and negative cases for false verification,
wrong mean/count/hash and impossible kernel unions. No fabricated traces are
needed for these checks; mutated validation objects are never rendered.

## Evidence and offline assets

- `report-data.json`: complete inputs, input SHA256s, attachment paths and hashes.
- `asset-manifest.json`: hashes of the generated site files.
- Plotly 3.1.0 is bundled from `https://cdn.plot.ly/plotly-3.1.0.min.js`, including its
  original license header. It loads locally and makes no CDN request.
- Image and trace assets remain local. No web fonts, external scripts, or telemetry.
- Invalid JSON/schema causes an error. A missing optional file shows pending.

The static snapshot does not refresh while experiments run. Re-run the generator
with freshly collected input after the experiment reaches its intended state.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s reports/rubin-gb300-qwen3 -p 'test_*.py' -v
```

## Real trace rendering

`render_trace.py` accepts a Torch / Chrome `.json` or `.json.gz` and emits a local
interactive HTML timeline, exact source SHA256, and a JSON aggregation. `--png`
captures this same page through local Chrome. This is explicitly labeled as a
rendering of original trace events, never as an Nsight UI screenshot.

```bash
python3 reports/rubin-gb300-qwen3/render_trace.py /path/to/actual.trace.json.gz \
  --output reports/rubin-gb300-qwen3/traces/rubin-step-12 \
  --title 'Rubin: optimizer step 12, rank 0' \
  --start-ms 0 --duration-ms 50 --device 0 --png
```

The window starts relative to the first GPU event on the selected device. Use
`--origin trace` to start relative to the first complete event instead. Source
timestamps and durations follow Chrome Trace Event microseconds. Categories
`kernel`, `gpu_memcpy`, and `gpu_memset` are GPU events. `cuda_runtime`,
`cuda_driver`, and CPU operations remain CPU events. Unknown categories stay
unclassified. Only complete `ph=X` events contribute durations.

Kernel totals use duration clipped to the selected window. They include every
intersecting kernel even when `--max-visible-events` caps the drawn timeline.
Overlapping streams can produce a cumulative duration above elapsed time, so
the renderer never converts it into wall utilization. Counts expose CPU/GPU/
unknown classifications and any timeline cap. The helper does not infer model
steps or GPU identity beyond fields in the trace; provide that scope explicitly.

Add the resulting `trace.png` and original raw trace to `profiles.json`. Synthetic
fixtures live only in isolated test/QA directories and must never enter that file.

Browser QA for slides (local macOS Chrome and the bundled Playwright runtime):

```bash
/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  reports/rubin-gb300-qwen3/check_slides.mjs \
  reports/rubin-gb300-qwen3/site reports/rubin-gb300-qwen3/qa
```

This captures every slide and checks overflow, browser errors, external requests,
keyboard navigation, mobile scaling, visible hardware identity, and the exact
step and generation-throughput chart selections against the embedded source rows. Inspect the
screenshots as well.

The runtime and generation-throughput charts use the same cohort: only rows
explicitly marked completed, unprofiled, and timing-eligible. Rollout 0 and
metadata warmup exclusions are omitted from both curves. In particular, GB300's
first generation includes FD-limit recovery and queued backlog; it is not a
steady-state baseline. No points are smoothed or filled. The recorded first-step
total remains visible in the footnote and all raw measurements remain unchanged
in `report-data.json`.

The scope slide displays `metadata.hardware.name`, `memory_mib_per_gpu`, and the
engineering-sample qualification. The recipe slide shows recorded parallelism,
training-token budget, rollout kernel choices, and CUDA graph settings.
Training-token budgets appear in a separate recipe-table row from
`metadata.recipe.max_training_tokens_per_gpu` (or the same field directly under
`metadata`). If recorded budgets differ, the recipe, performance and findings
slides identify the mismatch and reject a hardware-only speedup interpretation.
Equal budgets retain the normal presentation without a mismatch warning; missing
values remain unrecorded. Browser QA checks both equal and unequal cases against
the supplied metadata, never fills or alters the experiment measurements.

Optional `--run-health snapshot-health.json` accepts `miles-run-health-v1` with
`runs[{label, run_id, run_health, issue}]`. It records explicit investigation or
recovery notes without changing the collector JSON; identities must match.
Keep its observation time and evidence references current. Native comparison
`metadata.run_health` / `metadata.issue` takes precedence over this sidecar.
Ray `RUNNING` alone never suppresses a recorded issue. Unknown evaluation weight
phases remain visible, including when no optimizer updates have been observed.

Optional build `reproduction` fields (`repository`, `branch`, `dockerfile`,
`build_helper`, and local evidence paths) supply the appendix entry points.
`reproduction.txt` preserves those fields and the exact `image.reference` pull
command. Appendix hashes are explicitly labeled prefixes; full hashes remain in
the JSON and asset manifest. The build-stage flow is a conceptual summary, not a
claim about the exact execution order.

### Paired stage bars

The stage bars use `derived.paired_timing`, calculated directly from the supplied
`--runs` bytes by `generate.py`. The cohort is the intersection of exactly two
runs' completed, explicitly unprofiled, timing-eligible rollout IDs, excluding
rollout0 and every configured timing exclusion. Both bars use the same IDs and
show N/IDs on the slide, with count and median in hover text. The derivation is
bound to the input SHA256; it does not read an external analysis summary.

Individual step and generation-throughput curves keep **all eligible points for
each run**. Original `unprofiled_stage_statistics` and raw observations are left
unchanged in `inputs.comparison`. An empty intersection remains pending. If either
run lacks a finite value for one stage anywhere in the shared cohort, that stage
is pending on both sides; the renderer never silently shortens one side's cohort.

The findings may describe the measured step means and largest observed stage-timer
gap. This is a system-level observation: nested timers are not additive, generated
shapes can differ, and hardware causation awaits profiling. The existing differing
training-budget and software/image cautions remain visible.

`test_generate.py` covers intersection selection, exclusions, preservation of raw
statistics, empty cohorts and missing stage values. `check_slides.mjs` independently
recomputes the cohort and stage means/medians from raw rows, then checks both the
derived JSON and rendered bar values. It also continues to verify the unchanged
per-run line-curve selections.

## Retained September16 result

[Final measured results](FINAL_RESULTS.md) records both completed50-rollout/200-update
learning runs, the shared17-rollout timing cohort, and the actual short SGLang
decode comparison. Compact raw statistics and source hashes are checked in under
`evidence/`. The generated portable site also embeds the full comparison and
profile inputs and carries the original short SGLang gzip traces.

The separate trainer profiling replays did not produce complete usable trainer
traces: Rubin reached its original profiling deadline during export; GB300 hit
Ray's host-memory threshold during profiler finalization. Those outcomes do not
change the successful main runs or the independently verified SGLang captures.
