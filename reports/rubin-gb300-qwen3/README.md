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
  --output reports/rubin-gb300-qwen3/site
```

Open `site/index.html` directly, including from `file://`. The directory is portable.
There are 13 main slides and two appendices. Arrow keys, Space, Page Up/Down,
Home/End navigate. **F** toggles fullscreen and **O** opens the overview. Printing
uses landscape pages. Chart toolbar export produces standalone PNG figures.

The historical argument is optional if `--runs` already contains
`historical_baseline`. A draft with no supplied inputs is valid:

```bash
python3 reports/rubin-gb300-qwen3/generate.py
```

## Inputs

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
