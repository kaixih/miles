# Matched TRTLLM prefill/decode captures

This is a **new** diagnostic for the full TRTLLM BF16 GB300/Rubin experiment.
No existing report, main run, model or installed library is changed by these
helpers. Main training finishes before profiling. Its unprofiled correctness
and performance records stay separate.

## Budget and prerequisites

- Reserve **30 minutes per platform** for fresh one-GPU model load, FlashInfer
  JIT/graph creation, warmup and capture, plus **5–10 minutes** for retention.
  The run has one immutable absolute deadline at most 30 minutes away. A
  separate host guard must stop only the exact diagnostic container before the
  original Slurm lease expires. Startup alone can use up to 20 minutes.
- One GPU; TP1/EP1 BF16, Triton attention, Torch BF16 GEMM, **FlashInfer TRTLLM**
  MoE; static memory fraction0.55, max running requests128, context1536,
  seed1234, decode graph **ON**, prefill graph **OFF**. This matches an individual
  main-run rollout engine; four-engine host contention is not reproduced.
- Normal UID28644:GID30; verify actual image digest, container ID and label,
  one visible GPU, read-only source/model/data/input mounts, writable scoped
  output/cache mounts, and storage write/read/delete probes. Assert main Ray
  success, all50 rollouts/200 updates, no active workloads, and exact main
  container stopped before creating this new container. Do not reuse old
  `profile_operator.py` hardcoded jobs/deadlines.
- Reuse already staged original **Qwen3-30B-A3B HF weights** from the main run.
  No training checkpoint or model download is needed. Record the common initial
  model inventory ID; both platforms must use identical model file inventory,
  tokenizer and source dataset.
- Output cap **1GiB**, decoded trace cap128MiB/file, engine host RSS cap192GiB.
  Expect a few to tens of MiB compressed traces; retain exact actual sizes.
  Fresh local JIT caches are outside the output cap; reserve20GiB free local
  storage beyond the staged model. Preserve partial evidence on failure.

The existing successful wrapper and its cleanup fixes are imported from
`lab/rubin_two_node/run_sglang_graph_diagnostic.py` and
`sglang_graph_capture.py`. Keep this repository layout in the read-only source
mount. The adapter changes only the backend and captures ON once; it does not
run another OFF comparison.

## Freeze once, bind identically on both platforms

Inside a prepared platform container, CPU-only:

```bash
python3 /opt/miles/lab/rubin_trtllm_full/profiling/run_profile.py freeze \
  --model-path /models/Qwen3-30B-A3B --dataset /inputs/train.jsonl \
  --output /run-output/shared-frozen-token-input.json
```

Retain that exact file and its printed SHA256, then bind the identical file
read-only at `/profile-inputs/frozen-token-input.json` on both platforms. Do not
freeze independently and assume equivalence. The run re-tokenizes the dataset
on each image and rejects any mismatch against the frozen IDs.

The first128 eligible GSM8K prompts (at most512 input tokens each) use the
unchanged default thinking template. Previously this was10,883 input tokens;
the actual new file determines the reported count. Every request produces
exactly64 new tokens with ignored EOS, temperature1/top-p1/top-k−1/seed1234.
This short diagnostic output policy does not enter learning curves or rewards.

## Exact inside-container command

The host operator writes `/run-output/diagnostic-identity.json` after inspecting
the new container. Use the existing identity schema from
`lab/rubin_two_node/SGLANG_GRAPH_DIAGNOSTIC.md`, plus `main_run_id`; record the
actual tested immutable platform image reference, not a tag. Keep full Docker
inspection and completed-main evidence beside this attestation.

```bash
python3 /opt/miles/lab/rubin_trtllm_full/profiling/run_profile.py run \
  --identity-file /run-output/diagnostic-identity.json \
  --image 'ACTUAL_REPOSITORY@sha256:ACTUAL_TESTED_DIGEST' \
  --main-run-id ACTUAL_COMPLETED_MAIN_RUN_ID \
  --model-path /models/Qwen3-30B-A3B --dataset /inputs/train.jsonl \
  --frozen-token-input /profile-inputs/frozen-token-input.json \
  --frozen-input-sha256 ACTUAL_SHARED_FILE_SHA256 \
  --output-dir /run-output/profile --cache-dir /cache/sglang-trtllm-profile \
  --host ACTUAL_NODE_IPV4 --port 31081 --deadline-utc ACTUAL_ABSOLUTE_UTC
```

This only prints the plan. Execute the same command with `--execute` after the
host guard is armed. Supply `CUDA_VISIBLE_DEVICES=0` and
`MILES_PROFILE_RUN_ID=ACTUAL_CAPTURE_RUN_ID` in the container environment.
The resolved `/server_info` must confirm the exact backend/dtype/attention/GEMM
and graph configuration before any requests.

Each batch requires a verified radix-cache flush. The engine runs warmup1,
unprofiled measurements3, then a separately armed bounded CPU/GPU profile of
one batch. The legacy stage profiler captures prefill and four decode forwards;
the actual annotations, not filenames, determine which stage was recorded.
The wrapper requires complete prefill, four decode forwards and graph evidence.

## Retain, independently analyze, render

Stop only the exact owned diagnostic after its wrapper terminates. Retain all
profile files with source-before/source-after/destination SHA256 checks. Keep
the immutable plan, identity, terminal, engine log/server info, frozen input,
response receipts, capture claim/receipt, source hashes and raw gzip traces.

After copying verified artifacts locally, run once for each platform with a
fresh analysis directory:

```bash
python3 lab/rubin_trtllm_full/profiling/analyze_profile.py \
  --directory /ABSOLUTE/RETAINED/profile --platform rubin \
  --main-run-id ACTUAL_COMPLETED_MAIN_RUN_ID \
  --plan-sha256 RETAINED_PLAN_SHA256 --terminal-sha256 RETAINED_TERMINAL_SHA256 \
  --output /ABSOLUTE/NEW/rubin-profile-evidence --png
```

This checks runtime backend and exact input/measurement/trace identities. It
reuses the original timeline renderer to create prefill/decode HTML and PNG
screenshots, plus `profile-evidence.json`. Selection is the earliest complete
batch128 forward, never the fastest. Compare both platforms' `workload_sha256`,
`input_ids_sha256`, frozen-file SHA, initial-model inventory, and selected
forward `fields` before calling the profiles matched. Generated token values
and expert routing may differ even when input/context/output token counts
match; record this distinction.

Decode proof requires a matching CPU/GPU annotation, same-thread graph launch,
and actual GPU graph-node kernels with that launch's correlation ID. Report
the **GPU annotation elapsed time**, GPU kernel cumulative sum and interval
union separately. `cudaGraphLaunch` CPU submission time is not graph execution
time. Uncovered GPU timeline intervals do not identify CPU overhead or physical
GPU utilization by themselves. Profiler overhead remains in these windows.

The three HTTP timings include prefill/decode/queueing/response; they are neither
pure decode timings nor the four-engine Miles generation timer.

## Report reuse

For the **new** HTML slides directory, reuse the design/assets/navigation in
`reports/rubin-gb300-qwen3-cudagraph` and its `check_slides.mjs`; do not overwrite
that report. Its `generate.py` supports exact new run bindings and four separate
profile records. The old `prepare_diagnostic_report.py` adapter is tied to old
host-operator/OFF–ON contracts, so use the new `profile-evidence.json` to populate
the new report inputs. Original `reports/rubin-gb300-qwen3/render_trace.py`
provides the unchanged actual-trace visual style.

CPU contract checks:

```bash
python3 -B lab/rubin_trtllm_full/profiling/test_profile_contract.py
```
