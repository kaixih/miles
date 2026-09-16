# Short generation-only CUDA Graph diagnostic

## Host operator after completed learning

On dl3, use the original immutable main-driver config. Both calls are plan-only
unless `--execute` is supplied; there is no waiting mode or completion override.

```bash
python3 lab/rubin_two_node/run_cudagraph_diagnostic_operator.py \
  --config /EXACT_DURABLE_MAIN_ROOT/driver-config.json \
  --run-id UNIQUE_PLATFORM_JOB_DIAGNOSTIC_ID --order off,on --port 31081
```

After reviewing the plan, repeat with `--execute`. It requires the same RUNNING
Slurm allocation/node/UID and original end time with more than 40 minutes left;
the exact watchdog-locked Ray submission must be SUCCEEDED with the matching
`RUBIN_RUN_ID`, unchanged watchdog deadlines, no stop requests, zero main driver
and train exits, all 50 completed rounds and updates0–199. No other Ray job may
be active. Original source-plan/container/image identity must still match.

The main has a split storage layout. The login-host `run_dir` contains
`logs/qwen3_train.log`, `train_exit.json`, driver/launch receipts and the original
source plan. The node-local `node_run_dir` contains watchdog, checkpoint49
metadata and `logs/gpu-telemetry.csv`; it need not contain the training log or
exit receipt. The operator never synthesizes those missing node files.

Before stopping the exact completed main container, the operator copies and
SHA-verifies closed login artifacts and the original plan's recorded source
files into `<durable-main>/diagnostics/<ID>/main-login`. It parses that retained
training log for 50/200 completion, then rechecks the original login artifact
and source hashes immediately before stop. Login logs have an explicit 512 MiB
per-file allowance; JSON/source files retain a 128 MiB limit and all login
artifacts together are capped at 1 GiB.

Node-local watchdog, checkpoint metadata and telemetry are separately retained
in `main-before`, excluding checkpoint shards, models, inputs and caches.
Appending telemetry is retained as a recorded fixed prefix with before/after
lengths and a source-prefix rehash; post-stop telemetry/watchdog are retained
separately in `main-final`, with actual source stability recorded. The node's
128 MiB per-file/512 MiB total evidence bounds and diagnostic output cap remain
unchanged. Existing files and IDs are never overwritten.

It creates `<node-main-parent>/graph-diagnostic-<ID>/{source,run,cache}`, checks
normal-UID host/container read/write/delete probes, and launches a fresh labeled
`miles-graph-diagnostic-<ID>` container with the same immutable image and only
GPU0 visible. Models, dataset and copied reviewed helper sources are read-only.
The operator creates the identity file described below, including initial-model
file sizes, metadata hashes and source hashes; it never hashes all model tensors.

Before GPU work, a separate node-host process must arm an absolute guard for
that exact new container ID/image/label. This survives SSH loss and stops only
the diagnostic container at `min(start+1800, original-lease-120)`. The inside
wrapper also has its own deadline guard. A failed or timed-out SSH call never
renews either deadline. The host operator attempts bounded SHA-verified output
retention (approximately1 GiB maximum) in a `finally` block, then stops only the
exact labeled diagnostic container. If retention fails, partial durable files
and unchanged node-local files remain; no successful retention receipt is
claimed. Raw partial traces are preserved with explicit validity failures.

The host operator needs the sibling main-driver, metrics-parser, watchdog API
and capture helper modules available in its reviewed source directory. It
copies only the two standalone engine/capture helpers into the new container,
preserving the already-mounted main source snapshot unchanged.

## Inside-container contract

`run_sglang_graph_diagnostic.py` defaults to a CPU-only plan. It runs only with
`--execute`, inside a fresh dedicated container created after the platform's
main learning run is terminal. It never starts Ray or Miles training and never
stops a main job/container. Use the unchanged pinned platform image.

The external operator must inspect the actual Docker container, image ID/digest,
label, UID28644:GID30, one visible GPU, read-only repo/model/data binds and fresh
writable `/run-output` and `/cache` binds. Supply `CUDA_VISIBLE_DEVICES=0` (or one
explicit GPU UUID) and `MILES_PROFILE_RUN_ID=<run_id>`. An inside-container helper
cannot independently query Docker; its identity file is the operator's recorded
attestation, checked against the actual local UID and run-ID environment.

Create the identity JSON outside the new diagnostic output child directory:

```json
{
  "run_id": "rubin-jJOB-graph-diagnostic-v1",
  "container_name": "miles-graph-diagnostic-rubin-jJOB-v1",
  "container_id": "ACTUAL_64_HEX_CONTAINER_ID",
  "image_reference": "ACTUAL_REPOSITORY@sha256:PINNED_PLATFORM_DIGEST",
  "image_id": "sha256:ACTUAL_64_HEX_IMAGE_ID",
  "uid": 28644,
  "gid": 30,
  "labels": {"miles.graph_diagnostic": "rubin-jJOB-graph-diagnostic-v1"},
  "docker_inspect_verified_at": "ACTUAL_UTC_TIME",
  "main_terminal_verified": true,
  "source_commits": {"miles": "ACTUAL_COMMIT", "sglang": "aea7fb92c047c9c096eae66460acb25dec9ae5a9"}
}
```

Retain the full Docker inspection and main terminal evidence alongside this
attestation. Set the absolute deadline from the remaining allocation budget,
at most 30 minutes after launch, with time reserved for artifact retention. Do
not reset a main watchdog, reuse a main output directory, or extend a lease.

```bash
python3 lab/rubin_two_node/run_sglang_graph_diagnostic.py \
  --identity-file /run-output/diagnostic-identity.json \
  --model-path /models/Qwen3-30B-A3B \
  --dataset /inputs/train.jsonl \
  --output-dir /run-output/diagnostic-v1 \
  --cache-dir /cache/sglang-graph-diagnostic \
  --host ACTUAL_NODE_IPV4 --port UNUSED_PORT \
  --order off,on --deadline-utc ACTUAL_ABSOLUTE_UTC
```

Review the printed plan, then repeat with `--execute`. Prefer the opposite
`--order on,off` on the other platform. Both diagnostic engines use TP1, BF16,
Triton attention/MoE, Torch GEMM, 0.55 static memory fraction, 128 maximum running
requests, memory saver, seed1234, context1536 and identical skipped server
warmup. OFF uses the legacy decode disable flag; both explicitly disable
prefill graphs. ON also supplies explicit decode `full`/prefill `disabled` JSON.
The resolved `/server_info` graph configuration must match before requests.
Existing GC behavior is preserved; the wrapper never freezes/unfreezes GC to
work around startup warnings. Full engine logs and runtime server information
are retained for incident review.

The wrapper freezes the first 128 eligible dataset prompts with the installed
tokenizer's default thinking chat template and `add_generation_prompt=true`.
It filters prompts longer than 512 tokens without truncation or duplication.
Record the dataset, token IDs and workload hashes; check cross-platform token
identity before describing requests as identical. Each request uses explicit
sampling seed1234, temperature1/top_p1/top_k-1, 64 new tokens and ignored EOS.
This short output/EOS policy is diagnostic only and is excluded from rewards,
evaluation and learning throughput.

Each mode starts a fresh owned engine process group and fresh JIT caches. After
readiness, it performs one warmup and three measured batches. Every batch,
including the profile batch, first performs POST `/flush_cache?timeout=10` and
requires both HTTP200 and the pinned API's `Cache flushed.` response. This
resets radix cache, not CUDA graphs/JIT state. Responses retain actual cached
token counts; a successful flush does not promise zero shared-prefix reuse
within a batch. Model loading, graph construction and warmup are excluded from
the three measured request times. Cache/JIT construction remains separately
visible in each engine log. This one-engine diagnostic differs from four-engine
main-run host contention.

After those measurements, the wrapper arms one legacy stage profile per mode:
CPU+GPU, four scheduler steps, detailed annotations, no Python stacks, no shape
recording and no merged profiles. It sends one bounded 128-request batch, waits
up to 180 seconds for stable EXTEND/DECODE exports, and validates gzip, complete
JSON and trace hashes. Early prefill flush and interleaved stages are possible;
report actual batch/query/context fields. Missing or incomplete prefill remains
an explicit limitation. A four-DECODE-forward claim requires actual annotations.
No second arm or synthetic recovery request is sent after a failed capture.

The background guard enforces the original absolute deadline, 1 GiB of output
artifacts and a default 192 GiB aggregate engine-process RSS cap (configurable
64–256 GiB). It verifies PID/start ticks, UID, process group and inherited run ID
before signaling only its owned engine group. The guard also runs during HTTP
and tokenizer work. Polling and bounded termination introduce small shutdown
latency; the external container owner should retain an independent lease guard.
Fresh JIT caches under `/cache` are outside the 1 GiB trace/output cap. Caches and
partial trace files are preserved; there is no cleanup of model/shared data.

Outputs include `plan.json`, `events.json`, `frozen-token-input.json`, installed
package versions, and one `{off,on}/` directory containing complete engine logs,
server info, warmup/three measured response receipts, single-arm claim,
capture receipts, raw traces and `summary.json`. `terminal.json` records wrapper
completion or bounded failure. A completed workload does not itself prove graph
replay; consult `capture.decode_graph_replay_proven` and
`capture.four_decode_forwards_verified` separately.

`measurements[].generation_request_seconds` is HTTP wall time including prefill,
decode, queueing and response delivery. It is **not decode milliseconds**.
Actual SGLang timing fields are retained raw without deriving decode time from
them. Pure decode observations come from matched actual DECODE trace spans
(Chrome duration units are microseconds), with batch/context correspondence and
instrumentation overhead disclosed. Actual `cudaGraphLaunch` inside the same
CPU thread's DECODE span plus GPU kernels proves replay for that observation;
flags, capture startup messages and filenames alone do not. Report both modes'
raw repetitions and context distributions without claiming hardware causality.

Retain stable raw traces and receipts with source-before/source-after and
destination SHA256 verification before the external operator stops/removes its
dedicated container. This wrapper only stops its own engines, leaving the
container and output binds for that retention step.
