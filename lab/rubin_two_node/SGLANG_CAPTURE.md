# Capture one independent replay engine

`capture_sglang_replay.py` runs on the compute host as UID28644:GID30. It uses
only Python's standard library and Docker CLI. It does not submit training or
generation requests, change a model/backend, or instrument the main learning job.
The operator/replay helper must already have created a fresh profile container
and submitted the exact replay with its byte/runtime guard. Start this observer
while that replay is actively generating; a15-minute heartbeat is too slow.

The caller supplies an actual engine URL and container-namespace PID/start ticks.
Find them from the **fresh replay's** launch logs and `/proc`, not an older main
log or a guessed port. Select a direct `python -m sglang.launch_server` process,
not its shell wrapper, router, launch gate or scheduler child. Start ticks are
field22 of `/proc/PID/stat` (split after its final `) `, then index19). The script
checks its listening socket, profile run ID, host/port/model/TP/PP, container label,
immutable image ID/digest and normal UID. The exact Ray submission, runtime run ID
and entrypoint SHA must match the submitted `profile-plan.json`.

Example on the relevant compute host, using **actual** captured values:

```bash
python3 /path/to/capture_sglang_replay.py \
  --plan /tmp/miles-profile-RUN_ID/run/replay/profile-plan.json \
  --container miles-profile-RUN_ID \
  --image IMAGE_REFERENCE_AT_SHA256_DIGEST \
  --engine-url http://NODE_IP:ACTUAL_ENGINE_PORT \
  --engine-pid ACTUAL_CONTAINER_PID \
  --engine-start-ticks ACTUAL_PROC_START_TICKS
# Review the plan; repeat the same command with --execute to capture.
```

Default mode makes no Docker or HTTP calls. Execution is single-use: the
`sglang-capture.claim` file prevents automatically arming again after an ambiguous
request or process crash. Output is beneath the new replay output only:

- `/run-output/replay/sglang-capture.json`: identity, single payload/request,
  response code/text or error, original and capture deadlines, result and hashes.
- `/run-output/replay/traces/sglang/`: unique gzip Chrome trace, covered by the
  existing10GiB replay guard and normal-UID retention procedure.

The request uses `activities=["CPU","GPU"]`, `num_steps=4`, and
`profile_by_stage=false`, with stacks, shapes, merge and detailed annotations off.
It records **four consecutive scheduler forwards**, with the observed active
workload. It does not guarantee both prefill and decode, four requests, a complete
rollout, or an unprofiled performance measurement. Label actual scope from trace
events and surrounding replay logs; stage-like kernel names are only hints.
The non-stage manager starts immediately and stops before its fifth forward.

Polling is every2seconds, plus bounded Docker/HTTP call latency. The capture
deadline is the earlier of180seconds and the replay's already-recorded deadline;
it never extends the lease. HTTP200 is not capture success: the gzip must decode
to Chrome events containing both actual CPU operators and CUDA kernels. CPU trace
inspection is bounded to512MiB decoded. There is no extra generation to fill a
trace if the workload ends or pauses.

On timeout the observer attempts one `/stop_profile`. A verified exported trace
after manual stop is labeled **partial**, since four forwards are unproven.
If export/cancellation remains unproven, it requests stop only for the same exact
replay Ray job. It never stops containers, kills process names, or targets a
replacement job after identity changes. Ray/API outages can delay stopping; the
existing independent replay guard remains active. Keep this small observer
attached, or launch it detached with logs before the short generation window;
do not rely on the15-minute parent heartbeat to trigger the capture.

Both audited images contain SGLang source
`aea7fb92c047c9c096eae66460acb25dec9ae5a9`. Its request contract is in
`python/sglang/srt/managers/io_struct.py:2192`; HTTP endpoints at
`srt/entrypoints/http_server.py:1174`; scheduler profile predicates/export at
`srt/managers/scheduler_components/profiler_manager.py:87–160,313–446`.
The observer requires the default legacy mode: V2 manual stop is not implemented
in this pinned source. No real profiling or cross-image checkpoint resume has
been validated by the CPU contract tests.
