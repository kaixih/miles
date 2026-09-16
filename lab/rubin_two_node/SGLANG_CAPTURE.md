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
  --plan RAID_ROOT/miles-profile-RUN_ID/run/replay/profile-plan.json \
  --container miles-profile-RUN_ID \
  --image IMAGE_REFERENCE_AT_SHA256_DIGEST \
  --engine-url http://NODE_IP:ACTUAL_ENGINE_PORT \
  --engine-pid ACTUAL_CONTAINER_PID \
  --engine-start-ticks ACTUAL_PROC_START_TICKS
# Review the plan; repeat the same command with --execute to capture.
```

The same observer supports `--initial-policy` replay plans. It does not read a
checkpoint iteration or require checkpoint9: it compares the entire supplied
plan with the actual mounted plan and checks the exact submission, entrypoint,
run ID, container and engine. `checkpoint:null`, `initial_model_id` and rollout0–1
therefore require no observer runtime change. The helper owns model-manifest
validation. Retain `profile-plan.json` with the trace to establish whether this
was an initial-policy capture or a restored-checkpoint capture.

For the prepared initial-policy runs, set these values on the corresponding
compute host after its independent profile has actually launched:

| Variable | Rubin | GB300 |
|---|---|---|
| `CODE_ROOT` | `/home/scratch.kaixih_ent/repo/miles-rubin-cu134` | `/mnt/cifs/home/scratch.kaixih_ent/repo/miles-rubin-cu134` |
| `RAID_ROOT` | `/raid/dldata/miles-kaixih-j2198331-profile` | `/raid/tmp/miles-kaixih-j2198810-profile` |
| `NODE_IP` | `10.102.74.82` | `10.85.212.9` |
| `IMAGE` | `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c` | `radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986` |

Use the actual new `PROFILE_RUN_ID`, `ENGINE_PORT`, `ENGINE_PID` and
`ENGINE_START_TICKS`; none are inherited from a main run. This command is a
plan-only template; append `--execute` only after reviewing its identities:

```bash
python3 "$CODE_ROOT/lab/rubin_two_node/capture_sglang_replay.py" \
  --plan "$RAID_ROOT/miles-profile-$PROFILE_RUN_ID/run/replay/profile-plan.json" \
  --container "miles-profile-$PROFILE_RUN_ID" --image "$IMAGE" \
  --engine-url "http://$NODE_IP:$ENGINE_PORT" \
  --engine-pid "$ENGINE_PID" --engine-start-ticks "$ENGINE_START_TICKS" \
  --capture-timeout 180
```

Initial-policy traces describe the initial workload, not later learned-policy
timings. The two images differ, so their traces do not isolate GPU hardware.

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
in this pinned source. CPU contract tests include initial-policy plans with no
checkpoint and the same single-arm/exact-stop behavior. No actual GPU capture,
initial-policy replay, or cross-image checkpoint resume is established by them.
