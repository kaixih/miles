# Profile replay operator (prepared; not executed)

`profile_operator.py` is specific to the two allocations embedded in
`profile_operator.py`. Its default prints JSON without SSH, filesystem writes or process
control. Transfer it to **dl3** and execute only there as **UID28644:GID30**.
It uses the existing reviewed `profile_qwen3_replay.py`; it does not alter recipes,
images or packages. Both fresh containers set `nofile=65535:65535`, explicitly
recording the FD-limit correction found during the GB300 main run.

## Reviewable sequence

Set `OP` to the copied script's path on dl3. Repeat for `rubin` and `gb300`, with
different unique run IDs. These IDs are examples; do not reuse one after creation.
Run the platforms asynchronously: Rubin may profile as soon as its own main run
finishes. Record its checkpoint ID, then require that identical ID when GB becomes
ready. There is no requirement to wait for both main runs before starting Rubin.

```bash
python3 "$OP" rubin --run-id rubin-qwen3-final-profile
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile

# Only after the respective main job has fully completed and Rubin checkpoint49 exists:
python3 "$OP" rubin --run-id rubin-qwen3-final-profile --action prepare \
  --execute --allow-stop-completed-main
# Inspect the Rubin plan and record checkpoint.id.
python3 "$OP" rubin --run-id rubin-qwen3-final-profile --action submit \
  --execute --expected-checkpoint-id ACTUAL_SAME_64_HEX_ID
# Later, after GB main finishes; it must match the already recorded Rubin ID.
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile --action prepare \
  --execute --allow-stop-completed-main
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile --action submit \
  --execute --expected-checkpoint-id ACTUAL_SAME_64_HEX_ID

# After each exact replay job is terminal, retain before stopping its container:
python3 "$OP" rubin --run-id rubin-qwen3-final-profile --action retain --execute
python3 "$OP" rubin --run-id rubin-qwen3-final-profile --action stop --execute
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile --action retain --execute
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile --action stop --execute
```

Preparation rejects absent/partial evidence: exact original Ray ID and entrypoint,
`SUCCEEDED`, both launcher exits0, 50 logged complete rollouts/200 updates, final
shared tracker49, and no active Ray submission. GB main retention must also have
succeeded. Immediately before the exact completed main-container stop, it repeats
these checks. No broad process cleanup is used. Remaining GPU processes cause a
refusal; they are never killed. Partial preparation leaves its files for inspection and can already have stopped
the completed main container. It is deliberately not auto-resumable: inspect the
recorded stage, then use reviewed exact-path cleanup or an explicitly chosen new
run ID. Do not perform broad cleanup or blindly rerun prepare.

Both new clusters use TCP27379, dashboard29265, workers27400–27999 and explicit
agent/metrics ports27380–27385. Both original source plans were inspected: neither
contains `RAY_ADDRESS`, `RAY_API_SERVER_ADDRESS`, or `MILES_SCRIPT_EXTERNAL_RAY`.
The operator rejects their unexpected presence instead of silently rewriting them.
New container environment routes to its own cluster; original platform NIC/IP and
Megatron paths stay distinct. A new GB container gets only scoped `/root` traversal
`chmod o+x` (preserving existing permission bits); its upstream image and installed packages remain unchanged.

## Inputs and retention

| Item | Rubin | GB300 |
|---|---|---|
| Original source plan | `/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260915-j2198331/original-ray-job.json` | `/home/scratch.kaixih_ent/repro/miles-gb300-qwen3-gsm8k/20260915-j2198810-nfs/original-ray-job.json` through read-only `/mnt/cifs` host alias |
| Input files | Original main root `inputs/` | `/tmp/miles-gb300-j2198810/run/inputs/` |
| Shared actor checkpoint | `/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260915-j2198331/checkpoints` | Same directory through read-only `/mnt/cifs` host alias |
| Container checkpoint | `/profile-checkpoint`, read-only | Same |
| New output/cache | `/tmp/miles-profile-RUN_ID/{run,cache}` | Same pattern, local to GB node |
| Durable trace destination | Original main root `/profiles/RUN_ID/node-output/` | Original durable login-NFS main root `/profiles/RUN_ID/node-output/` |

Repo, model, source-plan, checkpoint and `/run-output/inputs` binds are read-only.
Only new output/cache paths are writable. The two original inputs were read-only
verified identical; preparation repeats their SHA checks. The source repository
is the current checked-out committed recipe, mounted read-only and recorded at
prepare time; do not edit it while replays run.

Retention is `rsync` **from dl3's normal UID to direct NFS**, never a GB CIFS
writer, with no100MiB exclusion. It hashes every file, checks source stability and
retained UID/hash/size, and records the exact terminal Ray job. `stop` rechecks
identity and unchanged artifacts, then uses `ray stop` inside only that profile
container and stops that exact container. It does not remove containers or traces.
If retention fails, inspect and retry retention; do not remove the local trace.

## Time, size and interpretation limits

Recorded leases end Rubin06:09:18UTC and GB07:09:02UTC on2026-09-16. The budget is
recomputed before submission: `min(4500s, lease remaining − 1200s retention)`.
Fewer than15 minutes available causes refusal. Loading counts against the budget;
a slow cold load may prevent useful profiling. There is no lease extension.
The existing detached guard limits traces to10GiB and stops only the exact profile
submission. Its checks are polled: bytes can overshoot and Ray API failures can
delay stopping. Twenty minutes is a reserve, not a guaranteed transfer duration.

Replay keeps optimizer/RNG resume, runs3rollouts/12updates, disables save/eval and
profiles `train_overall` start1/end2. Counter selection is relative to the new actor;
it covers first-rollout tail through the second training stage on all4ranks.
Cross-version optimizer-checkpoint loading is not yet validated. If it fails,
preserve the failure rather than changing the images or declaring comparable traces.

SGLang is a separate capture: obtain one TP1 replay-engine URL from its own logs,
then POST `/start_profile` with the helper's printed payload (`num_steps=4`, CPU/GPU,
stage split, no stack/shapes/merge). Never target the primary learning job. The
operator deliberately does not guess an engine URL or issue generation requests.
Actual captured workload/backend and resumed checkpoint must be confirmed from logs.
