# Profile replay operator (prepared; not executed)

`profile_operator.py` is specific to the two allocations embedded in
`profile_operator.py`. Deploy `checkpoint_metadata.py` alongside the operator,
retention worker and replay helper. Its default prints JSON without SSH, filesystem writes or process
control. Transfer it to **dl3** and execute only there as **UID28644:GID30**.
It uses the reviewed `profile_qwen3_replay.py` and preserves images and packages.
Both replays use the same frozen **iteration-9** checkpoint from local RAID and
explicitly use `--profile-max-tokens-per-gpu 4096`; this changes only
the replay token budget, not either main run. Both fresh containers set `nofile=65535:65535`, explicitly
recording the FD-limit correction found during the GB300 main run.

## Reviewable sequence

The Rubin source is **A2**: `raysubmit_qCmmzUzbwHrBKAiU`, run
`20260916-j2198331-qwen3-a2-mb4096`, container
`miles-rubin-qwen3-j2198331-a2-0`. GB300 remains **A1**:
`raysubmit_W7k1mh2Y9Uh9wmmk`. Its running main job must not be interrupted.

Set `OP` to the copied script's path on dl3. Repeat for `rubin` and `gb300`, with
different unique run IDs. These IDs are examples; do not reuse one after creation.
Run the platforms asynchronously: Rubin may profile as soon as its own main run
finishes all 50 rollouts/200 updates and its own native final checkpoint49 is
verified. Record the frozen checkpoint9 ID, then require that identical ID when GB becomes
ready. There is no requirement to wait for both main runs before starting Rubin.

```bash
python3 "$OP" rubin --run-id rubin-qwen3-final-profile
python3 "$OP" gb300 --run-id gb300-qwen3-final-profile

# Only after natural main completion/final49 and frozen9 + local inputs are verified:
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
native final tracker49, its iteration49 metadata/shards/rollout-state file, and no
active Ray submission. Both main runs use node-local outputs: each
`artifact-retention-exit.json` must report exit0 and UID28644.
The selected profiling checkpoint is separately frozen at iteration9. Its durable
retention record, each local RAID copy fingerprint and staged model inputs must
also pass the prerequisites below. A captured checkpoint9 never counts as main
completion or permission to stop a healthy main run.
Immediately before the exact completed main-container stop, it repeats
these checks. No broad process cleanup is used. Remaining GPU processes cause a
refusal; they are never killed. Partial preparation leaves its files for inspection and can already have stopped
the completed main container. It is deliberately not auto-resumable: inspect the
recorded stage, then use reviewed exact-path cleanup or an explicitly chosen new
run ID. Do not perform broad cleanup or blindly rerun prepare.

Both new clusters use TCP27379, dashboard29265, workers27400–27999 and explicit
agent/metrics ports27380–27385. Preparation checks each actual source plan for
`RAY_ADDRESS`, `RAY_API_SERVER_ADDRESS`, or `MILES_SCRIPT_EXTERNAL_RAY` and rejects
their unexpected presence instead of silently rewriting them.
New container environment routes to its own cluster; original platform NIC/IP and
Megatron paths stay distinct. A new GB container gets only scoped `/root` traversal
`chmod o+x` (preserving existing permission bits); its upstream image and installed packages remain unchanged.

## Inputs and retention

| Item | Rubin | GB300 |
|---|---|---|
| Original source plan | `/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs/original-ray-job.json` | `/home/scratch.kaixih_ent/repro/miles-gb300-qwen3-gsm8k/20260915-j2198810-nfs/original-ray-job.json` through read-only `/mnt/cifs` host alias |
| Main node-local output | `/tmp/miles-rubin-j2198331/run-a2-mb4096` | `/tmp/miles-gb300-j2198810/run` |
| Input files | Main node-local output `inputs/` | Main node-local output `inputs/` |
| RAID root | `/raid/dldata/miles-kaixih-j2198331-profile` | `/raid/tmp/miles-kaixih-j2198810-profile` |
| Staged model roots | RAID root `/models/{Qwen3-30B-A3B,Qwen3-30B-A3B_torch_dist}` | Same relative paths |
| Selected actor checkpoint | RAID root `/profile-checkpoint-9`, read-only | Same relative path and identical checkpoint ID |
| Container checkpoint | `/profile-checkpoint`, read-only | Same |
| New output/cache | RAID root `/miles-profile-RUN_ID/{run,cache}` | Same pattern, local to GB node |
| Durable trace destination | Original main root `/profiles/RUN_ID/node-output/` | Original durable login-NFS main root `/profiles/RUN_ID/node-output/` |

Repo, model, source-plan, checkpoint and `/run-output/inputs` binds are read-only.
Only new output/cache paths are writable. Locally staged models mount at the
original canonical `/home/scratch.kaixih_ent/models` path; recorded model argv is
unchanged. The profile checkpoint mount uses the verified local RAID copy, never
a live trainer checkpoint directory or the slow durable NFS copy. The two original inputs were read-only
verified identical; preparation repeats their SHA checks. The source repository
is the current checked-out committed recipe, mounted read-only and recorded at
prepare time; do not edit it while replays run.

Retention is `rsync` **from dl3's normal UID to direct NFS**, never a GB CIFS
writer, with no100MiB exclusion. It hashes every file, checks source stability and
retained UID/hash/size, and records the exact terminal Ray job. `stop` rechecks
identity and unchanged artifacts, then uses `ray stop` inside only that profile
container and stops that exact container. It does not remove containers or traces.
If retention fails, inspect and retry retention; do not remove the local trace.

### Required frozen checkpoint9 and local input staging

Snapshot capture, copying and staging are **external prerequisites**. This operator
never performs them, overwrites an existing destination, or stops a producer to
capture a checkpoint. The capture/staging workers must retain their actual checks.

A completed iteration9 save may be frozen while Rubin A2 keeps training. Require
tracker9 plus complete iteration9 metadata/shards and rollout-state9 to be stable
before capture. The bounded capture worker may pin immutable completed files via
its own temporary hardlinks; the tracker must be a separate copied value9, never
a hardlink to the producer's mutable tracker. Verify the pinned identity and
source stability, copy through a private partial sibling, verify, then rename to
`/raid/dldata/miles-kaixih-j2198331-profile/profile-checkpoint-9`. Only the worker's
own temporary pin links may be removed. Further main checkpoints must not change
this frozen root. This preserves an earlier workload for later profiling; it does
not start a profile while the main run is active.

Retain that frozen snapshot through **dl3 as UID28644:GID30** at:

- Durable origin: `/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs/profile-checkpoint-9`.
- Durable verification: the same run root `/checkpoint-retention-profile-9.json`.
- GB local copy: `/raid/tmp/miles-kaixih-j2198810-profile/profile-checkpoint-9`.

Check free space and reserve the existing allocation's retention time. Use new
roots; an existing/partial destination needs explicit review, with no blind delete
or overwrite. Never write through the GB CIFS view. Require successful rsync plus
matching shard names/sizes, SHA256 of small metadata and normal UID ownership;
reject symlinks. No extra full tensor-content hashing is required.

The durable record uses this contract (actual values only):

```json
{
  "exit_code": 0,
  "rsync_exit_code": 0,
  "uid": 28644,
  "run_id": "20260916-j2198331-qwen3-a2-mb4096",
  "source_node": "vr-nvl72-ts2-l11-038-c15",
  "source_root": "/tmp/miles-rubin-j2198331/run-a2-mb4096/checkpoints",
  "intermediate_root": "/raid/dldata/miles-kaixih-j2198331-profile/profile-checkpoint-9",
  "destination_root": "/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs/profile-checkpoint-9",
  "iteration": 9,
  "verification": "rsync_transfer_plus_sizes_and_metadata_sha256",
  "source_checkpoint_id": "ACTUAL_FROZEN_HELPER_FINGERPRINT",
  "destination_checkpoint_id": "SAME_ACTUAL_HELPER_FINGERPRINT",
  "files": {
    "latest_checkpointed_iteration.txt": {"bytes": 1, "sha256": "ACTUAL_SHA256"},
    "iter_0000009/common.pt": {"bytes": 1, "sha256": "ACTUAL_SHA256"},
    "iter_0000009/.metadata": {"bytes": 1, "sha256": "ACTUAL_SHA256"},
    "rollout/global_dataset_state_dict_9.pt": {"bytes": 1, "sha256": "ACTUAL_SHA256"},
    "iter_0000009/__0_0.distcp": {"bytes": 1}
  }
}
```

The example shows the legacy format; its sizes are schematic. Include every
copied shard and actual metadata size/hash. The installed Rubin Megatron writes
`common_state/shard_0_1` as `BytesStorageMetadata` inside DCP, so that format has
no `common.pt`. The shared validator requires `metadata.json` to declare
`torch_dist` version1 and statically checks the corresponding DCP metadata entry
without executing pickle content. Both formats are supported; this structural
check does not establish successful optimizer resume.

Fingerprint order matches the replay helper exactly: common.pt only when present,
rollout-state9, .metadata then metadata.json,
and sorted relative shard names/sizes. The source ID describes the immutable
snapshot captured from the original native root, not its later advancing tracker.
Both local copies must independently match this verified durable fingerprint.

Each RAID root must also contain `manifests/input-staging.json`, with
`schema_version:1`, `status:"complete"`, `uid:28644`, `gid:30`, exact `node` and
`raid_root`, and `verification:"shard_inventory_sizes_and_metadata_sha256"`.
`models` contains exactly `Qwen3-30B-A3B` and `Qwen3-30B-A3B_torch_dist`. Each entry
records `source_root`, `destination_root`, `file_count`, `total_bytes`, `files`, and
`fingerprint`. Source roots use actual host paths: canonical scratch on Rubin,
`/mnt/cifs/home/scratch.kaixih_ent/models/...` on GB300. Destination roots are under
that platform's RAID root `/models/`. HF excludes cache/git artifacts; the reference
copy contains only `release/` and `latest_checkpointed_iteration.txt`.

Each `files` entry has `bytes`; all files except `.safetensors`/`.distcp` also have
`sha256`. Its fingerprint is SHA256 of `json.dumps(files, sort_keys=True).encode()`.
The operator rechecks the complete destination inventory, sizes, metadata hashes,
ownership, source/destination identity and absence of symlinks before mounting.
It does not reread model shard contents or copy from NFS during preparation.

Only after all these prerequisites **and full natural main completion** pass does
profiling start from frozen iteration9: rollouts10–11, 8 optimizer updates. Each
main's native final checkpoint49 remains a separate completion requirement.

## Time, size and interpretation limits

Recorded leases end Rubin06:09:18UTC and GB07:09:02UTC on2026-09-16. The budget is
recomputed before submission: `min(4500s, lease remaining − requested retention reserve)`.
The default reserve remains1200s; an explicit `--retention-margin-seconds600` is
allowed for this new profile workflow. Values below600s are refused. Use the
chosen option explicitly on prepare, submit, retain and stop.
Fewer than15 minutes available causes refusal. Loading counts against the budget;
a slow cold load may prevent useful profiling. There is no lease extension.
The existing detached guard limits traces to10GiB and stops only the exact profile
submission. Its checks are polled: bytes can overshoot and Ray API failures can
delay stopping. A reserve is not a guaranteed transfer duration.

Retention has one overall deadline: the earlier of retain-start plus the requested
reserve and the recorded lease minus120s. Driver-log capture, both complete source
hash passes, rsync, destination hashing and verified-record publication share this
same budget; no stage renews it. Subprocess timeouts, local/remote deadline checks
and a POSIX alarm bound work. Source manifests retain their absolute deadline even
if SSH disconnects. Partial copies and audit files remain after failure; retention
never stops a container automatically. `retention.json` is accepted only together
with a matching completed `retain-attempt.json`. The audit records actual bytes,
absolute deadline, stage durations and total elapsed time; failed expired actions
emit their final status to stdout without starting a new filesystem write.

The separate explicit stop action uses one deadline ending no later than the
recorded lease. It checks terminal identity, completed retention and a fresh source
hash before either scoped stop command. This lets shutdown use the120s left by
retention without extending the allocation. A failed verification or exhausted
budget does not trigger an automatic container stop.
Main watchdog cutoffs remain Rubin 05:20/05:50 UTC and GB300 06:20/06:50 UTC;
this operator never changes them.

Main-run token caps differ: Rubin A2 uses4096, while the uninterrupted GB300 A1
uses8192. Their main performance curves must disclose that difference. Both
profile replays explicitly use4096. The helper records old/new token budgets in
`profile_token_budget_override`, including a log-prob token cap if originally
explicit; it never increases a budget or changes sampling, global batch, model,
parallelism, image, or packages. Review this field in both saved replay plans.

Replay keeps optimizer/RNG resume from checkpoint9, runs rollouts10–11 /8updates, disables save/eval and
profiles `train_overall` start1/end2. Counter selection is relative to the new actor;
it covers first-rollout tail through the second training stage on all4ranks.
Miles calls `prof.step()` once per actor training call, after its four optimizer
updates and before CPU weight backup. The schedule is warmup at counter0,
record-and-save at1, and none at2; the second training call synchronously exports
the gzip traces. A third rollout adds no export boundary. The second rollout's
post-train CPU backup, offload and final weight sync are outside this trace,
although the replay still completes them before normal disposal.
The installed Rubin PyTorch2.15 and GB300 PyTorch2.13 source paths, SHA256 values
and numbered excerpts were retained locally in
`outputs/rubin-gb300-qwen3/profile-two-rollout-source-evidence.json` (with a short
Markdown explanation alongside it); these are source evidence, not a completed capture.

The installed Miles profiler enables shapes, stacks, memory and FLOP recording
on every trainer rank. These add CPU/memory/export overhead, so replay durations
are not unprofiled performance evidence. The gzip exporter first writes an
uncompressed temporary JSON under `/cache/tmp`; the10GiB trace-directory guard
does not bound this temporary file or in-memory profiler buffers. Retention
hashes all output files before and after rsync and hashes the durable copies;
the reserve covers this work as well as transfer and shutdown.
Cross-version optimizer-checkpoint loading is not yet validated. If it fails,
preserve the failure rather than changing the images or declaring comparable traces.

SGLang is a separate capture. Use the bounded observer described in
[SGLANG_CAPTURE.md](SGLANG_CAPTURE.md), with one explicitly identified TP1 replay
engine and the submitted replay plan. Its payload uses `num_steps=4`, CPU/GPU,
`profile_by_stage=false`, with stacks, shapes and merging disabled. These are
four consecutive scheduler forwards; they do not guarantee both prefill and
decode coverage. Never target the primary learning job. The operator deliberately
does not guess an engine URL or issue generation requests.
Actual captured workload/backend and resumed checkpoint must be confirmed from logs.
