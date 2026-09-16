# Optional checkpoint-9 GB copy fallback

**Prepared only. Not a launch instruction.** The existing single-stream worker
remains the default. Root may select this fallback only after two measurements
of the actual checkpoint GB copy predict a miss of the unchanged **2026-09-16
05:40 UTC** cutoff. The earlier model-copy rate is not a checkpoint measurement.
No healthy training job may be stopped to enable this fallback.

`copy_gb_checkpoint_parallel.py` runs on dl3 as UID 28644, GID 30. Default mode
prints a plan and performs no SSH, process inspection, or filesystem writes:

```sh
python3 lab/rubin_two_node/copy_gb_checkpoint_parallel.py
```

It uses only the verified durable checkpoint
`20260916-j2198331-a2-mb4096-nfs/profile-checkpoint-9`, identity
`b7cb2def84d3abc6614953c215c8b97ab6e26bac5b741a2d3d36fa17bd2b2e80`.
Destination is exactly
`gb300-nvl-012-compute04:/raid/tmp/miles-kaixih-j2198810-profile/profile-checkpoint-9.partial`.
The completed destination must not already exist.

## Explicit handoff, required before execution

A human/parent-reviewed JSON record, owned by UID 28644, must be supplied with
its **explicit SHA256**. It binds the precise original dl3 worker, local copy
children, and GB copy processes observed **before** they became terminal. This
helper cannot create that evidence, stop the worker, or authorize a handoff.
An original worker still alive, an unobserved copy identity, an unreadable live
ordinary-user process, or a fresh checkpoint writer causes refusal.

Required record keys:

```json
{
  "schema_version": 1,
  "authorization": "parallel_checkpoint9_copy_after_original_writers_terminal",
  "reviewed": true,
  "reviewer": "REPLACE_WITH_REVIEWER",
  "reviewed_at": "REPLACE_WITH_UTC_ISO_TIMESTAMP",
  "source_root": "/home/scratch.kaixih_ent/repro/miles-rubin-qwen3-gsm8k/20260916-j2198331-a2-mb4096-nfs/profile-checkpoint-9",
  "destination_root": "/raid/tmp/miles-kaixih-j2198810-profile/profile-checkpoint-9.partial",
  "checkpoint_id": "b7cb2def84d3abc6614953c215c8b97ab6e26bac5b741a2d3d36fa17bd2b2e80",
  "deadline_utc": "2026-09-16T05:40:00+00:00",
  "original_worker_host": "dlcluster-login-03",
  "gb_host": "gb300-nvl-012-compute04",
  "original_worker_terminal": true,
  "original_gb_copy_terminal": true,
  "original_state_sha256": "REPLACE_WITH_TERMINAL_STATE_FILE_SHA256",
  "original_worker": "REPLACE_WITH_OBSERVED_PROCESS_IDENTITY",
  "original_local_copy_processes": ["REPLACE_WITH_OBSERVED_COPY_IDENTITIES"],
  "original_gb_copy_processes": ["REPLACE_WITH_OBSERVED_COPY_IDENTITIES"]
}
```

Each process identity is an object with positive integer `pid`, `start_ticks`,
`uid: 28644`, its actual `argv` list, and `cmdline_sha256` from the raw
NUL-separated `/proc/PID/cmdline` bytes (including its trailing NUL). The worker
argv must identify `retain_profile_checkpoint.py --execute-retention`.
Local/GB lists must be nonempty and identify copy processes, including the
original rsync receiver. The reviewed original state file is
`checkpoint-retention-profile-9-state.json`; its PID must match and its phase
must be `staging_durable_checkpoint_to_gb_raid` or `failed`. The ten-minute review
freshness limit does not replace two fresh process/writable-FD scans on each
host. PID reuse is distinguished by start ticks. Scope is ordinary UID 28644;
root-owned processes are not claimed to have been inspected.

Only after this evidence has been reviewed is the following execution interface
available (placeholder values below intentionally do not authorize anything):

```sh
python3 lab/rubin_two_node/copy_gb_checkpoint_parallel.py \
  --execute-reviewed-handoff \
  --handoff-record /absolute/path/to/reviewed-handoff.json \
  --handoff-sha256 REVIEWED_64_HEX_SHA256
```

## Copy and publication contract

- Metadata is copied separately, then exactly four distinct immutable `.distcp`
  shards run with at most four parallel rsync processes: one file, one writer.
- `--partial` and normal rsync delta/transfer checksums preserve and reuse owned
  partial data. There is no `--delete`, `--inplace`, `--append`, `--checksum`, or
  extra complete tensor hashing. Existing completed files may be skipped by
  rsync's size/mtime quick check; their prior successful rsync is the transport
  evidence. No speedup or transfer completion is guaranteed.
- A durable verified record, metadata SHA256, shard sizes, ownership, absence of
  symlinks, four-shard count, and the shared embedded-common/legacy metadata
  contract must match before copying and again before publication. Unexpected
  partial files/directories are refused and left untouched.
- Exact target locks and a new local audit directory make execution one-shot.
  Existing locks/audits require inspection; this helper neither removes them nor
  retries. Full verification precedes rename and the standard
  `checkpoint9-gb-stage.json` completion record consumed as staging evidence.
- The **one absolute 05:40 UTC budget** covers review validation, both process
  scans, metadata/shard transfer, source recheck, destination metadata hashing,
  rename, and completion-record writes. Each SSH call has a bounded timeout;
  receiver rsync computes its remaining absolute time after SSH startup.
  GNU `timeout` applies only to newly launched rsync children, with a short
  cleanup reserve. No existing process is signaled by this script.
- Failure leaves partials, per-transfer logs/receipts, and lock/audit files.
  There is no retry, budget extension, cleanup, or automatic container/job stop.
  If time expires between rename and completion-record publication, the renamed
  destination remains for manual review and is not claimed complete by a record.
  After cutoff, failure reporting is stderr-only rather than new file writes.

Audit files live in the durable run's `checkpoint9-parallel-fallback/`; raw
original retention state and logs remain unchanged. Keep CLI stdout/stderr in
an external operator log as well. Dependencies are the existing Python stdlib,
`retain_profile_checkpoint.py`, `checkpoint_metadata.py`, rsync, GNU timeout,
and SSH; no package installation is part of this workflow.

CPU-only verification:

```sh
python3 -m unittest lab.rubin_two_node.test_copy_gb_checkpoint_parallel -v
```
