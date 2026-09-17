# Retain final model checkpoint49 before profiling

Run from the later committed repository checkout on **dl3**, after the main
experiment exits successfully and before beginning any profiling:

```bash
python3 lab/rubin_trtllm_full/retain_final_checkpoint.py \
  --config /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/RUN_ID/driver-config.json \
  --execute
```

Without `--execute`, this prints the plan and makes no remote calls. The helper
imports the canonical `checkpoint_metadata.py` and `retain_profile_checkpoint.py`
inventory helpers from the sibling `lab/rubin_two_node` directory. Its own and
those dependency hashes are recorded. The frozen main source stays unchanged.

Without `--job-id`, the accepted jobs remain Rubin **2212643** and GB300
**2212644**. For an explicitly selected replacement, supply its positive decimal
job ID; it must match the driver configuration, exact run ID, durable directory,
and node-local job path. Both main exit receipts must be zero, the saved main plan must record
50 rollouts / 200 updates and model-only checkpoint saving, and the actual job
must still run on its original allocated node under UID 28644:GID 30 with the
unchanged lease end.

The September 17 replacement Rubin run uses job **2213753** and the original
frozen runtime/image, with a separate run history. Its retention invocation is:

```bash
python3 lab/rubin_trtllm_full/retain_final_checkpoint.py \
  --config /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-rubin-j2213753-trtllm/driver-config.json \
  --job-id 2213753 \
  --execute
```

This is a fresh run's final checkpoint49; no checkpoint is inferred from the
failed first Rubin run. The existing GB300 controller continues using its
original invocation without `--job-id`. Replacement preparation accepts Slurm's
observed one-second start/end rounding (04:23:10–12:23:11 UTC), while retention
still requires the actual lease end to match **12:23:11 UTC** exactly and keeps
the same 120-second reserve.

The fresh **r2** attempt within job **2213753** uses an explicit tag:

```bash
python3 lab/rubin_trtllm_full/retain_final_checkpoint.py \
  --config /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-rubin-j2213753-trtllm-r2/driver-config.json \
  --job-id 2213753 --attempt-tag r2 \
  --execute
```

Its node-local parent is exactly `miles-kaixih-j2213753-trtllm-r2`, and its
durable run directory ends in `20260917-rubin-j2213753-trtllm-r2`. Both must
match the explicit tag; an untagged or different attempt's checkpoint cannot
be selected. This is a fresh run, not checkpoint resume or reuse of either
failed attempt. The original lease end and all retention guards remain
unchanged. The driver config intentionally omits an `attempt_tag` field for
compatibility with the frozen main runner; the CLI and exact run/node paths
carry the tag.

The data route is **node-local checkpoint49 → SSH rsync launched on dl3 → direct
NFS**. It does not write through GB300's CIFS mount. Only manifest-listed final
checkpoint files, `latest_checkpointed_iteration.txt`, and
`rollout/global_dataset_state_dict_49.pt` are copied. Source tensor files are
verified through rsync's transfer checksum, sizes, and unchanged source
inode/size/mtime before and after transfer. Small metadata files additionally
require identical SHA256. No extra full tensor SHA256 read is claimed.

The deadline is the earlier of start plus **30 minutes** and lease end minus
**120 seconds**, with 90 seconds reserved for verification after copying. Both
the new local copy process group and the remote rsync have deadlines. At least
the actual checkpoint size plus 10% and 10 GiB of NFS headroom is required.
The expected model checkpoint is about 61 GB; the operator rejects an inventory
over 128 GiB, so an accidentally enabled optimizer checkpoint does not silently
become a several-hundred-GB copy.

The worker reserves `<run_dir>/final-checkpoint` with exclusive `mkdir` and
creates an invocation-specific `_INCOMPLETE.json` marker before copying. The
actual NFS mount does not support `renameat2(RENAME_NOREPLACE)`, so completion
uses the verified PASS receipt instead of an atomic directory rename. After
metadata, inventory, and source stability pass, the worker removes only its own
unchanged marker and writes the PASS receipt. **Directory existence alone never
means complete.** Require the PASS receipt and an absent incomplete marker.

No existing destination is overwritten or reused, even if it is empty. Existing
`final-checkpoint`, legacy `.partial`, or retention-state artifacts prevent an
automatic retry; inspect them first. Failures preserve all checkpoint bytes and
partial data. A crash after marker removal but before receipt creation still
has no PASS receipt and must not be reported as successful retention.

Success receipt: **`<run_dir>/final-checkpoint-retention.json`**, including:

- `status: "PASS"`, `iteration: 49`, `run_id`, and `destination_root`.
- Equal `source_checkpoint_id` / `destination_checkpoint_id`.
- `source_stable_before_after: true`, `rsync_exit_code: 0`.
- `verification: "rsync_transfer_plus_sizes_and_metadata_sha256"`.
- File inventory, timing, helper/dependency hashes, and main completion evidence.

The controller should require this receipt and no incomplete marker before profiling. Detailed state,
source snapshots, rsync output, and the file list are siblings of the final
checkpoint. This worker neither stops a training container nor releases a node.
