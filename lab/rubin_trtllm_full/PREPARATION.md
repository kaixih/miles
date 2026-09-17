# Prepare the two full TRTLLM runs

`prepare_platform.py` defaults to Rubin allocation **2212643** and GB300
allocation **2212644**. An explicit `--job-id` selects a replacement allocation
and derives a separate run directory and container identity. Omitting this flag
preserves the original jobs. Run it on `dl3` as UID 28644:GID 30. Allocation and image
build/pull are done externally. The image must already be locally addressable by
its registry digest. The worker never allocates, builds an image, removes a
container, releases a node, or deletes model/checkpoint data.

Supply the immutable source directory, its commit, and a source manifest with
`git_commit` and `source_sha256`. The manifest must include the orchestrator,
Qwen3 launcher, GSM8K reward, watchdog, and updated main driver. Manifest hashes
are verified before preparation and must not change between attempts.

First inspect the plan by omitting `--execute`. A GB300 execution looks like:

```bash
python3 prepare_platform.py \
  --platform gb300 \
  --source /home/scratch.kaixih_ent/repro/NEW_FROZEN_SOURCE/source \
  --source-commit COMMIT_40_HEX \
  --source-manifest /home/scratch.kaixih_ent/repro/NEW_FROZEN_SOURCE/source-manifest.json \
  --image REGISTRY_IMAGE_AT_SHA256 \
  --wait-until 2026-09-17T08:00:00Z \
  --execute --execute-main
```

Use `--platform rubin` with the same frozen source/manifest; omit `--image` to
select the tested Rubin image with digest `405315ba…`. Rubin rejects a different
image. GB300 takes the externally built upstream derivative's immutable digest.
`--execute-main` is optional: without it the worker finishes in
`READY_NOT_TRAINING` after preflight.

The worker waits only for its explicitly selected job. Every remote operation rechecks the
normal-user owner, one actual allocated node, four GPUs, original start/end
times, and `TimeLimit=08:00:00`. Slurm start/end timestamps may differ by eight
hours plus one second due to rounding; the helper accepts that one-second
difference and always enforces the actual, unchanged `EndTime`. Pending allocations never permit node
access. Preparation refuses to begin with fewer than four hours remaining.

Storage is verified local ext4/XFS/btrfs with at least 400 GiB initially free;
400 GiB accommodates roughly 122 GB of input weights, a roughly 61 GB final
model checkpoint, and cache/temporary headroom. The final save omits optimizer
state. Data and source use the direct NFS login path and GB300's equivalent
`/mnt/cifs` host view. Both image-writer and normal-user read/delete probes run
before bulk output. On GB300 compute04, the exact prior local input directory
is reused read-only if its canonical **26 HF files + 20 converted files** and
metadata fingerprints match. Otherwise only those canonical files are copied.
Model payload identity is inherited from this retained inventory; shard bytes
are checked by size, and metadata/config files by SHA256.

The verified orchestration bootstrap creates one four-GPU container. Only
that container's `/root` traversal permission is adjusted for upstream GB300
Megatron imports. Preflight verifies launcher arguments, installed patch
provenance, dataset hashes, and a four-rank NCCL allreduce via
`python3 -m torch.distributed.run` with explicit `127.0.0.1` rendezvous. The
50-rollout / 200-update run uses TRTLLM BF16, decode CUDA Graph ON, prefill graph
OFF, and a final model-only checkpoint at interval 50.

Preparation receipts live under each exact campaign run directory:

```text
/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/
  20260917-gb300-j2212644-trtllm/
  20260917-rubin-j2212643-trtllm/
  20260917-rubin-j2213753-trtllm/  # replacement run, separate history
```

Each attempt has separate logs and failure receipts under `prep-attempts/`;
successful stage receipts live under `prep-stages/`. After inspecting a failure,
repeat the original command with `--resume`. This revalidates the same source,
image, allocation, local storage, model inventory, and completed container
bootstrap. It can resume partial input copying. An interrupted bootstrap that
left a container without a successful receipt refuses automatic adoption;
inspect that exact container and receipt before deciding recovery. Once any
main-launch evidence exists, preparation refuses to submit training again.

The main driver owns immutable lease-relative watchdog deadlines. An exit-zero
worker is not a claim of successful training: validate Ray SUCCEEDED, all
50 rollouts / 200 updates, numerical checks, and checkpoint separately. The
caller retains logs, profile results, and the final model checkpoint on durable
storage, verifies the copy, and only then releases the allocation. The worker
does not start a background checkpoint copy during training.

## Replacement Rubin run on September 17

The first Rubin run ended after 32 completed rollouts following a GPU hardware
fault. Its failed state and results remain separate. Replacement job **2213753**
uses node `vr-nvl72-ts2-l11-038-c15`, with recorded start **04:23:10 UTC** and end
**12:23:11 UTC**. It starts a fresh 50-rollout run from the original model;
there is no checkpoint resume or concatenation with the failed run.

The replacement controller uses a separate campaign directory but references
the original, unchanged `20260917-campaign/source-manifest.json` and its
`source-v2` directory. It requires frozen runtime commit
`1ce18e4e1930bfbdaaa4a776c3a82dcebed5fa0d` and the same Rubin image `405315ba…`.
The mutable host preparation helper is selected explicitly and its SHA256 is
recorded separately; the main driver and profiling executable remain frozen.

Plan-only invocation (add `--execute` only when ready to launch):

```bash
python3 -u lab/rubin_trtllm_full/campaign_worker.py \
  --platform rubin --job-id 2213753 \
  --campaign-root /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign-rubin-rerun-j2213753 \
  --source-manifest /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign/source-manifest.json \
  --prepare-helper /home/scratch.kaixih_ent/repo/miles-rubin-cu134/lab/rubin_trtllm_full/prepare_platform.py \
  --reuse-frozen-input-from /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign \
  --wait-until 2026-09-17T08:00:00Z
```

The controller consumes the new campaign's image READY receipt, prepares only
job 2213753, and copies both `frozen-token-input.json` and its receipt from the
original campaign **byte-for-byte**. It validates file/workload hashes and
records the reuse proof before training. It neither re-tokenizes the inputs nor
changes the original GB300 worker, its job 2212644, or its receipts.

For a separately prepared allocation, `--prepared-only` requires its existing
image READY receipt, verified `READY_NOT_TRAINING` state, and exact driver
configuration. The replacement still requires the explicit host helper for
allocation checks. Do not edit that helper after the controller records its
SHA256. A preexisting controller claim or main-launch evidence prevents an
automatic resubmission.

## Explicit fresh attempt within the same Rubin lease

`--attempt-tag r2` creates run `20260917-rubin-j2213753-trtllm-r2`, node-local
root `miles-kaixih-j2213753-trtllm-r2`, container prefix
`miles-rubin-qwen3-trtllm-j2213753-r2`, and profile
`rubin-j2213753-trtllm-r2-profile-v1`. The campaign must be separate and end in
`-r2`. This starts a new 50-rollout / 200-update run; earlier claims, logs,
failures and directories are never resumed, reset or overwritten.

The tagged attempt permits its own frozen source manifest and exact commit.
The supplied manifest SHA256 and every listed file are verified and pinned.
Launcher, reward and full main-driver hashes must match the original `1ce18e4e`
manifest, preserving the learning recipe. The September 17 `r2` runtime is
`45dfbda0693585b1c43f43eb4947cd65bad8897f`; its scoped runtime change adds bounded
GET retries during worker discovery. The Rubin image, canonical model/data
inventories and original allocation end remain unchanged.

Plan-only command for this explicit attempt:

```bash
python3 -u lab/rubin_trtllm_full/campaign_worker.py \
  --platform rubin --job-id 2213753 --attempt-tag r2 \
  --campaign-root /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign-rubin-j2213753-r2 \
  --source-manifest /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign-rubin-j2213753-r2/source-manifest.json \
  --source-manifest-sha256 759cf3583b2c8f4e6f87e07094eb8f81ce124ea052e938cd683e08f7d570d34e \
  --prepare-helper /home/scratch.kaixih_ent/repo/miles-rubin-cu134/lab/rubin_trtllm_full/prepare_platform.py \
  --reuse-node-models /tmp/miles-kaixih-j2213753-trtllm/models \
  --reuse-frozen-input-from /home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign \
  --wait-until 2026-09-17T08:00:00Z
```

Add `--execute` only for the reviewed launch. Model reuse rechecks the complete
canonical inventory and metadata fingerprints and mounts it read-only; failure
stops preparation instead of silently copying another 122 GB. The original
frozen profile input and receipt are copied byte-for-byte. Host helper SHAs are
recorded separately from the frozen runtime. Preparation still requires four
hours remaining; all watchdog and retention deadlines use the original
12:23:11 UTC lease end. Ports are unchanged, so the previous owned container
must be stopped by the operator before fresh bootstrap; these helpers never
stop or delete a previous attempt to make room.

The attempt tag is carried in names, controller claims and explicit helper
arguments. It is intentionally not an extra main-driver config field, because
the unchanged frozen main driver accepts only its documented schema. Final
collection/report bindings must explicitly select this new run and source
commit; do not splice older attempts into its curves.
