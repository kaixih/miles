# Prepare the two full TRTLLM runs

`prepare_platform.py` is scoped to Rubin allocation **2212643** and GB300
allocation **2212644**. Run it on `dl3` as UID 28644:GID 30. Allocation and image
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

The worker waits only for its fixed job. Every remote operation rechecks the
normal-user owner, one actual allocated node, four GPUs, original start/end
times, and exact eight-hour lease. Pending allocations never permit node
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
