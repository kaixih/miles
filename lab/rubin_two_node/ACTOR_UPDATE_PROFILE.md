# Bounded actor-update replay

This is a separate diagnostic. It does not modify the frozen main source or repeat
50 training rounds. Run the same original initial HF/release policy, immutable
images, 4 GPUs / TP1 / PP1 / CP1 / EP4, and the same **real** 2048-sample rollout
file on both platforms. The original completed runs did not save that file: a
newly generated, SHA-bound initial-policy batch must be disclosed as such. It is
not the exact historical actor batch that produced the 1.77× main timing ratio.

`plan_actor_update_replay.py` only prints JSON. Its input can be the original
launcher JSON, Ray job JSON, or operator-plan JSON containing `ray_job`. It verifies
the rollout file SHA, initial fresh-run recipe and unchanged scheduler horizon of
50 rollouts. It removes saving/evaluation/native profiling and exits after one
replayed rollout (four optimizer updates). No SGLang engines are instantiated in
this replay. Native reference/current logprobs and advantages are recomputed.

Mount the two companion Python files read-only at `/opt/actor-profile`, with this
directory on PYTHONPATH. Mount the original `/opt/miles` source, policy models and
trusted rollout dump read-only. A `.pt` debug dump is a trusted pickle input; do
not substitute an untrusted external file. Verify 2048 real samples with tokens,
response lengths, rewards, loss masks and rollout logprobs using the installed
Miles `Sample` loader before the run. The initial model manifest ID is supplied
by the outer operator and must be verified there against mounted model files;
the planner does not silently turn a supplied ID into a tensor checksum claim.

Example (inside the prepared namespace; stdout is the complete plan):

```sh
python3 /opt/actor-profile/plan_actor_update_replay.py \
  --source-plan /inputs/source-plan.json \
  --run-id actor-profile-gb300-JOB-v1 \
  --rollout-path /inputs/rollout-0.pt --rollout-sha256 SHA256 \
  --initial-model-id VERIFIED_INITIAL_MANIFEST_ID \
  --train-step-source-sha256 a80fad89392d2e6abefe560d9082949f7eabe41eb4757d11bb31e7e8caa395ef \
  --max-tokens-per-gpu 4096 --prompt-data-path /inputs/train.jsonl \
  --output-root /run-output/actor-profile \
  --deadline-epoch ORIGINAL_ABSOLUTE_PHASE_DEADLINE
```

The deadline must be within one hour and within the original allocation. An
independent exact-container/job guard must already be armed; the hook's local
budget is additional protection, not a substitute. Submit only the reviewed plan's
exact `submission_id`, `entrypoint` and `runtime_env` through the phase operator.
The init hook also validates UID28644:GID30, 4-rank world, run identity, rollout
SHA, policy paths and the exact native `model.py` SHA before wrapping its function.

All ranks execute normal updates. Each rank records the four update timings and
selected sample token hashes, lengths, response lengths and microbatch indices.
Step 0 is warmup; **only rank 0 step 1** starts `torch.profiler`. It captures one
whole update with CPU/CUDA activities and `with_stack=False`, `record_shapes=False`,
`profile_memory=False`, `with_flops=False`. This excludes the separate logprob pass,
rollout wait, actor offload and weight synchronization. The selected native
`train_one_step` also includes its normal loss/checksum bookkeeping.

Ranges identify schedule forward/backward, optimizer, observed MoE classes,
checkpointed forward and checkpoint recompute-plus-backward when those installed
symbols exist. Collective wrappers are explicitly labelled **enqueue**, not
communication walltime. Native autograd/c10d/CUDA kernels remain the source for GPU
communication and overlap analysis; imported aliases may bypass Python enqueue
wrappers. Missing optional annotation symbols are visible in the recorded list,
not invented as measured phases. `actor.checkpoint_recompute_and_backward` is not
pure recompute time. Ranges nest, so their durations must not be added naively.

Timings use a host monotonic clock bracketed by whole-update CUDA synchronization.
An explicit diagnostic barrier **before** each timed window aligns all ranks and
keeps rank-0 metadata/export pauses out of peers' step-2 times. There are no extra
synchronizations inside individual forward/backward/MoE/communication ranges.
Profiled step 1 is excluded from throughput comparison; steps 2/3 supply unprofiled
matched diagnostic timings. They are not interchangeable with the main run's
`perf/actor_train` timer, which covers four updates and has different boundaries.
Compare rankwise packing fingerprints between platforms before claiming paired
work; report any mismatch. Rank-0 traces alone do not describe every EP rank.

The hook allows at most 180 seconds for capture, 120 seconds for export (both
capped by the original absolute deadline), 8 GiB RSS growth, and 512 MiB rank-0
output. Raw trace validation reserves half the output budget for gzip. Limits are
polled and can overshoot; a violation records the reason and exits only the
current diagnostic actor. The independent operator remains responsible for the
rest of that exact diagnostic job/container. Completed receipts require parsed
JSON, GPU kernels, the selected-update range, no Python function stack events,
and exact raw/compressed hashes. Partial files and failure receipts must be
retained honestly. No new run or deadline extension occurs automatically.

CPU validation:

```sh
python3 -m unittest lab.rubin_two_node.test_actor_update_profile -v
```

Before native GPU execution, run the planner and native argument parser in the
exact image and verify its actual hook/schedule symbols. The local tests simulate
profiler orchestration and metadata; they do not claim native GPU validation.

Validate the real rollout dump in the exact image with CPU-only visibility:

```sh
CUDA_VISIBLE_DEVICES='' python3 /opt/actor-profile/validate_actor_batch.py \
  /inputs/rollout-0.pt --expected-sha256 SHA256
```

The validator uses installed `Sample.from_dict` and `Sample.validate`, checks
256 contiguous prompt groups of 8, and re-evaluates the saved binary reward using
the frozen GSM8K scorer. Missing masks are valid: native conversion expands them
to all ones. Missing/misaligned/nonfinite logprobs, removed/failed samples, invalid
token bounds, inconsistent group prompts or changed input hashes fail. Output
contains summaries and hashes, not raw prompts/responses. Native annotation
availability is a separate check after a container exists; local tests alone
never establish native GPU compatibility. Identical duplicate source flags are
canonicalized; conflicting duplicates are rejected. Prompt-data is still needed
because native debug mode constructs the dataset before reading the frozen dump.
