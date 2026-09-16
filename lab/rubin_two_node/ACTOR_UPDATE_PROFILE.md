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
  --schedule-source-sha256 REVIEWED_PLATFORM_SCHEDULE_SHA256 \
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
Step 0 is warmup. In **rank 0 step 1**, only the second microbatch (index 1)
starts `torch.profiler`. A single `actor.selected_microbatch` range spans its
complete forward and paired backward, including recomputation and MoE work, with
CPU/CUDA activities and `with_stack=False`, `record_shapes=False`,
`profile_memory=False`, `with_flops=False`. The rest of the four real updates
continues normally. Optimizer execution, the schedule's final gradient-sync
function, reference/current logprobs, rollout wait, offload and weight-sync are
outside this trace. Any collective launched by the selected backward itself can
still be present. The receipt explicitly sets optimizer/final-gradient-sync to
null; it does not extrapolate a microbatch into an observed whole-update trace.

This scope replaces the failed full-update attempt: despite stacks being off,
its raw trace grew to about 1.98 GB and was interrupted by size guards. The
512 MiB output and 256 MiB raw limits remain unchanged; partial v1 output is not
valid profiling evidence.

Ranges identify schedule forward/backward, observed MoE classes,
checkpointed forward and checkpoint recompute-plus-backward when those installed
symbols exist. Collective wrappers are explicitly labelled **enqueue**, not
communication walltime. Native autograd/c10d/CUDA kernels remain the source for GPU
communication and overlap analysis; imported aliases may bypass Python enqueue
wrappers. Missing optional annotation symbols are visible in the recorded list,
not invented as measured phases. `actor.checkpoint_recompute_and_backward` is not
pure recompute time. Ranges nest, so their durations must not be added naively.

Timings use a host monotonic clock bracketed by whole-update CUDA synchronization.
An explicit diagnostic barrier **before** each timed window aligns all ranks and
keeps rank-0 metadata/export pauses out of peers' step-2 times. For the selected microbatch only, one CUDA synchronization drains previous GPU
work **before** profiler start, and another completes the selected backward
**before** profiler stop. There are no synchronizations inserted inside individual
forward/backward/MoE/communication annotations. The small trace is exported only
after the whole `train_one_step` (including optimizer) returns; no export is
performed while a native forward/backward call is on the stack.
Profiled step 1 is excluded from throughput comparison; steps 2/3 supply unprofiled
matched diagnostic timings. They are not interchangeable with the main run's
`perf/actor_train` timer, which covers four updates and has different boundaries.
Compare rankwise packing fingerprints between platforms before claiming paired
work; report any mismatch. Rank-0 traces alone do not describe every EP rank.

The hook allows at most 180 seconds for the selected update/capture, 120 seconds for export (both
capped by the original absolute deadline), 8 GiB RSS growth, and 512 MiB rank-0
output. Raw trace validation reserves half the output budget for gzip. Limits are
polled and can overshoot; a violation records the reason and exits only the
current diagnostic actor. The independent operator remains responsible for the
rest of that exact diagnostic job/container. Completed receipts require parsed
JSON, GPU kernels, the selected-microbatch range, no Python function stack events,
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

The required `--schedule-source-sha256` is platform-specific. Inspect and review
the exact installed PP1 forward/backward sequence before providing its hash; do
not blindly hash an unreviewed file to bypass the ABI check. The verified GB300
Megatron `schedules.py` SHA is
`968b3c2ab8e80b5d30992bb572bae8aa7cbc6b2fba7137d851bb71ee51bf489f`:
the non-overlapping PP1 branch calls F0/B0, F1/B1, ... in order. The hook checks
TP1/PP1/CP1/EP4, the exact `forward_backward_no_pipelining` callable and source
hash, refuses combined overlapping-MoE schedules, and verifies call counters,
iterator offsets and the exact forward output tensor passed to backward.
Observed counts must equal the planned microbatch count. Both schedule wrappers
and active profiler/range contexts are restored on error; failed partial captures
are not exported as COMPLETE.

New receipts retain target `[0,1,0]` for the full update identity, and add
`capture_window="single_forward_backward_microbatch"`, `microbatch_index=1`,
`selected_microbatch` (actual local indices, global sample indices where present,
tokens/lengths and response totals), and `schedule_observation` (observed full
update forward/backward counts, one captured pair, actual selected indices).
The trace filename remains `actor-update.json.gz` for operator compatibility;
its measured scope is unambiguously the receipt's single microbatch.
