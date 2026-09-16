# Initial-policy profile: source contract

Prepared locally; no GPU replay was launched by these checks. Checkpoint resume
remains the helper default. `--initial-policy` explicitly selects the already
staged HF model and converted `release` weights with fresh optimizer/RNG state.

Read-only source audit of the actual mounted Miles repository:

- `miles/utils/arguments.py:3136–3147`, SHA256
  `45ce4992566d641f3ab2847b59b475d3ce24905bd3bbebe1306c5c77f10ba30f`:
  raw-mode runs without `--load` set `load=ref_load`, `no_load_optim=True`,
  `no_load_rng=True`, `finetune=True`, and `start_rollout_id=0`.
- `miles/backends/megatron_utils/actor.py:197–199,233–234`, SHA256
  `2380636fda072d7125d09e7924f266f2142b8996b9f4169b6c2f6eabf0a06fec`:
  the actor initializes model/optimizer, then separately loads the reference.
  No training-checkpoint optimizer or RNG is restored in initial-policy mode.
- `miles/utils/profile_utils.py`, SHA256
  `f620b223dfcb3e6893c33c8f2ec4491d9628388b01e07d21dcb18919ca14fae4`:
  start1/end2 means wait0, warmup1, active1, repeat1. `actor.py:644` calls
  `prof.step` once per completed training rollout and before the CPU backup.
  The second call triggers gzip export; a third rollout is unnecessary.

The mode preserves the original `--num-rollout 50` scheduler horizon and source
sampling/model/training arguments. It adds `--start-rollout-id 0` and
`--debug-exit-after-rollout 2`: rollouts0–1, four optimizer updates each. It removes
save/eval/W&B output settings, enables the existing native profiler, and permits
the already reviewed explicit4096-token profile budget. It adds neither `--load`
nor `--use-checkpoint-opt-param-scheduler`.

The operator binds the existing `input-staging.json` read-only. The helper checks
actual read-only HF/reference mounts, UID28644, complete relative file inventory,
shard sizes and metadata hashes against that manifest, and requires the reference
tracker to select `release`. It never reads tensor shard contents for hashing.
The common model ID is SHA256 of `json.dumps({model_name: fingerprint},
sort_keys=True).encode()` for the exact HF/reference pair. Platform-specific host
paths are excluded; each full manifest SHA remains separately recorded.

Plans record both original argv (W&B values omitted) and profile argv, dataset
SHA256, shuffle setting, explicit seeds or their unchanged source declarations,
and initialization mode. Model/data evidence is checked again before submission.
No claim of identical generated responses or restored training RNG is made.

Interpret these as initial-workload backend/system traces. The capture includes
the first rollout tail and subsequent offload/weight sync/generation wait through
the second train end. Record actual lengths, token counts and microbatches when
comparing; the results do not substitute for late-policy timings or isolate GPU
hardware from the different runtime images. Exclude concurrent auxiliary-copy
windows from performance comparisons.
