# Two-node Rubin validation

## Nonzero training validation: PASS (2026-09-15)

Job `2195289` ran Qwen3.5-35B-A3B on c15/c16, four SM10.7 GPUs per node, using
the preserved FA2 image and clean runtime source
`e05d2549cc1027a9c229b40253e1a7b059e38968`. Training was BF16 TP2/PP1/CP1/EP8;
rollout used two TP4 engines. The recipe disabled thinking, used Miles' original
`math` scorer and original DAPO labels, and sampled with temperature 0.7,
top-p 0.8, top-k 20 and a 4096-token response limit. Each accepted batch contained
four prompts with four samples each; every group had nonzero reward variance.

| Rollout | Samples | Raw reward mean | Gradient norm | Changed served tensor hashes | Truncated |
| --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.75 | 0.5861834288 | 1904 | 8/16 |
| 1 | 16 | 0.50 | 0.6526616216 | 1860 | 12/16 |

The reward means describe filtered training batches, not dataset accuracy.
Centered GRPO losses were 3.7252903e-08 and 1.4901161e-08; the nonzero gradient
norms and changed parameter hashes establish real updates despite near-zero
mean losses.

The terminal collector passed all 33 checks and the learning audit passed all
90 checks. All 32 accepted samples reproduced their rewards with the unchanged
`grade_answer_verl` on CPU. The recorder retained all 39 groups reaching the
filter (8 accepted, 31 dropped), with no omitted suffix. The DAPO file contained
17398 rows and retained SHA256
`cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6`.

All eight ranks completed both normal optimizer steps and exactly three weight
synchronizations. Initial, post-step-0 and post-step-1 engine maps each contained
4028 tensor entries and matched across both engines. Rollout 1's effective weight
version was 2. Trainer parameter hashes also changed on every rank between the
two steps. The actual training log proved bidirectional cross-node MNNVL with
1088 deduplicated channel records and zero Socket data channels; backend discovery
messages were excluded from the data-path count. There were no fatal markers.

Ray job `raysubmit_pqZPyhdRM3d2RFiY` finished `SUCCEEDED` at
`2026-09-15 19:18:39.930 UTC`; Ray driver and both independent launcher exit codes
were zero. The source commit and all three audited runtime files matched Git,
launch records and both mounted container copies. The final training-log SHA256 is
`48659035cd2a00081ded85da02873aa121126cf69eea22f1a27c61a87879872f`.

The run was armed at `18:16:40.483726 UTC`, so Ray completion took 3719.45 seconds
including loading. A changed c16 NFS mount caused cold reads measured as low as
11.69 MiB/s (54.1 ms READ RTT), later rising to about 67 MiB/s. One explicit
watchdog extension changed the stopping deadline from `19:31:40.483726` to
`21:00 UTC`, preserving the original armed time and the same job. The archived
original state and handover are retained; elapsed time was not reset.

Earlier learning attempts are retained separately:

- **a1 — STOPPED:** a 4096-token thinking/`deepscaler` run accepted no mixed-reward
  group before its 4500-second budget. The last metrics snapshot accounted for
  at least 112 completed training requests, all 4096 tokens; this is a lower
  bound rather than a final request total. No optimizer step was established.
- **a2 — FAILED after its first optimizer step:** non-thinking/`math` scoring gave
  a real mean reward of 0.75 and valid steps on all ranks. The following checksum
  hook rejected native-FP32 `linear_attn.A_log`, which has no separate master copy.
  It failed before gradient metrics or post-step synchronization were recorded.
  Commit `e05d2549` fixes the diagnostic's mapping to the owned FP32 optimizer
  shard, preserves BF16 validation, and passes all 26 checksum tests.
  No training math, attention kernels or native package versions changed.

The compact result is [learning-validation-summary.json](learning-validation-summary.json).
Detailed evidence is retained under
`/home/scratch.kaixih_ent/repro/miles-rubin-two-node/20260915-j2195289/attempt3`:
`learning-aggregate.json`, `learning-validation.json`, `ray-job.json`, the two
exit records, events, filter audit and authoritative `.pt` sample dumps. Plain
single-turn samples did not produce separate trajectory JSONL files. The summary
records hashes and proof references without publishing responses or tensor maps.
This establishes two real updates for this recipe, not convergence, improved
model quality, or unrestricted long-context/backend correctness.

## Earlier 256-token execution smoke

**Earlier execution-smoke PASS:** attempt 3 completed the two-iteration Qwen3.5-35B-A3B Miles run,
confirmed at 2026-09-15 01:12:12 UTC. The validated configuration is two nodes with four
SM10.7 GPUs each, BF16 training TP2/PP1/CP1/EP8, two TP4 SGLang rollout engines,
real HF/checkpoint loading, and colocate/offload with cross-node MNNVL.
The compact result is in [validation-summary.json](validation-summary.json).

## Final execution evidence

| Rollout | Generated samples | Generation time | Served weight version | Valid optimizer-step ranks |
| --- | --- | --- | --- | --- |
| 0 | 16 | 23.8680779934 s | 1 | 0–7 |
| 1 | 16 | 10.7652921677 s | 2 | 0–7 |

Each rank logged `outcome=NORMAL valid_step=true` for both optimizer steps.
Rank-0 training metrics were finite. Each of the eight ranks completed three
`update_weights` calls with `ok=true`: initial synchronization, after rollout 0,
and after rollout 1. The evidence collector reported `log_evidence_status=complete`,
`all_required_log_evidence_present=true`, zero fatal markers, and zero rejected
events.

Across the final run and communicator reloads, the training log contained 6,144
`P2P/MNNVL` lines, including 640 explicit cross-node ring-edge lines for
3→4 and 7→0. There were zero Socket data edges and zero NCCL warning lines.
These are log-line counts across repeated communicator creation, not counts of
unique physical links.

Ray API job `raysubmit_8Q9uwHZXQjqUKhhL` reported `SUCCEEDED`. Both the outer
launcher exit file and `train_exit.json` recorded exit 0. The API's separate
`exit_code` field was unavailable (`null`), so the exit-code evidence comes from
those two files.

All responses reached 256 tokens, all math rewards were zero, and both training
gradient norms were zero. This proves the rollout/backward/optimizer/offload/
weight-synchronization workflow executed; it does not establish learning or
changes to parameter values.

Runtime source commit was `7ada69f96ee7678d81bc98881386117f793d050a`;
`recipe-commit-attempt3-fa2.json` confirms all five recorded runtime file hashes
matched that commit. The final training log SHA256 is
`23a321c10fb40de05a38578d8f45a479d7ce1fa940c952dd265e74900e8b3738`.
The run used the FA2 image digest below.

## Preserved containers

The ARM64 core, original FLA extension, and current FLA/FA2 extension were pushed
to `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin` and independently
checked through the GitLab API. The core was recovered by digest through Enroot
on both nodes; both Qwen3.5 extensions were pulled by digest through Docker on
both nodes. Full identities are recorded in `registry-images.json`.

The current image is tagged `qwen35-fa2-cu134-20260915`, with manifest digest
`sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`
and image ID
`sha256:3054a5be3f8233acac5e91c700803b3aeb1bd4ed6d1d792f2981091727b1efaf`.
It adds the scoped TE Python patch on the preserved FLA-only image
(`676354cc...`), which supplies `fla-core==0.5.2` and
`flash-linear-attention==0.5.2`. All 40 protected native package versions and
the Torch ABI remain unchanged; no native libraries were recompiled.

## MNNVL: PASS

Job `2179787` used `vr-nvl72-ts2-l11-038-c17` and `c18`, four SM10.7 GPUs each.
The unchanged `mnnvl-comm-smoke` launcher and Python test at commit
`36b106a4a2135e3267541610376af908fbb0b5cc` ran one Pyxis rank per GPU.
`NCCL_MNNVL_ENABLE` was unset and `NCCL_NVLS_ENABLE=0`.

- Ranks 0–3 were on c17 and 4–7 on c18.
- Cross-node edges 3→4 and 7→0 used `P2P/MNNVL`.
- All three message sizes produced exactly `8**25 = 3.777893186295716e+22`.
- There were no NCCL warnings/errors and no data transport edges via `NET/Socket`.

| Message size | Average all-reduce time |
| --- | --- |
| 1 MiB | 0.063 ms |
| 64 MiB | 0.578 ms |
| 512 MiB | 4.318 ms |

The fabric reports Completed/Success, one ClusterUUID and CliqueId32766.
Its health also reports Limited Capacity / Bandwidth Degraded, so these timings
are not a performance baseline.

Runtime checks verified Python `/opt/sglang/bin/python3`, Torch
`2.15.0.dev20260818+cu134`, CUDA13.4 and NCCL2.30.7. IMEX channel2020 was accessible.
Both Pyxis and Docker writer probes used UID28644:GID30; the normal user could
read and delete container-created shared output. All probe artifacts were removed.

## Earlier attempts and fixes

The recipe uses 32 Ray CPUs and four GPUs per node, with 600 worker ports.
An earlier bootstrap exhausted its original 200-port range when Ray detected
352 CPUs per node; the corrected bootstrap passed.

Attempt 1 was intentionally stopped during initial weight synchronization.
Miles' default `NCCL_CUMEM_ENABLE=0` selected cross-node `NET/Socket`, taking about
4.1 seconds per update bucket with a 512 MiB limit. Its Ray status is `STOPPED`,
neither a workload failure nor success. Ray returned a zero launcher exit code,
which alone is insufficient evidence of completion.

Attempt 2 enabled `NCCL_CUMEM_ENABLE=1` in both containers and the Ray runtime,
retaining `NCCL_NVLS_ENABLE=0`, automatic MNNVL selection, and the same
colocate/offload configuration:

- The checkpoint loaded and resharded from TP1/PP8 to TP2/PP1; both SGLang engines
  loaded real HF weights and became healthy.
- Initial training communicator logs contained 256 `P2P/MNNVL` edges, including
  cross-node 3→4 and 7→0, with no Socket edges or NCCL warnings.
  This is an initialization count, not a total across later communicator reloads.
- Initial weight synchronization completed all 132 buckets in about 7.7 seconds.
- Rollout 0 generated all 16 samples in 24.7376 seconds using weight version 1.
  Responses reached the 256-token limit and math rewards were zero.
- The first training backward failed on all eight ranks in TE/cuDNN
  `fused_attn_bwd`, at `fused_attn_f16_arbitrary_seqlen.cu:1025`, with
  `No valid execution plans built`. Ray job `raysubmit_T2QV5K33Jjt8YNZb`
  reported `FAILED`; the launcher exited 1. No completed optimizer step or
  post-training weight synchronization was established.

## Scoped FA2 workaround: kernel checks PASS

Training full attention uses causal THD BF16, head dimension 256, zero dropout,
and GQA 8:1 per TP2 rank. Direct source-built FA2 2.8.3.post1 forward/backward
passed on SM10.7. Forcing FA2 through unmodified TE 2.19 instead returned
`NoBackend`: its head-dimension-above-192 allowlist omitted `(10, 7)`.

`patch_te_sm107_fa2.py` permits only SM10.7 BF16 with Q/K/V head dimension 256
and zero dropout, preserving all other backend checks. Combined with
`--attention-backend flash`, it selects FA2 through TE and disables the failing
cuDNN path. The helper checks the TE version and exact source anchor before
patching; hashes are recorded in `registry-images.json`.

Both direct FA2 and patched TE passed output and Q/K/V gradient comparisons
against Torch FP32 math for lengths `[512]`, `[256, 512]`, and `[257, 385, 129]`.
All relative L2 errors were below 0.31%. The same probes passed in the newly
built independent image, which was then pushed, API-verified, and pulled on
both nodes.

These are packs without inter-sequence gaps. Actual Miles CP1 packing leaves
`cu_seqlens_q_padded/kv_padded=None`; any trailing alignment padding becomes an
additional dummy sequence. TE therefore infers `pad_between_seqs=False`.
An explicit gap-padding probe returned `NoBackend`, an unsupported boundary
retained by the patch. No extra packing change is required for this recipe.

Attempt 3 retained real HF loading, the same model/layout/offload and cuMem/MNNVL
settings, changing the image and training attention backend. Its full-run PASS
is documented above; the gap-padding boundary remains outside this recipe.

## Durable evidence

Detailed logs for the earlier 256-token execution smoke are under:

`/home/scratch.kaixih_ent/repro/miles-rubin-two-node/20260914-j2179787`

Key evidence includes:

- `evidence-attempt3-fa2.json`: per-rollout/per-rank completion checks and proof excerpts.
- `ray-job-attempt3-fa2.json`, `train-launcher-attempt3-fa2.exit`, and
  `train_exit.json`: Ray `SUCCEEDED` and independent zero launcher exits.
- `logs/qwen35_train.log`: final successful attempt, with SHA256 recorded above.
- `recipe-commit-attempt3-fa2.json` and `recipe-source-attempt3-fa2.json`:
  exact runtime source identity and file hashes.
- `attention-probe-final-image.log`: numerical probes in the independently built image.
- `mnnvl-default.log` and `mnnvl-default-summary.json`: eight-GPU communication gate.
- `logs/qwen35_train-attempt1-socket.log` and `ray-job-attempt1-socket.json`:
  intentionally stopped first attempt.
- `logs/qwen35_train-attempt2-cudnn.log`, `ray-job-attempt2-cudnn.json`, and
  `train_exit-attempt2-cudnn.json`: successful rollout followed by failed backward.
- `recipe-source-attempt2-cumem.json`: source hashes for attempt 2.
- `attention_probe.py`, `attention-probe-direct-fa2.log`,
  `attention-probe-te-flash-before-patch.log`, and
  `attention-probe-te-flash-patched.log`: numerical probes and backend selection.
- `bootstrap.json`: final two-node container and Ray setup.

Original image build and push logs are in the sibling experiment
`/home/scratch.kaixih_ent/repro/miles-rubin-container/20260914-j2170666`.
