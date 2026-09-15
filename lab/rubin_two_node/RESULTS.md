# Two-node Rubin validation

Registry preservation, MNNVL communication, and the scoped FA2 kernel checks
passed. Attempt 2 failed during the first training backward after a successful
rollout. Attempt 3 is running with the FA2 fix; the two-iteration
Qwen3.5 run has not yet passed.

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

Job `2179787` holds `vr-nvl72-ts2-l11-038-c17` and `c18`, four SM10.7 GPUs each.
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

## Qwen3.5 execution: not yet complete

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
- Actual training communicators reported 256 `P2P/MNNVL` edges, including
  cross-node 3→4 and 7→0, with no Socket edges or NCCL warnings.
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

Attempt 3 retains real HF loading, the same model/layout/offload, and cuMem/MNNVL
settings, changing the image and training attention backend. Success still
requires both rollout IDs, completed backward/optimizer steps with finite
metrics, initial plus two post-training weight synchronizations, and Ray
`SUCCEEDED`. Zero reward or zero gradient must not be reported as evidence that
parameter values changed or learning improved.

## Durable evidence

All detailed logs are under:

`/home/scratch.kaixih_ent/repro/miles-rubin-two-node/20260914-j2179787`

Key evidence includes:

- `mnnvl-default.log` and `mnnvl-default-summary.json`: eight-GPU communication gate.
- `logs/qwen35_train-attempt1-socket.log` and `ray-job-attempt1-socket.json`:
  intentionally stopped first attempt.
- `logs/qwen35_train-attempt2-cudnn.log`, `ray-job-attempt2-cudnn.json`, and
  `train_exit-attempt2-cudnn.json`: successful rollout followed by failed backward.
- `recipe-source-attempt2-cumem.json`: source hashes for attempt 2.
- `attention_probe.py`, `attention-probe-direct-fa2.log`,
  `attention-probe-te-flash-before-patch.log`, and
  `attention-probe-te-flash-patched.log`: numerical probes and backend selection.
- `bootstrap.json`, `logs/qwen35_train.log`, and eventual `train_exit.json`:
  current attempt; inspect Ray job status as well as the launcher exit code.

Original image build and push logs are in the sibling experiment
`/home/scratch.kaixih_ent/repro/miles-rubin-container/20260914-j2170666`.
