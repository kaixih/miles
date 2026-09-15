# Draft: Rubin d256 attention backward fails during cuDNN runtime compilation

## Summary

On a Rubin SM10.7 GPU with Transformer Engine 2.19.0 and cuDNN 9.25.0 Developer
Preview, BF16 causal attention with head dimension 256 succeeds in forward but
fails in backward with `No valid execution plans built`. A standalone PyTorch
+ TE example reproduces this with one head and only 16 tokens. Changing the head
dimension to 128 makes the same example pass.

The default cuDNN diagnostic log reports runtime kernel compilation failure.
A separate diagnostic using external NVRTC exposes seven compilation errors:
the generated SM107 d256 backward kernel references undefined identifiers
`oob_M_0`, `reg_0_0`, and `ptr_0`.

This is a report against the recorded developer-preview stack. We would like
help identifying the affected cuDNN generated-kernel path and the appropriate
compatible build or fix. We have not tested a newer driver/cuDNN combination.

## Exact environment

| Component | Recorded value |
| --- | --- |
| GPU | Rubin, compute capability 10.7, single device |
| Host architecture | Linux aarch64 |
| NVIDIA driver | 610.47.04 |
| CUDA compiler | 13.4.46 |
| PyTorch | `2.15.0.dev20260818+cu134` |
| Transformer Engine | `2.19.0+5e52bef` |
| TE source commit | `5e52befd5262c06289106338c308079d6adb391f` |
| TE build architectures | `NVTE_CUDA_ARCHS=107a` |
| cuDNN package | `nvidia-cudnn-cu13==9.25.0.15` |
| cuDNN frontend | `1.28.0` |
| External NVRTC package | `13.4.46rc1` |

Full image and source identities are in [environment.json](environment.json).
No package was installed, upgraded, or resolved during reproduction.
The minimal D128-pass/D256-fail pair was verified in both immutable images:
the original image without the FA2 eligibility patch and the later patched
image. The same pair also holds with external NVRTC and without a diagnostic shim.

The [cuDNN 9.25 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-25-0-developer-preview)
describe early Rubin/CUDA 13.4 support. The currently published
[9.25 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.25.0/reference/support-matrix.html)
lists R615 or later for Rubin, while this experiment uses R610 preview software.
That compatibility difference remains relevant. The observed compilation errors
do not by themselves prove that upgrading the driver resolves this failure.

## Reproduce

Run the attached [minimal_repro.py](minimal_repro.py) on one GPU in the recorded
environment. It imports only PyTorch and Transformer Engine and forces cuDNN
attention before the TE import.

```bash
timeout -k 10s 120s python3 -u minimal_repro.py 256  # forward PASS; backward FAIL
timeout -k 10s 120s python3 -u minimal_repro.py 128  # forward and backward PASS
```

Inputs are independent contiguous BF16 tensors with shape `[16, 1, D]` and stride
`[D, D, 1]`. Q/K/V have one head each, cumulative lengths `[0, 16]`, maximum
length 16, `padding_causal` mask, zero dropout, and no inter-sequence padding.
Backward receives a nonzero random upstream gradient. The script does not use
Miles, Ray, Megatron, weights, data, process groups or activation checkpointing.

## Actual failure and diagnostic evidence

TE selects `FusedAttention (sub-backend 1)`. Forward completes and synchronizes.
Backward raises:

```text
fused_attn_f16_arbitrary_seqlen.cu:1025
cuDNN Error: [cudnn_frontend] Error: No valid execution plans built.
```

At the pinned TE source line, the failing call is `mha_graph->build_plans(handle)`.
Graph validation, operation-graph construction, heuristic-A plan creation and
support checking have already returned successfully. The failure occurs before
backward execution.

With `CUDNN_LOGLEVEL_DBG=3`, the default compiler path reports:

```text
CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED (4001)
Encountered runtime kernel compilation failure at: compilationResult != NVRTC_SUCCESS
```

cuDNN normally uses a bundled NVRTC implementation. The optional diagnostic in
the README sets `CUDNN_NVRTC_RUNTIME_LINKING=1` to use the already installed
external compiler and records its failure without changing compile inputs.
This separate process returns `NVRTC_ERROR_COMPILATION` and seven errors in:

```text
cudnn_generated_fort_native_sdpa_sm107_flash_bprop_f16_knob_2_128x128x256_1x4x1_cga2x1x1_kernel2_0.cu
```

The errors identify undeclared `oob_M_0`, `reg_0_0`, and `ptr_0` around generated
lines 5762–5771. These exact identifier diagnostics were observed with external
NVRTC; the baseline bundled-compiler log exposes only the compilation-failed
status. Both paths reach the same outer TE backward error.

## Pass/fail controls

All rows below use the same recorded GPU/software stack and cuDNN backend unless
specified otherwise. Passing numerical controls compare output and Q/K/V
gradients with PyTorch FP32 math SDPA.

| Change | Forward | Backward / result |
| --- | --- | --- |
| Minimal: BF16, THD, S16, Hq=Hkv=1, D256 | PASS | Backward plan build fails |
| Only change D256 → D128 | PASS | PASS, numerical comparison passes |
| S512, D64 or D128, Hq=1 or 8, Hkv=1 | PASS | PASS, numerical comparison passes |
| Dqk192 / Dv128, S512, Hq8/Hkv1 | PASS | PASS, numerical comparison passes |
| S512, D256, Hq=1 or 8, Hkv=1 | PASS | Backward plan build fails |
| BSHD or SBHD, S512, D256, Hq8/Hkv1 | PASS | Backward plan build fails |
| BSHD or SBHD, same inputs with D128 | PASS | PASS, numerical comparison passes |
| FP16 instead of BF16, THD D256 | PASS | Backward plan build fails |
| D256, lengths 16/64/128/256/512 or pack [257,385,129] | PASS | Backward plan build fails |
| D256 with padding-only mask, without causality | PASS | Backward plan build fails |
| D256 with deterministic mode enabled | PASS | Backward plan build fails |
| D256 with TE core-attention checkpointing enabled | PASS | Backward plan build fails |
| D256 with explicit padded boundaries equal to cumulative lengths | PASS | Backward plan build fails |
| D256 with TE-wrapped FlashAttention 2 control | PASS | PASS, numerical comparison passes |

The complete results also retain deliberately invalid THD masks and unsupported
D192/D192 as distinct input/backend rejections. A length-one D128 test completed
TE forward/backward, then its FP32 reference encountered an unrelated Triton
assembler error. It is excluded from the numerical pass/fail pair. The minimal
example uses length 16 to avoid length-one special cases. The cuDNN 9.25 release
notes also list backward with K/V sequence length one as unsupported.

## Interpretation and open questions

**Established:** the failure is independent of the original model, GQA, THD
packing, recomputation, distributed execution and zero training gradients.
The cuDNN backend accepts the graph and fails while compiling a backward plan.
The external-compiler diagnostic reports undeclared names in generated CUDA
source, pointing toward a cuDNN code-generation issue on this path.

**Still unresolved:** whether the bundled compiler fails on exactly the same
generated-source defect; which cuDNN build fixes the problem; and whether this
preview stack has an additional driver/toolkit compatibility requirement.
No host driver change or newer native-library test has been performed.

The separate TE FlashAttention eligibility issue should be tracked independently:
the original TE Python allowlist omitted SM10.7 for FA2 d256. A scoped allowlist
patch permits the numerical FA2 control. It does not affect forced cuDNN backend
selection or native plan compilation, as confirmed by the unpatched-image control.

Questions for the TE/cuDNN team:

1. Is the generated SM107 d256 backward kernel defect known in cuDNN 9.25.0.15?
2. Which supported driver/CUDA/cuDNN build should we use for the next comparison?
3. Can TE expose the underlying plan-compilation error or exclude this specific
   known-bad combination when no valid backward plan can be built?
