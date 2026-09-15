# Source diagnosis: TE 2.19 / cuDNN 9.25 backward on Rubin

Snapshot: 2026-09-15. This note records source evidence and the first isolated
failure. Use the final result matrix and issue draft for the completed experiment.

## Confirmed failure stage

The original small FA2 numerical probe, forced onto TE/cuDNN in an independent
container, fails for the first case: BF16, one sequence of length 512, Q heads 8,
KV heads 1, Q/K/V dimension 256, causal THD, dropout 0. It reaches backward and
raises the same `fused_attn_f16_arbitrary_seqlen.cu:1025` error as training.
This reproduction needs no Miles, Ray, Megatron, weights, data, checkpoint
recomputation, process group, or cross-node operation.

The bounded level-3 cuDNN log contains a more precise cause:

```text
status: CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED (4001)
Reason: Encountered runtime kernel compilation failure at: compilationResult != NVRTC_SUCCESS
Reason: rtk(kernelNumRunning)->compile(compilerFlags, this->useNvrtcSassPath, true )
```

The remaining traceback passes through `compile_internal()`, `ptr.compile()`,
`engine_post_checks(...)`, `finalize_internal()`, and `ptrDesc->finalize()`.
There is no explicit driver error or PTX-version error in this log. The generic
TE exception hides a runtime kernel compilation failure; it does not establish
which compiler option, generated source construct, or lower component failed.

Evidence filenames (private storage locations are recorded separately):

- `original-cudnn.log`: isolated failure.
- `original-cudnn-debug.log`: 5,782,909 bytes / 99,716 lines.
- Debug lines 90120–90136: failed `cudnnBackendFinalize` and compiler traceback.
- Debug line 95884: outer TE exception.

## TE source facts

Commit: `5e52befd5262c06289106338c308079d6adb391f`.

- [`fused_attn.cpp:343–358`](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/common/fused_attn/fused_attn.cpp#L343)
  explicitly admits training with d256/d256 on SM100–109, cuDNN >=9.25 for THD
  (>=9.23 for dense), no bias/dropout, and vanilla softmax. GQA divisibility is
  valid for 8:1. Therefore the observed input passes an intentional eligibility
  branch; the source does not categorically exclude Rubin or d256.
- [`fused_attn_f16_arbitrary_seqlen.cu:1021–1025`](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu#L1021)
  validates the graph, builds its operation graph, creates heuristic-A execution
  plans, checks support, then builds plans. The exception is the final step.
  It occurs before backward workspace sizing completes and before graph execution.
- Forward can use direct cumulative lengths with recent cuDNN; backward uses
  converted sequence lengths and ragged offsets. Dense-vs-THD controls distinguish
  whether that difference matters.

Directly retrieved source hashes:

| File | SHA256 |
| --- | --- |
| `fused_attn.cpp` | `956ee977263721e13567c9f98e87a6051def044369dbaf68b38932aa0e17002e` |
| `fused_attn_f16_arbitrary_seqlen.cu` | `8d5f3cc3010b78c52a767f137d208730789eeb6dca30e8e4005bd115d5d80330` |

The latter has 1,428 lines. A browser cached response for the raw SHA URL returned
different content and line numbers; these references were checked by fetching
the exact URL directly, and line 1025 agrees with the installed binary's error.

## Installed cuDNN frontend facts

The installed `nvidia-cudnn-frontend` distribution is 1.28.0. TE's
`build_tools/utils.py` resolves headers from this distribution; TE's CMake checks
`CUDNN_FRONTEND_INCLUDE_DIR` and uses that directory. No bundled frontend Git
submodule is present in this TE commit.

Header paths below are relative to the installed frontend include directory:

- `cudnn_frontend/node/scaled_dot_product_flash_attention.h:1319–1327` applies
  the SM-major-10 branch, marks d256/d256 as `is_d256_on_blackwell`, and **forces
  `attributes.is_deterministic_algorithm = true`**. Thus a user-level
  nondeterministic setting does not avoid this algorithm for d256.
- Lines 1321–1324 accept the d192 special case only with dV128. A d192/d192
  negative test is an eligibility rejection, not the same compilation failure.
- Lines 1344–1346 require Q head count divisible by K and V head counts; 8:1
  satisfies that rule.
- `cudnn_frontend/plans.h:678–700` attempts available plan candidates and emits
  the generic “No valid execution plans built” error when none succeeds.

## Previous probes and training-specific factors

The earlier cuDNN smoke test in
`lab/rubin_container/docker/verify_rubin_attention.py` used BF16 **BSHD,
B2/S64/H4/D64, ordinary MHA**. It was not a d256 THD test. The existing d256
numerical probes explicitly exercised direct FA2 and TE-wrapped FA2; they did
not demonstrate that cuDNN d256 backward passed.

Source review of the original Miles checkout confirmed:

- `miles/backends/megatron_utils/parallel.py:92–100` passes identical Q/KV
  cumulative lengths and actual maximum length; padded-boundary fields are absent.
- Qwen3.5 full-attention layers retain Megatron's TE specification in
  `miles_plugins/models/qwen3_5.py:233–265`.
- Original resolved args recorded `deterministic_mode=False`, zero attention
  dropout, full/uniform recomputation, and THD layout.

The isolated failure makes recomputation, the model's Q/K normalization/RoPE,
Miles padding, model tensor views, distributed training, and zero rewards
unnecessary conditions for this failure. It does not prove those features are
free of unrelated issues.

## Ranked open hypotheses and distinguishing evidence

1. **d256's forced deterministic backward compiler path.** Compare d128/d128
   against d256/d256, plus valid d192/d128. Then compare MHA and GQA, dense and
   THD, causal and non-causal. Preserve the actual selected backend and distinguish
   unsupported input from a backend compilation failure.
2. **Generated source or compiler-option incompatibility on SM107.** The cuDNN
   trace proves NVRTC compilation failed but omits the NVRTC result/log. The
   optional `nvrtc_trace.c` diagnostic records failed compile result, options,
   NVRTC version and bounded program log, without dumping generated source.
3. **Developer-preview stack limitation.** Installed CUDA runtime and external
   NVRTC are `13.4.46rc1`; the host driver was reported as 610.47.04. The official
   [cuDNN 9.25 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.25.0/reference/support-matrix.html)
   requires CUDA >=13.4 and driver >=615.xx for SM10.7. The
   [cuDNN release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html)
   identify 9.25.0 as a developer preview and 9.25.1 as GA. The current
   [CUTLASS changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)
   likewise requires R615 for its Rubin kernels. This environment differs from
   that documented support baseline, but **does not prove this compilation
   failure's cause**. NVRTC compilation failure is not equivalent to driver
   rejection. No driver or package changes have been attempted.

## Bundled versus external NVRTC

The first preload diagnostic reproduced the failure but printed no tracer lines
(`nvrtc-trace-d256.log`). Inspection of `libcudnn_graph.so.9` found the version
marker `cudnn_nvrtc_version_13_4_38` and bundled NVRTC compiler implementation.
The exported `nvrtcHelper::load()` checks `CUDNN_NVRTC_RUNTIME_LINKING` against
the string `1`. With that value it opens `libnvrtc.so.13` and resolves NVRTC
entrypoints with `dlsym`; without it, it populates an internal function table.
This explains why ordinary public-symbol interception need not see baseline
compiles. The package's external NVRTC version alone is not evidence of which
compiler cuDNN actually uses.

An explicit `CUDNN_NVRTC_RUNTIME_LINKING=1` process is a useful diagnostic
control. It changes the compiler provider and must be recorded as a separate
configuration, including the loaded library path. It is not merely a logging
flag.

The first external-provider diagnostic **also failed**, with forward completed
and backward plan construction rejected. In `external-nvrtc-d256.log`:

- Line 28 records `NVRTC_ERROR_COMPILATION` (result 6), NVRTC API version 13.4.
- Lines 65–93 report seven compile errors in
  `cudnn_generated_fort_native_sdpa_sm107_flash_bprop_f16_knob_2_128x128x256_1x4x1_cga2x1x1_kernel2_0.cu`.
- The errors identify undefined `oob_M_0`, `reg_0_0`, and `ptr_0` at generated
  source lines 5762–5771.

These exact identifiers were captured only with explicitly enabled external
NVRTC and the diagnostic preload. They strengthen the generated-source/compiler
path hypothesis; they are not evidence that the bundled compiler produced the
same detailed diagnostics. The default-provider cuDNN log independently proves
runtime compilation failure but does not expose its compiler program log.

The first tracer version printed unreadable option bytes. The corrected tracer
snapshots options before compilation and replaces nonprintable bytes with `?`,
leaving the underlying arguments untouched. The option strings still contain
non-text data, so no compiler-option values are inferred from either record.
The compiler error code and readable program log are the diagnostic evidence.

The final corrected-tracer pair (`external-trace-v2-d128.log` and
`external-trace-v2-d256.log`) preserves D128 numerical PASS and D256 backward
FAIL. D256 again reports the same seven undefined-identifier errors. Controls
without the tracer and in the image without the FA2 patch preserve the same pair.

## Optional NVRTC diagnostic

Build `nvrtc_trace.c` on target Linux with the command in its header. Preload it
only for one bounded diagnostic process. It handles direct calls and glibc
`dlsym("nvrtcCompileProgram")` resolution, leaves compile arguments and return
codes unchanged, and emits no output for successful compiles. Each failed
compile writes at most 64 KiB. Source dumping is absent. Program logs over
8 MiB are omitted to cap allocation. Options are captured before compilation,
limited to 128 entries of 256 bytes each, and restricted to printable ASCII in
the diagnostic output.

Local `cc -std=c11 -fsyntax-only -Wall -Wextra` passed. The experiment driver
reported a successful Linux shared-library build, an unchanged baseline
failure with no tracer output, and captured compiler errors under external
runtime linking. The tracer's coverage of the default bundled compiler is
therefore absent; the external-provider control is separate.
This source-review subtask did not run GPUs.
