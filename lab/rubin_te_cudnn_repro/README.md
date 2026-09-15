# Rubin TE / cuDNN attention backward reproduction

A single Rubin GPU reproduces the original cuDNN backward plan-building failure
using only random PyTorch tensors and Transformer Engine. The compact example
uses **one head, 16 tokens, BF16, d256, causal THD attention, dropout 0**.
Forward succeeds; backward fails. Changing only d256 to d128 passes.

See [ISSUE_DRAFT.md](ISSUE_DRAFT.md) for the English report and
[SOURCE_NOTES.md](SOURCE_NOTES.md) for source-level evidence. No issue has been
submitted. The original training workload is unnecessary for this reproduction.

## Run the minimal pair

Use the existing pinned environment described in [environment.json](environment.json).
No installation or dependency resolution is performed by these scripts.

```bash
timeout -k 10s 120s python3 -u minimal_repro.py 256
timeout -k 10s 120s python3 -u minimal_repro.py 128
```

Expected on the recorded stack:

| Dimension | Forward | Backward | Process exit |
| --- | --- | --- | --- |
| 256 | PASS | `No valid execution plans built`, TE `.cu:1025` | 1 |
| 128 | PASS | PASS, finite nonzero gradients | 0 |

The script sets `NVTE_FLASH_ATTN=0`, `NVTE_FUSED_ATTN=1`, and
`NVTE_UNFUSED_ATTN=0` before importing TE, so another backend cannot mask the
failure. There are no weights, datasets, distributed initialization, checkpoint
recomputation, model modules, or framework launchers.

For a numerical comparison with PyTorch FP32 math attention:

```bash
timeout -k 10s 120s python3 -u repro.py --heads 1 --lengths 16 --dim 128 --check
timeout -k 10s 120s python3 -u repro.py --heads 1 --lengths 16 --dim 256 --check
```

`repro.py` defaults to the original per-rank head dimensions (Q heads 8, KV
heads 1, d256, BF16, causal THD) with a representative sequence length of 512. Use `--help` for controlled changes.
`--checkpoint` exercises TE core-attention checkpointing. The reproduction does
not require the original full-block activation recomputation.

## Evidence and matrix

[matrix_results.json](matrix_results.json) contains sanitized per-case results,
including input parameters, actual failure stage, and numerical errors for
passing controls. Unsupported inputs and reference-toolchain failures are
classified separately from cuDNN backward failures.

```bash
python3 run_matrix.py matrix_initial.json /writable/output/matrix-initial --timeout 90
python3 run_matrix.py matrix_followup.json /writable/output/matrix-followup --timeout 90
```

Each output directory must be new. The Linux runner launches cases sequentially,
caps each log at 32 MiB, disables core dumps, and kills the case's process group
on timeout. Read each case's `exit_code` and events in `results.json`; the runner's
own successful exit means collection finished, not that every case passed.
For any zero-norm reference, the probe reports absolute L2 error in the same
error field to avoid division by zero.
The 23-case and 10-case lists are exploratory evidence, including deliberately
invalid combinations; use the two minimal commands for a focused report.

## Compiler diagnostics

The unmodified baseline cuDNN log identifies
`CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED`, underneath TE's generic error:

```bash
(ulimit -c 0; ulimit -f 32768
 CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stderr \
   timeout -k 10s 120s python3 -u minimal_repro.py 256 > cudnn-debug.log 2>&1)
```

An optional diagnostic shim exposes the compiler's detailed error. Build on the
target Linux architecture into a writable temporary directory:

```bash
cc -std=c11 -shared -fPIC -O2 -Wall -Wextra nvrtc_trace.c -ldl -o /tmp/nvrtc_trace.so
(ulimit -c 0; ulimit -f 32768
 CUDNN_NVRTC_RUNTIME_LINKING=1 LD_PRELOAD=/tmp/nvrtc_trace.so \
   timeout -k 10s 120s python3 -u minimal_repro.py 256 > nvrtc-diagnostic.log 2>&1)
```

This is a **separate diagnostic configuration**: the environment switch makes
cuDNN use the existing external NVRTC library instead of its bundled compiler.
It is not a fix. The shim preserves compile inputs and return codes, records
only failed compiles, bounds each record to 64 KiB, and does not dump source.
Default compilation and external compilation both fail; the external diagnostic
reports undefined identifiers in cuDNN's generated SM107 d256 backward kernel.

Raw logs, node identities and registry access details are retained in the
private experiment record. Shared files contain portable scripts, exact version
identities, compact results and the draft report.
