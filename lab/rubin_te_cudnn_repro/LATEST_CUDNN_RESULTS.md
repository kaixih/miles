# cuDNN 9.26 comparison

**The d256 backward failure is resolved in every tested input with cuDNN
9.26.0.51.** Only cuDNN changed; the other 399 installed package versions and
driver 610.47.04 stayed the same.

| cuDNN | Input | Forward / backward |
| --- | --- | --- |
| 9.25.0.15 | THD, 16 tokens, 1 head, d128 | PASS / PASS |
| 9.25.0.15 | Same input, d256 | PASS / FAIL |
| 9.26.0.51 | THD, 16 tokens, 1 head, d128 | PASS / PASS |
| 9.26.0.51 | Same input, d256 | PASS / PASS |
| 9.26.0.51 | GQA 8:1, d256, lengths `[512]` | PASS / PASS |
| 9.26.0.51 | GQA 8:1, d256, lengths `[257,385,129]` | PASS / PASS |

All inputs used BF16 causal attention with zero dropout. Both GQA cases also
passed output and Q/K/V-gradient comparisons against PyTorch FP32 math; the
largest relative L2 error was **0.3081%**. Mapped libraries, `cudnnGetVersion()`,
and PyTorch all confirmed runtime version **92600** in each latest test.

Exact versions, input dimensions, wheel URL/SHA256, image digests, numerical
errors and log hashes are in [latest_cudnn_results.json](latest_cudnn_results.json).
These single-GPU tests do not establish full-model cuDNN training validation.

## Reproduce the isolated upgrade

Use a disposable container from the tested base manifest digest
`sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`.
Download the aarch64 wheel from the URL in the result JSON and verify SHA256:

```text
8e5b4f5eccff721416f70ecf4353172fc4d5cdd8ca869fe6f983a4a5b0147b20
```

Run the following inside that disposable container, with a container-local
working directory. The constraint override applies only to this command:

```bash
PIP_CONSTRAINT=/dev/null python3 -m pip install \
  --no-deps --no-cache-dir --force-reinstall \
  ./nvidia_cudnn_cu13-9.26.0.51-py3-none-manylinux_2_27_aarch64.whl
python3 minimal_repro.py 128
python3 minimal_repro.py 256
python3 repro.py --heads 8 --kv-heads 1 --dim 256 --lengths 512 --check
python3 repro.py --heads 8 --kv-heads 1 --dim 256 --lengths 257,385,129 --check
```

Compare package snapshots before and after; only `nvidia-cudnn-cu13` should
change. Each process used a 120-second timeout and a 32 MiB log limit.
No external NVRTC override or diagnostic preload was used.
