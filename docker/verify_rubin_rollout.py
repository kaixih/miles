#!/usr/bin/env python3
"""Offline Rubin smoke: SGLang runtime import and FlashInfer BF16 decode.

Run inside the cu134 image with a writable FlashInfer cache and --network none.
The first decode may compile one kernel from FlashInfer's bundled CUDA sources.
No model, tokenizer, server, or external kernel artifact is requested.

APIs checked against:
  https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/decode.py
  https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/flashinfer/jit/attention/modules.py
  https://github.com/sgl-project/sglang/blob/aea7fb92c047c9c096eae66460acb25dec9ae5a9/python/sglang/srt/layers/attention/flashinfer_backend.py
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import time


def report(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    args = parser.parse_args()

    # Container --network none also enforces the no-download boundary for JIT.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("MAX_JOBS", "4")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this smoke requires a Rubin GPU")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 7):
        raise RuntimeError("Expected Rubin CC 10.7, got %r" % (capability,))
    if torch.version.cuda != "13.4":
        raise RuntimeError("Expected the native cu134 Torch build, got %r" % torch.version.cuda)
    fi_version = importlib.metadata.version("flashinfer-python")
    if fi_version != "0.6.18":
        raise RuntimeError("Expected FlashInfer 0.6.18, got %r" % fi_version)

    # Import the real runtime backend and its native dependencies. Instantiating
    # it would require a ModelRunner and model weights, outside this smoke.
    backend = importlib.import_module("sglang.srt.layers.attention.flashinfer_backend")
    if not hasattr(backend, "FlashInferAttnBackend"):
        raise RuntimeError("SGLang runtime backend class is missing")
    import flashinfer

    report(
        "runtime_import_pass",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=fi_version,
        sglang=importlib.metadata.version("sglang"),
        sglang_runtime_file=backend.__file__,
        gpu=torch.cuda.get_device_name(device),
        capability=list(capability),
        device=args.device,
    )

    # Both lengths share the same JIT specialization; the second call checks
    # a fresh input through the cached kernel. Decode attends to the entire KV
    # sequence, so the independent float32 SDPA reference is non-causal.
    torch.manual_seed(20260914)
    heads, head_dim = 4, 64
    with torch.inference_mode():
        for kv_len in (127, 257):
            q = torch.randn(heads, head_dim, device=device, dtype=torch.bfloat16)
            k = torch.randn(kv_len, heads, head_dim, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            started = time.monotonic()
            report("flashinfer_decode_start", kv_len=kv_len)
            actual = flashinfer.single_decode_with_kv_cache(
                q,
                k,
                v,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                use_tensor_cores=False,
            )
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            if actual.dtype != torch.bfloat16 or actual.device != device:
                raise RuntimeError("FlashInfer did not return BF16 output on the requested GPU")
            with sdpa_kernel(SDPBackend.MATH):
                expected = F.scaled_dot_product_attention(
                    q.float()[None, :, None, :],
                    k.float().transpose(0, 1)[None, :, :, :],
                    v.float().transpose(0, 1)[None, :, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                )[0, :, 0, :]
            torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
            report(
                "flashinfer_decode_pass",
                kv_len=kv_len,
                heads=heads,
                head_dim=head_dim,
                output_shape=list(actual.shape),
                max_abs_error=(actual.float() - expected).abs().max().item(),
                elapsed_seconds=round(elapsed, 3),
                elapsed_includes_jit=True,
            )
    report("rollout_native_smoke_pass")


if __name__ == "__main__":
    main()
