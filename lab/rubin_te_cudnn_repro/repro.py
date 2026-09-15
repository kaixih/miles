#!/usr/bin/env python3
"""Single-GPU PyTorch + Transformer Engine attention forward/backward probe."""

import argparse
import importlib.metadata
import json
import os
import time
import traceback


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="512")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--v-dim", type=int)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--format", choices=["thd", "bshd", "sbhd"], default="thd")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--max-seqlen", type=int)
    parser.add_argument("--backend", choices=["cudnn", "flash"], default="cudnn")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--padded-boundaries", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--check", action="store_true", help="Compare with PyTorch FP32 math SDPA")
    args = parser.parse_args()
    os.environ["NVTE_FLASH_ATTN"] = str(int(args.backend == "flash"))
    os.environ["NVTE_FUSED_ATTN"] = str(int(args.backend == "cudnn"))
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    os.environ.setdefault("NVTE_DEBUG", "1")
    os.environ.setdefault("NVTE_DEBUG_LEVEL", "2")
    import torch
    import torch.nn.functional as F
    import transformer_engine
    from transformer_engine.pytorch import DotProductAttention

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(args.deterministic)
    lengths = [int(x) for x in args.lengths.split(",")]
    assert lengths and min(lengths) > 0
    assert args.heads % args.kv_heads == 0
    if args.format != "thd":
        assert len(set(lengths)) == 1, "Dense formats require equal lengths"
    maximum = args.max_seqlen or max(lengths)
    assert maximum >= max(lengths)
    mask = args.mask or ("padding_causal" if args.format == "thd" else "causal")
    assert mask in {"padding_causal", "causal", "padding", "no_mask"}
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    cu = torch.tensor(cumulative, device="cuda", dtype=torch.int32)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    versions = {"torch": torch.__version__, "te": transformer_engine.__version__,
                "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    for package in ["nvidia-cudnn-cu13", "nvidia-cudnn-frontend"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    emit("config", args=vars(args), lengths=lengths, effective_mask=mask,
         effective_max_seqlen=maximum, versions=versions,
         gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
         env={k: v for k, v in os.environ.items() if k.startswith(("NVTE_", "CUDNN_", "CUDA_FORCE_", "CUDA_DISABLE_"))})

    v_dim = args.v_dim or args.dim

    def make(heads, dim):
        shape = ((sum(lengths), heads, dim) if args.format == "thd" else
                 (len(lengths), lengths[0], heads, dim) if args.format == "bshd" else
                 (lengths[0], len(lengths), heads, dim))
        return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)

    q, k, v = make(args.heads, args.dim), make(args.kv_heads, args.dim), make(args.kv_heads, v_dim)
    emit("tensors", **{name: {"shape": list(t.shape), "stride": list(t.stride())}
                       for name, t in [("q", q), ("k", k), ("v", v)]},
         cu_seqlens=cumulative, padded_boundaries=cumulative if args.padded_boundaries else None)
    layer = DotProductAttention(args.heads, (args.dim, v_dim), num_gqa_groups=args.kv_heads,
                                qkv_format=args.format, attention_dropout=0.0,
                                attn_mask_type=mask).cuda()
    kwargs = {"max_seqlen_q": maximum, "max_seqlen_kv": maximum,
              "checkpoint_core_attention": args.checkpoint}
    if args.format == "thd":
        kwargs.update(cu_seqlens_q=cu, cu_seqlens_kv=cu)
        if args.padded_boundaries:
            kwargs.update(cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu)
    start = time.monotonic()
    emit("forward_start")
    out = layer(q, k, v, **kwargs).reshape(*q.shape[:-1], v_dim)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    emit("forward_pass", elapsed_s=time.monotonic() - start)
    dout = torch.randn_like(out)
    emit("backward_start", dout_norm=dout.float().norm().item())
    out.backward(dout)
    torch.cuda.synchronize()
    for tensor in [q, k, v]:
        assert torch.isfinite(tensor.grad).all()
    assert v.grad.float().norm() > 0
    emit("backward_pass", elapsed_s=time.monotonic() - start,
         grad_norms={name: t.grad.float().norm().item() for name, t in [("dq", q), ("dk", k), ("dv", v)]})
    if args.check:
        qr, kr, vr = [t.detach().float().requires_grad_(True) for t in [q, k, v]]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if args.format == "thd":
                refs = []
                for lo, hi in zip(cumulative[:-1], cumulative[1:]):
                    tensors = [t[lo:hi].transpose(0, 1)[None] for t in [qr, kr, vr]]
                    ref_part = F.scaled_dot_product_attention(*tensors, is_causal="causal" in mask, enable_gqa=True)
                    refs.append(ref_part[0].transpose(0, 1))
                ref = torch.cat(refs)
            else:
                permutation = (0, 2, 1, 3) if args.format == "bshd" else (1, 2, 0, 3)
                tensors = [t.permute(permutation) for t in [qr, kr, vr]]
                ref = F.scaled_dot_product_attention(*tensors, is_causal="causal" in mask, enable_gqa=True)
                ref = ref.permute(0, 2, 1, 3) if args.format == "bshd" else ref.permute(2, 0, 1, 3)
        ref.backward(dout.float())
        errors = {}
        for name, actual, expected in [("out", out, ref), ("dq", q.grad, qr.grad),
                                       ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)]:
            difference = (actual.float() - expected).norm()
            denominator = expected.norm()
            errors[name] = (difference / denominator).item() if denominator > 0 else difference.item()
            assert errors[name] < 0.02, (name, errors[name])
        emit("numerical_pass", relative_l2=errors)
    emit("pass", elapsed_s=time.monotonic() - start, max_memory_allocated=torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("fail", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise SystemExit(1)
