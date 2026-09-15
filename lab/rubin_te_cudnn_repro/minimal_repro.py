"""Run: python minimal_repro.py [256|128]. Requires one CUDA GPU, Torch and TE."""

import os
import sys

# Set before importing TE; forbid fallback to another attention implementation.
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

import torch
from transformer_engine.pytorch import DotProductAttention

torch.manual_seed(123)
dim = int(sys.argv[1]) if len(sys.argv) > 1 else 256
length = 16
cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
q, k, v = [torch.randn(length, 1, dim, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
attention = DotProductAttention(
    1, dim, num_gqa_groups=1, qkv_format="thd",
    attention_dropout=0.0, attn_mask_type="padding_causal",
).cuda()
out = attention(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu,
                max_seqlen_q=length, max_seqlen_kv=length)
torch.cuda.synchronize()
print(f"forward PASS: dim={dim}", flush=True)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
assert all(torch.isfinite(t.grad).all() for t in [q, k, v])
assert all(t.grad.float().norm() > 0 for t in [q, k, v])
print(f"backward PASS: dim={dim}", flush=True)
