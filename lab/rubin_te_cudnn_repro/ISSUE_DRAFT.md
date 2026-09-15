# Rubin: attention backward failure with cuDNN 9.25; passes with 9.26

## Summary

On Rubin (SM10.7), we hit an attention backward failure with Transformer Engine
2.19.0 and cuDNN 9.25.0.15. BF16 causal attention with head dimension 256 passes
forward but fails backward with `No valid execution plans built`. The same
script passes when the head dimension is 128. Updating only cuDNN to 9.26.0.51
makes both head dimensions pass on the same GPU, with the other versions unchanged.

## Environment

| Component | Version |
| --- | --- |
| GPU | Rubin, SM10.7, one GPU |
| Platform | Linux aarch64 |
| NVIDIA driver | 610.47.04 |
| CUDA | 13.4.46 |
| PyTorch | `2.15.0.dev20260818+cu134` |
| Transformer Engine | `2.19.0+5e52bef` |
| cuDNN | `9.25.0.15` and `9.26.0.51` |
| cuDNN frontend | `1.28.0` |

## Reproduce

Save this as `minimal_repro.py`. It uses only PyTorch and TE, with one head and
16 tokens, and forces the cuDNN attention backend.

```python
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
```

Run with either cuDNN version:

```bash
python3 minimal_repro.py 256
python3 minimal_repro.py 128
```

| cuDNN | Head dimension | Forward | Backward |
| --- | --- | --- | --- |
| 9.25.0.15 | 256 | PASS | FAIL |
| 9.25.0.15 | 128 | PASS | PASS |
| 9.26.0.51 | 256 | PASS | PASS |
| 9.26.0.51 | 128 | PASS | PASS |

The cuDNN 9.25.0.15 / d256 run reports:

```text
fused_attn_f16_arbitrary_seqlen.cu:1025
cuDNN Error: [cudnn_frontend] Error: No valid execution plans built.
```
