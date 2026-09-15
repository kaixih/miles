"""Enable the numerically checked Rubin BF16/d256/no-dropout FA2 path in TE 2.19.

The FA2 CUDA kernels are already built for SM107 in the base image. TE's Python
head-dimension check omits SM107, while its cuDNN backward path fails for this
Qwen3.5 packed-attention configuration. All other backend checks stay in force.
"""

import hashlib
import importlib.metadata
from pathlib import Path


def main():
    dist = importlib.metadata.distribution("transformer_engine")
    if not dist.version.startswith("2.19.0"):
        raise RuntimeError(f"Expected the pinned TE 2.19 build, got {dist.version}")
    path = Path(dist.locate_file("transformer_engine/pytorch/attention/dot_product_attention/utils.py"))
    original = path.read_text()
    old = "                and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))\n"
    new = old + """                # Verified FA2 forward/backward path for the Rubin Qwen3.5 recipe.
                and not (
                    device_compute_capability == (10, 7)
                    and qkv_dtype == torch.bfloat16
                    and head_dim_qk == 256
                    and head_dim_v == 256
                    and attention_dropout == 0.0
                )
"""
    if new in original:
        print(f"TE_SM107_FA2_PATCH_ALREADY_APPLIED {path}")
        return
    if original.count(old) != 1:
        raise RuntimeError("Pinned TE head-dimension check changed; review before patching")
    patched = original.replace(old, new)
    compile(patched, str(path), "exec")
    path.write_text(patched)
    print(f"TE_SM107_FA2_PATCH_APPLIED version={dist.version} path={path}")
    print(f"before_sha256={hashlib.sha256(original.encode()).hexdigest()}")
    print(f"after_sha256={hashlib.sha256(patched.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
