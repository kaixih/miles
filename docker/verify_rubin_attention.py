"""One TE/cuDNN BF16 attention check; run in a fresh GPU-enabled process.

API checked against TransformerEngine commit
5e52befd5262c06289106338c308079d6adb391f (2.19).
No models, checkpoints, downloads, process groups, or output files.
"""


def te_fused_attention():
    import os
    from unittest.mock import patch

    # Ensure both passes use TE's cuDNN path, with no FlashAttention fallback.
    backend_env = {
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
        "NVTE_FUSED_ATTN_USE_FAv2_BWD": "0",
    }
    with patch.dict(os.environ, backend_env):
        import torch
        import torch.nn.functional as F
        import transformer_engine.pytorch as te
        from torch.nn.attention import SDPBackend, sdpa_kernel

        assert torch.cuda.get_device_capability() == (10, 7)
        torch.manual_seed(107)
        attention = te.DotProductAttention(
            num_attention_heads=4,
            kv_channels=64,
            attention_dropout=0.0,
            qkv_format="bshd",
            attn_mask_type="causal",
        ).train()
        selected = []

        def record_backend(_module, _args, kwargs, _output):
            selected.append(kwargs["fused_attention_backend"])

        hook = attention.fused_attention.register_forward_hook(record_backend, with_kwargs=True)
        tensors = [
            torch.randn(2, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]
        references = [tensor.detach().float().requires_grad_() for tensor in tensors]
        try:
            output = attention(*tensors)
        finally:
            hook.remove()
        assert len(selected) == 1, f"Expected one cuDNN fused-attention call, got {selected}"

        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(
                *(tensor.transpose(1, 2) for tensor in references),
                dropout_p=0.0,
                is_causal=True,
            ).transpose(1, 2).reshape(2, 64, 256)
        upstream_grad = torch.randn_like(output)
        grads = torch.autograd.grad(output, tensors, grad_outputs=upstream_grad)
        reference_grads = torch.autograd.grad(reference, references, grad_outputs=upstream_grad.float())
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=0.02)
        for name, actual, expected in zip("qkv", grads, reference_grads):
            assert torch.isfinite(actual).all().item(), f"Nonfinite {name} gradient"
            assert torch.count_nonzero(actual).item(), f"Zero {name} gradient"
            torch.testing.assert_close(actual.float(), expected, rtol=0.08, atol=0.035)
        print(f"PASS TE/cuDNN BF16 causal attention forward/backward: backend={selected[0]}", flush=True)


if __name__ == "__main__":
    te_fused_attention()
