#!/usr/bin/env python3
"""Small, download-free runtime checks for a Miles Rubin container.

Single GPU: python /opt/miles-smoke/verify_rubin.py
NCCL: python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 /opt/miles-smoke/verify_rubin.py --distributed

Required checks fail the process. FlashAttention is skipped only when absent;
an installed but broken or unsupported FlashAttention is a failure.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F


def log(message):
    print(message, flush=True)


def finite_nonzero(name, value):
    if value is None:
        raise AssertionError(f"{name}: missing tensor")
    if not torch.isfinite(value).all().item():
        raise AssertionError(f"{name}: nonfinite values")
    if not torch.count_nonzero(value).item():
        raise AssertionError(f"{name}: all zeros")


def environment():
    log(f"Python={sys.version.split()[0]} machine={platform.machine()}")
    for package in (
        "torch", "transformer-engine", "transformer-engine-torch",
        "transformer-engine-cu13", "apex", "flash-attn", "flash-attn-3",
        "sglang", "megatron-core", "torch-memory-saver", "miles",
        "flashinfer-python", "triton", "nvidia-cudnn-cu13",
    ):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "<no distribution metadata>"
        log(f"package {package}={version}")
    log(f"torch CUDA={torch.version.cuda} cuDNN={torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    log(f"compiled torch architectures={torch.cuda.get_arch_list()}")
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        log(f"GPU {index}: {torch.cuda.get_device_name(index)}, CC={capability}")
        if capability != (10, 7):
            raise AssertionError(f"GPU {index}: expected Rubin CC (10, 7), got {capability}")
    if not torch.cuda.is_bf16_supported():
        raise AssertionError("PyTorch does not report BF16 support")


def imports():
    for name in (
        "miles", "sglang", "megatron.core",
        "megatron.core.extensions.transformer_engine", "torch_memory_saver",
    ):
        module = importlib.import_module(name)
        log(f"import {name}: {getattr(module, '__file__', '<namespace>')}")
    from torch_memory_saver.utils import get_binary_path_from_package

    binary = Path(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
    if not binary.is_file():
        raise FileNotFoundError(f"TMS preload binary is missing: {binary}")
    log(f"TMS preload binary: {binary} ({binary.stat().st_size} bytes)")


def miles_actor_import():
    # This traverses the real training actor, model provider, optimizer and
    # checkpoint imports without initializing Ray or loading a model.
    module = importlib.import_module("miles.backends.megatron_utils.actor")
    assert hasattr(module, "MegatronTrainRayActor")
    log(f"Miles Megatron actor: {module.__file__}")


def bf16_gemm():
    x = torch.rand(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.rand(128, 96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().float().requires_grad_()
    weight_ref = weight.detach().float().requires_grad_()
    result = x @ weight
    reference = x_ref @ weight_ref
    result.float().square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.02)
    for name, actual, expected in (
        ("input grad", x.grad, x_ref.grad),
        ("weight grad", weight.grad, weight_ref.grad),
    ):
        finite_nonzero(name, actual)
        torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.01)
    finite_nonzero("GEMM output", result)


def te_linear():
    import transformer_engine.pytorch as te

    layer = te.Linear(128, 64, bias=True, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = layer.weight.detach().clone().requires_grad_()
    bias_ref = layer.bias.detach().clone().requires_grad_()
    result = layer(x)
    reference = F.linear(x_ref, weight_ref, bias_ref)
    result.float().square().mean().backward()
    reference.float().square().mean().backward()
    torch.testing.assert_close(result, reference, rtol=0.03, atol=0.01)
    for name, actual, expected in (
        ("TE input grad", x.grad, x_ref.grad),
        ("TE weight grad", layer.weight.grad, weight_ref.grad),
        ("TE bias grad", layer.bias.grad, bias_ref.grad),
    ):
        finite_nonzero(name, actual)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.002)
    finite_nonzero("TE output", result)


def check_adam(optimizer_class, label):
    parameter = torch.nn.Parameter(torch.linspace(0.1, 1.0, 128, device="cuda"))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_class([parameter], lr=0.001, adam_w_mode=False)
    reference_optimizer = torch.optim.Adam([reference], lr=0.001, foreach=False, fused=False)
    initial = parameter.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        parameter.square().mean().backward()
        reference.square().mean().backward()
        optimizer.step()
        reference_optimizer.step()
    finite_nonzero(f"{label} parameters", parameter)
    torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=1e-6)
    if torch.equal(parameter, initial):
        raise AssertionError(f"{label} optimizer did not update parameters")
    if parameter.square().mean() >= initial.square().mean():
        raise AssertionError(f"{label} optimizer did not reduce the loss")


def te_adam():
    from megatron.core.optimizer import Adam
    from transformer_engine.pytorch.optimizers import FusedAdam

    if Adam is not FusedAdam:
        raise AssertionError(f"Megatron selected {Adam}, expected TE FusedAdam")
    log(f"Megatron Adam implementation: {Adam.__module__}.{Adam.__name__}")
    # Standard Megatron BF16 training hands FP32 master parameters to Adam.
    check_adam(Adam, "TE")


def apex_adam():
    from apex.optimizers import FusedAdam

    check_adam(FusedAdam, "Apex")


def flash_attention():
    if importlib.util.find_spec("flash_attn") is None:
        log("SKIP FlashAttention: flash_attn is not installed")
        return "skipped"
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel

    tensors = [
        torch.randn(2, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    references = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    result = flash_attn_func(*tensors, dropout_p=0.0, causal=True)
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(
            *(tensor.transpose(1, 2) for tensor in references), dropout_p=0.0, is_causal=True,
        ).transpose(1, 2)
    result.float().square().mean().backward()
    reference.float().square().mean().backward()
    torch.testing.assert_close(result, reference, rtol=0.05, atol=0.02)
    for name, actual, expected in zip("qkv", tensors, references):
        finite_nonzero(f"FlashAttention {name} grad", actual.grad)
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0.08, atol=2e-5)
    finite_nonzero("FlashAttention output", result)


def nccl_all_reduce():
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("--distributed requires torchrun with at least two processes")
    torch.cuda.set_device(local_rank)
    capability = torch.cuda.get_device_capability()
    if capability != (10, 7):
        raise AssertionError(f"rank {local_rank}: expected CC (10, 7), got {capability}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        rank = dist.get_rank()
        value = torch.full((4096,), float(rank + 1), device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        expected = world_size * (world_size + 1) / 2
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
        log(f"PASS NCCL rank={rank}/{world_size} device={local_rank} sum={expected}")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true", help="run only the torchrun NCCL check")
    args = parser.parse_args()
    if args.distributed:
        nccl_all_reduce()
        return
    environment()
    torch.cuda.set_device(0)
    torch.manual_seed(107)
    failed = []
    skipped = []
    for check in (imports, miles_actor_import, bf16_gemm, te_linear, te_adam, apex_adam, flash_attention):
        log(f"RUN {check.__name__}")
        try:
            result = check()
            torch.cuda.synchronize()
            if result == "skipped":
                skipped.append(check.__name__)
            else:
                log(f"PASS {check.__name__}")
        except Exception:
            failed.append(check.__name__)
            log(f"FAIL {check.__name__}")
            traceback.print_exc(file=sys.stdout)
    log(f"SUMMARY failures={failed} skipped={skipped}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
