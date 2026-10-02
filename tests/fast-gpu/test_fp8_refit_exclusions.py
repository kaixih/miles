from tests.ci.ci_register import register_cuda_ci

# These CPU tensor tests import SGLang's GPU-dependent quantization stack.
register_cuda_ci(est_time=10, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8


@pytest.fixture
def weight():
    return torch.tensor([[-2.0, -1.0], [0.0, 2.0]], dtype=torch.bfloat16)


def _quantize(megatron_suffix, names, weight, **config):
    return dict(
        quantize_params_fp8(
            SimpleNamespace(),
            f"module.module.decoder.layers.{megatron_suffix}",
            [(name, weight) for name in names],
            {"quant_method": "fp8", "activation_scheme": "dynamic", **config},
        )
    )


def _assert_quantized(result, name):
    assert result[name].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(result[name].float(), torch.tensor([[-448.0, -224.0], [0.0, 448.0]]))
    scale = result[name.removesuffix(".weight") + ".weight_scale"]
    torch.testing.assert_close(scale, torch.tensor([2.0 / 448.0]), rtol=0, atol=0)


@pytest.mark.parametrize("config_key", ["ignored_layers", "modules_to_not_convert"])
@pytest.mark.parametrize("hf_prefix,ignored_prefix", [("model.", ""), ("", "model."), ("model.", "model."), ("", "")])
@pytest.mark.parametrize(
    "megatron_suffix,hf_suffix",
    [
        ("linear_kv_up_proj", "kv_b_proj"),
        ("wq_b", "indexer.wq_b"),
        ("wk", "indexer.wk"),
    ],
)
def test_glm_dsa_exclusions_preserve_bf16(weight, config_key, hf_prefix, ignored_prefix, megatron_suffix, hf_suffix):
    module = f"layers.3.self_attn.{hf_suffix}"
    name = f"{hf_prefix}{module}.weight"
    result = _quantize(
        f"3.self_attention.{megatron_suffix}.weight", [name], weight, **{config_key: [ignored_prefix + module]}
    )
    assert list(result) == [name]
    assert result[name] is weight


def test_block_exclusion_bypasses_quantization(weight):
    name = "model.layers.3.self_attn.kv_b_proj.weight"
    result = _quantize(
        "3.self_attention.linear_kv_up_proj.weight",
        [name],
        weight,
        modules_to_not_convert=[name.removesuffix(".weight")],
        weight_block_size=[128, 128],
    )
    assert list(result) == [name]
    assert result[name] is weight


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"modules_to_not_convert": ["mlp.gate"]},
        {"ignored_layers": [], "modules_to_not_convert": ["mlp"]},
        {"ignored_layers": None, "modules_to_not_convert": ["mlp"]},
        {"ignored_layers": ["self_attn"], "modules_to_not_convert": ["mlp"]},
    ],
)
def test_unexcluded_weights_quantize_with_scale(weight, config):
    name = "model.layers.3.mlp.gate_up_proj.weight"
    result = _quantize("3.mlp.linear_fc1.weight", [name], weight, **config)
    assert len(result) == 2
    _assert_quantized(result, name)


@pytest.mark.parametrize("mlp", ["mlp", "mlp.shared_experts"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("mapping", [{}, {"gate_up_proj": ["left", "right"]}])
def test_fused_projections_require_consistent_exclusions(weight, mlp, partial, mapping):
    projections = mapping.get("gate_up_proj", ["gate_proj", "up_proj"])
    if partial:
        projections = projections[:1]
    ignored = [f"{mlp}.{projection}" for projection in projections]
    name = f"model.layers.3.{mlp}.gate_up_proj.weight"
    config = {"modules_to_not_convert": ignored, "packed_modules_mapping": mapping}
    if partial:
        with pytest.raises(ValueError, match="some but not all shards"):
            _quantize(f"3.{mlp}.linear_fc1.weight", [name], weight, **config)
    else:
        result = _quantize(f"3.{mlp}.linear_fc1.weight", [name], weight, **config)
        assert list(result) == [name]
        assert result[name] is weight


@pytest.mark.parametrize(
    "linear,projections", [("linear_fc1", ["gate_proj", "up_proj"]), ("linear_fc2", ["down_proj"])]
)
def test_routed_expert_exclusion_keeps_entire_fused_moe_bf16(weight, linear, projections):
    config = {"modules_to_not_convert": ["model.layers.3.mlp.experts.0.gate_proj"]}
    for layer in (3, 4):
        for expert in (0, 1):
            names = [f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight" for projection in projections]
            result = _quantize(f"{layer}.mlp.experts.{linear}.weight{expert}", names, weight, **config)
            if layer == 3:
                assert list(result) == names
                assert all(result[name] is weight for name in names)
            else:
                assert len(result) == 2 * len(names)
                for name in names:
                    _assert_quantized(result, name)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
