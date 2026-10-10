import json
import shlex
from dataclasses import replace
from unittest.mock import Mock

import pytest
from scripts import run_glm5_3_flash
from scripts.run_glm5_3_flash import ScriptArgs, _train
from typer.testing import CliRunner


@pytest.fixture
def official_fp8_config():
    # Projected from the retained official-fp8-config.json, not a text-only HF schema.
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 45},
    }


def test_direct_hf_changes_only_checkpoint_initialization(monkeypatch, tmp_path, official_fp8_config):
    """An opt-in must preserve the full-model training and rollout recipe."""
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(official_fp8_config))
    backend = Mock()
    monkeypatch.setattr(ScriptArgs, "create_backend", lambda self: backend)
    monkeypatch.setattr("scripts.run_glm5_3_flash.U.get_default_wandb_args", lambda *args, **kwargs: "")
    args = ScriptArgs(model_name="GLM-5.3-Flash", num_nodes=8, hf_checkpoint=str(checkpoint), num_rollout=2)

    _train(args)
    converted = backend.execute_train.call_args.kwargs
    _train(replace(args, direct_hf_init=True))
    direct = backend.execute_train.call_args.kwargs

    converted_argv = shlex.split(converted["train_args"])
    direct_argv = shlex.split(direct["train_args"])
    assert converted_argv[converted_argv.index("--ref-load") + 1] == "/root/ckpt/glm5.3-flash_torch_dist"
    assert direct_argv[direct_argv.index("--ref-load") + 1] == str(checkpoint)
    assert direct_argv[direct_argv.index("--megatron-to-hf-mode") + 1] == "raw"
    assert "--custom-megatron-init-path" not in direct_argv
    assert direct_argv[direct_argv.index("--num-rollout") + 1] == "2"
    assert direct["megatron_model_type"] == "glm5.3-flash"

    # Compare every other argument and the runtime environment, not a subset of defaults.
    mode_index = direct_argv.index("--megatron-to-hf-mode")
    del direct_argv[mode_index : mode_index + 2]
    direct_argv[direct_argv.index("--ref-load") + 1] = args.ref_load
    assert direct_argv == converted_argv
    assert direct | {"train_args": converted["train_args"]} == converted


def test_default_still_uses_the_converted_slice_checkpoint():
    args = ScriptArgs()
    assert not args.direct_hf_init
    assert args.ref_load == "/root/ckpt/glm5.3-flash-4layer_torch_dist"


def test_direct_hf_cli_passes_the_requested_checkpoint_and_update_count(monkeypatch, tmp_path, official_fp8_config):
    (tmp_path / "config.json").write_text(json.dumps(official_fp8_config))
    launch = Mock()
    monkeypatch.setattr(run_glm5_3_flash, "_train", launch)
    result = CliRunner().invoke(
        run_glm5_3_flash.app,
        [
            "--model-name",
            "GLM-5.3-Flash",
            "--num-nodes",
            "8",
            "--hf-checkpoint",
            str(tmp_path),
            "--direct-hf-init",
            "--num-rollout",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    (args,) = launch.call_args.args
    assert args.direct_hf_init and args.ref_load == str(tmp_path)
    assert args.num_rollout == 2


def test_direct_hf_rejects_the_slice_recipe_before_reading_weights():
    with pytest.raises(ValueError, match="requires the full GLM-5.3-Flash"):
        ScriptArgs(direct_hf_init=True)


@pytest.mark.parametrize(
    ("model_type", "num_hidden_layers"),
    [
        ("glm5_next", 4),
        ("glm_moe_dsa", 45),
    ],
)
def test_direct_hf_rejects_an_incompatible_checkpoint(tmp_path, official_fp8_config, model_type, num_hidden_layers):
    config = official_fp8_config
    config["model_type"] = model_type
    config["text_config"]["num_hidden_layers"] = num_hidden_layers
    # A top-level value cannot override the actual text model's layer count.
    config["num_hidden_layers"] = 45
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires a 45-layer glm5_next HF checkpoint"):
        ScriptArgs(model_name="GLM-5.3-Flash", hf_checkpoint=str(tmp_path), direct_hf_init=True)
