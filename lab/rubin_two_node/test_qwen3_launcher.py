"""CPU-only launcher contracts; no Torch, Ray, shell command, or GPU is used."""

import importlib.util
import shlex
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass
class _Config:
    output_dir: str = "/run-output"


def _load_launcher():
    utility = types.ModuleType("miles.utils.external_utils.command_utils")
    utility.ExecuteTrainConfig = _Config
    utility.create_run_id = lambda: "cpu-test"
    utility.dataclass_cli = lambda function: function
    packages = {name: types.ModuleType(name) for name in [
        "miles", "miles.utils", "miles.utils.external_utils", "typer",
    ]}
    packages[utility.__name__] = utility
    path = Path(__file__).with_name("run_qwen3_30b_a3b_gsm8k_rubin.py")
    spec = importlib.util.spec_from_file_location("_qwen3_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    packages[spec.name] = module
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


class CheckpointLauncherTests(unittest.TestCase):
    def test_default_checkpoint_arguments_are_unchanged(self):
        args = launcher.ScriptArgs(model_dir="/models", output_dir="/output")
        self.assertEqual(shlex.split(launcher._checkpoint_args(args)), [
            "--hf-checkpoint", "/models/Qwen3-30B-A3B",
            "--ref-load", "/models/Qwen3-30B-A3B_torch_dist",
            "--megatron-to-hf-mode", "raw", "--save", "/output/checkpoints",
            "--save-interval", "50",
        ])

    def test_positive_retention_keeps_full_checkpoint_and_explicit_save_path(self):
        args = launcher.ScriptArgs(save_interval=10, save_retain_interval=1000000,
                                   save_dir="/output with spaces/checkpoints")
        argv = shlex.split(launcher._checkpoint_args(args))
        self.assertEqual(argv[argv.index("--save-retain-interval") + 1], "1000000")
        self.assertEqual(argv[argv.index("--save-interval") + 1], "10")
        self.assertEqual(argv[argv.index("--save") + 1], args.save_dir)
        self.assertNotIn("--no-save-optim", argv)
        self.assertNotIn("--load", argv)

    def test_invalid_retention_is_rejected(self):
        for values in [{"save_retain_interval": -1},
                       {"save_interval": 0, "save_retain_interval": 1}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                launcher.ScriptArgs(**values)

    def test_saving_disabled_emits_no_retention_or_save_flags(self):
        argv = shlex.split(launcher._checkpoint_args(launcher.ScriptArgs(save_interval=0)))
        self.assertNotIn("--save", argv)
        self.assertNotIn("--save-interval", argv)
        self.assertNotIn("--save-retain-interval", argv)


if __name__ == "__main__":
    unittest.main()
