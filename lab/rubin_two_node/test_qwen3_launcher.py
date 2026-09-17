"""CPU-only launcher contracts; no Torch, Ray, shell command, or GPU is used."""

import importlib.util
import shlex
import sys
import types
import unittest
from dataclasses import dataclass, replace
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


class CudaGraphLauncherTests(unittest.TestCase):
    def test_default_remains_eager_for_decode_and_prefill(self):
        args = launcher.ScriptArgs()
        argv = shlex.split(launcher._build_train_args(args))
        self.assertFalse(args.sglang_enable_cuda_graph)
        self.assertEqual(argv.count("--sglang-disable-cuda-graph"), 1)
        self.assertEqual(argv.count("--sglang-disable-piecewise-cuda-graph"), 1)

    def test_decode_graph_opt_in_changes_only_its_disable_flag(self):
        for options in ({}, {"max_tokens_per_gpu": 4096, "save_interval": 10,
                             "save_retain_interval": 1000000, "enable_eval": False}):
            with self.subTest(options=options):
                eager = launcher.ScriptArgs(**options)
                enabled = replace(eager, sglang_enable_cuda_graph=True)
                before = shlex.split(launcher._build_train_args(eager))
                after = shlex.split(launcher._build_train_args(enabled))
                self.assertEqual(after, [x for x in before if x != "--sglang-disable-cuda-graph"])
                self.assertIn("--sglang-disable-piecewise-cuda-graph", after)
                self.assertNotIn("--sglang-enforce-piecewise-cuda-graph", after)
                self.assertNotIn("--sglang-enable-torch-compile", after)


class MoeBackendLauncherTests(unittest.TestCase):
    def test_backend_changes_only_its_value_and_rejects_unsupported_values(self):
        args = launcher.ScriptArgs(num_rollout=2, max_tokens_per_gpu=4096,
                                   save_interval=0, sglang_enable_cuda_graph=True)
        before = shlex.split(launcher._build_train_args(args))
        self.assertEqual(before.count("--sglang-moe-runner-backend"), 1)
        backend_index = before.index("--sglang-moe-runner-backend") + 1
        self.assertEqual(before[backend_index], "triton")
        self.assertTrue(args.enable_eval)
        for backend in ("triton", "flashinfer_cutlass", "flashinfer_trtllm"):
            with self.subTest(backend=backend):
                selected = replace(args, sglang_moe_runner_backend=backend)
                after = shlex.split(launcher._build_train_args(selected))
                expected = before.copy()
                expected[backend_index] = backend
                self.assertEqual(after, expected)
        for backend in ("", "auto", "flashinfer_cutedsl", "triton --sglang-disable-cuda-graph"):
            with self.subTest(unsupported=backend), self.assertRaisesRegex(ValueError, "MoE backend"):
                replace(args, sglang_moe_runner_backend=backend)


if __name__ == "__main__":
    unittest.main()
