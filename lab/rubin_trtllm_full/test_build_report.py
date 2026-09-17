"""CPU checks that guard against misleading new-report conclusions."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import build_report as build


def recipe_metadata():
    return {
        "actual_recipe_available": True,
        "recipe": {"model": "Qwen3-30B-A3B", "global_batch_size": "512",
                   "max_training_tokens_per_gpu": 4096, "max_logprob_tokens_per_gpu": 4096,
                   "reward_function": "lab.rubin_two_node.gsm8k_verl_reward.reward_func",
                   "sglang_moe_runner_backend": "flashinfer_trtllm", "save_optimizer": False},
        "gpus": 4, "expected_rollouts": 50, "optimizer_steps_per_rollout": 4,
        "graph": {"decode_requested": True, "prefill_requested": False},
        "profiling_coverage_known": True, "profiled_rollouts": [],
        "input_preparation": {"data": {k: {"sha256": v} for k, v in build.DATASETS.items()}},
        "actual_ray_entrypoint_argv": shlex.split("""
            python3 /opt/miles/train.py --hf-checkpoint /models/Qwen3-30B-A3B
            --num-rollout 50 --num-steps-per-rollout 4 --rollout-batch-size 256
            --n-samples-per-prompt 8 --global-batch-size 512
            --rollout-max-prompt-len 512 --rollout-max-response-len 1024 --rollout-max-context-len 1536
            --rollout-temperature 1.0 --rollout-top-p 1.0 --rollout-top-k -1 --lr 1e-06
            --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
            --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --max-tokens-per-gpu 4096
            --rollout-num-gpus-per-engine 1 --sglang-ep-size 1 --sglang-context-length 1536
            --actor-num-nodes 1 --actor-num-gpus-per-node 4 --num-gpus-per-node 4
            --n-samples-per-eval-prompt 1 --eval-temperature 1.0 --eval-top-p 0.7 --eval-top-k -1
            --eval-max-prompt-len 512 --eval-max-response-len 1024 --eval-max-context-len 1536
            --sglang-attention-backend triton --sglang-bf16-gemm-backend torch
            --sglang-moe-runner-backend flashinfer_trtllm --sglang-dtype bfloat16
            --custom-rm-path lab.rubin_two_node.gsm8k_verl_reward.reward_func
            --bf16 --apply-chat-template --colocate --no-save-optim --sglang-disable-piecewise-cuda-graph
        """),
    }


def numerical_comparison():
    runs = []
    for platform in build.PLATFORMS:
        rows = []
        for i in range(50):
            rows.append({"rollout_id": i, "training_stage_complete": True,
                         "common": {"training_reward_mean": .5, "response_length_mean_tokens": 100,
                                    "truncated_ratio": 0},
                         "metrics": {"rollout/weight_version/min": i + 1, "rollout/weight_version/max": i + 1,
                                     "rollout/weight_version/mixed_version_ratio": 0},
                         "train_steps": [{"metrics": {"train/train_rollout_logprob_abs_diff": .001,
                                                        "train/grad_norm": 1}} for _ in range(4)],
                         "eval": [{"metrics": {"eval/gsm8k": .5}, "line": i + 1}]
                                 if i in (0, 9, 19, 29, 39, 49) else []})
        runs.append({"label": platform, "rows": rows})
    return {"runs": runs}


def comparison():
    runs = []
    for label, actor in (("rubin", 200), ("gb300", 100)):
        rows = []
        for i, seconds, rate in ((0, 1000, 900), (1, 10, 100), (2, 90, 10), (3, 800, 100)):
            rows.append({"rollout_id": i, "training_stage_complete": True, "profiled": False,
                         "unprofiled_timing_eligible": i in (1, 2),
                         "common": {"step_seconds": actor + seconds, "actor_train_seconds": actor,
                                    "rollout_seconds": seconds, "output_tokens_per_gpu_generation_second": rate,
                                    "log_probs_seconds": 3, "ref_log_probs_seconds": 5, "update_weights_seconds": 2}})
        runs.append({"label": label, "metadata": {"exclude_timing_rollouts": [0, 3]},
                     "completed_training_rollouts": [0, 1, 2, 3], "rows": rows})
    return {"runs": runs}


class ReportBoundary(unittest.TestCase):
    def test_retry_suffix_keeps_explicit_main_binding(self):
        experiment = {"schema": "qwen3-trtllm-full-experiment-v1", "experiment_id": build.EXPERIMENT,
                      "run_bindings": [{"label": "rubin", "run_id": "20260917-rubin-j2213753-trtllm-r2"},
                                       {"label": "gb300", "run_id": "20260917-gb300-j2212644-trtllm"}]}
        bindings = build.bind_experiment(experiment)
        self.assertEqual(bindings["rubin"], "20260917-rubin-j2213753-trtllm-r2")
        for old_run in ("20260917-rubin-j2213753-trtllm", "20260917-rubin-j2212643-trtllm"):
            data = {"schema": "miles-qwen3-comparison-v1", "experiment_id": build.EXPERIMENT,
                    "runs": [{"label": "rubin", "metadata": {"run_id": old_run}}]}
            with self.subTest(old_run=old_run), self.assertRaisesRegex(ValueError, "Old or unbound"):
                build.validate_main(data, None, Path("/tmp"), bindings)
        for index, suffix in ((0, "-r3"), (0, "-r2/old"), (1, "-r2")):
            changed = copy.deepcopy(experiment)
            changed["run_bindings"][index]["run_id"] = changed["run_bindings"][index]["run_id"].removesuffix("-r2") + suffix
            with self.subTest(index=index, suffix=suffix), self.assertRaisesRegex(ValueError, "unbound"):
                build.bind_experiment(changed)

    def test_recipe_accepts_observed_defaults_and_equivalent_numeric_arguments(self):
        metadata = recipe_metadata()
        self.assertTrue(build.check_recipe(metadata))
        argv = metadata["actual_ray_entrypoint_argv"]
        argv[argv.index("--lr") + 1] = "0.000001"
        argv[argv.index("--rollout-temperature") + 1] = "1"
        argv.extend(["--log-probs-max-tokens-per-gpu=4096", "--sglang-tp-size=1"])
        self.assertTrue(build.check_recipe(metadata))

    def test_actual_recipe_drift_is_rejected_despite_unchanged_metadata(self):
        changes = {"--rollout-batch-size": "128", "--n-samples-per-prompt": "16",
                   "--global-batch-size": "256", "--num-steps-per-rollout": "8",
                   "--rollout-max-prompt-len": "1024", "--rollout-max-response-len": "512",
                   "--rollout-max-context-len": "2048", "--sglang-context-length": "2048",
                   "--tensor-model-parallel-size": "2", "--expert-model-parallel-size": "2",
                   "--rollout-num-gpus-per-engine": "2", "--sglang-ep-size": "2",
                   "--actor-num-gpus-per-node": "8", "--sglang-attention-backend": "flashinfer",
                   "--sglang-bf16-gemm-backend": "cutlass", "--sglang-dtype": "float16",
                   "--rollout-temperature": "0", "--rollout-top-p": "0.7", "--rollout-top-k": "1",
                   "--eval-top-p": "1", "--eval-max-response-len": "2048", "--lr": "NaN"}
        for flag, value in changes.items():
            metadata = recipe_metadata(); argv = metadata["actual_ray_entrypoint_argv"]
            argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                build.check_recipe(metadata)

    def test_missing_duplicate_filter_and_conflicting_recipe_arguments_are_rejected(self):
        additions = [[], ["--rollout-batch-size", "256"], ["--rollout-batch-size=256"],
                     ["--dynamic-sampling-filter-path", "custom.filter"],
                     ["--dynamic-sampling-filter-path=custom.filter"], ["--fp16"],
                     ["--sglang-disable-cuda-graph"], ["--sglang-cuda-graph-backend-decode", "disabled"],
                     ["--sglang-cuda-graph-backend-prefill", "full"],
                     ["--log-probs-max-tokens-per-gpu", "2048"], ["--sglang-tp-size", "2"],
                     ["--sglang-dp-size", "2"]]
        for extra in additions:
            metadata = recipe_metadata(); argv = metadata["actual_ray_entrypoint_argv"]
            if extra:
                argv.extend(extra)
            else:
                argv.remove("--bf16")
            with self.subTest(extra=extra), self.assertRaises(ValueError): build.check_recipe(metadata)
        metadata = recipe_metadata(); del metadata["actual_ray_entrypoint_argv"]
        with self.assertRaisesRegex(ValueError, "entrypoint argv"): build.check_recipe(metadata)

    def test_final_rejects_stale_missing_or_mixed_rollout_weight_versions(self):
        good = numerical_comparison()
        self.assertTrue(all(n["all_rollout_versions_clean"] for n in build.numerical_summary(good).values()))
        for changed in ({"rollout/weight_version/min": 1, "rollout/weight_version/max": 1},
                        {"rollout/weight_version/mixed_version_ratio": .25},
                        {"rollout/weight_version/min": None}):
            data = copy.deepcopy(good); data["runs"][0]["rows"][20]["metrics"].update(changed)
            self.assertFalse(build.numerical_summary(data)["rubin"]["all_rollout_versions_clean"])
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=build.ROOT) as output, \
                    tempfile.TemporaryDirectory() as inputs, \
                    patch.object(build, "validate_main", return_value={"complete": True}), \
                    patch.object(build, "profiles", return_value={"complete": True, "matched": True}), \
                    patch.object(build, "performance", return_value={"paired": {"status": "available"}}):
                (Path(inputs) / "comparison.json").write_text(json.dumps(data))
                args = argparse.Namespace(experiment=build.ROOT / "experiment.json", inputs=Path(inputs),
                                          output=Path(output), profile_rubin=None, profile_gb300=None, final=True)
                with self.assertRaisesRegex(ValueError, "clean advancing rollout weight versions"):
                    build.build(args)
                self.assertFalse((Path(output) / "index.html").exists())

    def test_old_comparison_cannot_be_new_results(self):
        with self.assertRaisesRegex(ValueError, "Only the new"):
            build.validate_main({"schema": "miles-qwen3-comparison-v1", "experiment_id": "qwen3-cudagraph-nightly-v1"},
                                None, Path("/tmp"), {})

    def test_final_rejects_no_evidence(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as output, tempfile.TemporaryDirectory() as inputs:
            args = argparse.Namespace(experiment=build.ROOT / "experiment.json", inputs=Path(inputs),
                                      output=Path(output), profile_rubin=None, profile_gb300=None, final=True)
            with self.assertRaisesRegex(ValueError, "Final report requires"):
                build.build(args)
            self.assertFalse((Path(output) / "index.html").exists())

    def test_timing_exclusions_and_weighted_throughput(self):
        value = build.performance(comparison())
        self.assertEqual(value["paired"]["rollout_ids"], [1, 2])
        self.assertEqual(value["weighted_output_tokens_per_gpu_generation_second"]["rubin"], 19)

    def test_actor_regression_remains_visible(self):
        value = build.performance(comparison())
        self.assertEqual(value["ratios"]["actor_train"]["gb300_over_rubin"], .5)
        self.assertEqual(value["ratios"]["actor_train"]["rubin_time_reduction"], -1)
        self.assertLess(value["ratios"]["step"]["rubin_time_reduction"], 0)

    def test_missing_metric_does_not_shorten_one_platform_cohort(self):
        data = comparison()
        data["runs"][0]["rows"][2]["common"]["actor_train_seconds"] = None
        value = build.performance(data)
        self.assertIsNone(value["ratios"]["actor_train"])
        self.assertEqual(value["paired"]["statistics"]["rubin"]["actor_train"]["missing_rollout_ids"], [2])

    def test_changed_profile_attachment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "trace.png"
            p.write_bytes(b"altered image")
            with self.assertRaisesRegex(ValueError, "attachment identity/hash"):
                build.copy_checked(str(p), "0" * 64, root, "copied.png")
            self.assertFalse((root / "evidence").exists())


if __name__ == "__main__":
    unittest.main()
