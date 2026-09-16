"""CPU-only contract tests; never import Torch, start Ray, or launch a process."""

import importlib.util
import json
import shlex
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch


@dataclass
class _Config:
    cuda_core_dump: bool = False
    num_nodes: int = 1
    extra_env_vars: str = ""
    output_dir: str = "/unused"


def _load_launcher():
    utility = types.ModuleType("miles.utils.external_utils.command_utils")
    utility.ExecuteTrainConfig = _Config
    utility.create_run_id = lambda: "cpu-test"
    utility.dataclass_cli = lambda function: function
    utility.repo_base_dir = Path("/source/miles")
    utility.shell_safe_model_args = lambda _: "--num-layers 48 --kv-channels 128"
    utility.exec_command_cpu = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("Shell execution forbidden"))

    def execute_train(**kwargs):
        utility.exec_command_cpu("pkill -9 sglang; pkill -9 miles; true;")
        runtime = {"env_vars": kwargs["extra_env_vars"]}
        command = (
            "export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && ray job submit "
            f"--runtime-env-json={shlex.quote(json.dumps(runtime))} -- python3 /source/miles/train.py "
            f"{utility.shell_safe_model_args(None)} {kwargs['train_args']}"
        )
        utility.exec_command_cpu(command)

    utility.execute_train = execute_train
    packages = {name: types.ModuleType(name) for name in ["miles", "miles.utils", "miles.utils.external_utils"]}
    packages[utility.__name__] = utility
    packages["typer"] = types.ModuleType("typer")
    path = Path(__file__).with_name("profile_qwen3_replay.py")
    spec = importlib.util.spec_from_file_location("_profile_replay_under_test", path)
    module = importlib.util.module_from_spec(spec)
    packages[spec.name] = module
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


P = _load_launcher()


def _source_argv():
    return shlex.split(
        "--num-layers 48 --kv-channels 128 --actor-num-nodes 1 --actor-num-gpus-per-node 4 "
        "--num-gpus-per-node 4 --rollout-num-gpus-per-engine 1 --tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 --rollout-batch-size 256 --n-samples-per-prompt 8 "
        "--global-batch-size 512 --num-steps-per-rollout 4 --ref-load '/reference with spaces' "
        "--max-tokens-per-gpu 8192 --rollout-max-context-len 1536 "
        "--hf-checkpoint /hf --prompt-data /data/train.jsonl --num-rollout 50 "
        "--save /original/checkpoints --save-interval 50 --save-trigger-sentinel /original/stop "
        "--eval-prompt-data gsm8k /original/test.jsonl --eval-interval 10 "
        "--use-wandb --wandb-key private-key --save-debug-event-data /original/events "
        "--rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --lr 1e-6 "
        "--use-kl-loss --kl-loss-coef .001 --custom-rm-path original.reward "
        "--no-load-optim --no-load-rng --finetune"
    )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        checkpoint = self.root / "checkpoint"
        directory = checkpoint / "iter_0000049"
        directory.mkdir(parents=True)
        (directory / "common.pt").write_bytes(b"optimizer checkpoint metadata")
        (directory / ".metadata").write_bytes(b"distributed index")
        (directory / "__0_0.distcp").write_bytes(b"test shard")
        (checkpoint / "rollout").mkdir()
        (checkpoint / "rollout/global_dataset_state_dict_49.pt").write_bytes(b"dataset cursor")
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("49\n")
        source = self.root / "source.json"
        source.write_text(json.dumps({
            "argv": _source_argv(),
            "extra_runtime_env": {"NCCL_CUMEM_ENABLE": "1", "RUBIN_RUN_ID": "old", "WANDB_API_KEY": "secret"},
        }))
        self.args = P.ScriptArgs(
            source_plan=str(source), checkpoint_root=str(checkpoint), checkpoint_iteration=49,
            output_dir=str(self.root / "profile"), run_id="isolated-test",
        )

    def _write_source_argv(self, argv):
        path = Path(self.args.source_plan)
        source = json.loads(path.read_text())
        source["argv"] = argv
        path.write_text(json.dumps(source))

    def test_default_preserves_recorded_training_and_logprob_budgets(self):
        self._write_source_argv(_source_argv() + ["--log-probs-max-tokens-per-gpu", "6144"])
        plan = P._build_plan(self.args)
        groups = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        self.assertEqual(P._one_value(groups, "--max-tokens-per-gpu"), "8192")
        self.assertEqual(P._one_value(groups, "--log-probs-max-tokens-per-gpu"), "6144")
        self.assertFalse(plan["profile_token_budget_override"]["enabled"])
        self.assertEqual(plan["profile_token_budget_override"]["overrides"], [])

    def test_explicit_profile_budget_changes_only_two_recorded_flags(self):
        self._write_source_argv(_source_argv() + ["--log-probs-max-tokens-per-gpu", "6144"])
        baseline = P._build_plan(self.args)
        plan = P._build_plan(replace(self.args, profile_max_tokens_per_gpu=4096))
        before = P._flag_groups(shlex.split(baseline["entrypoint"])[2:])
        after = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        self.assertEqual(len(before), len(after))
        changes = [(old, new) for old, new in zip(before, after) if old != new]
        self.assertEqual(changes, [(["--max-tokens-per-gpu", "8192"], ["--max-tokens-per-gpu", "4096"]),
                                  (["--log-probs-max-tokens-per-gpu", "6144"],
                                   ["--log-probs-max-tokens-per-gpu", "4096"])])
        self.assertEqual(plan["checkpoint"], baseline["checkpoint"])
        self.assertEqual(plan["runtime_env"], baseline["runtime_env"])
        self.assertEqual(plan["planned_rollouts"], 2)
        self.assertEqual(plan["planned_optimizer_steps"], 8)
        self.assertEqual(P._one_value(after, "--global-batch-size"), "512")
        self.assertEqual(P._one_value(after, "--num-steps-per-rollout"), "4")
        self.assertNotEqual(plan["entrypoint_sha256"], baseline["entrypoint_sha256"])
        override = plan["profile_token_budget_override"]
        self.assertTrue(override["enabled"])
        self.assertEqual(override["recorded_context_limit_tokens"], 1536)
        self.assertEqual([(x["old"], x["new"]) for x in override["overrides"]], [(8192, 4096), (6144, 4096)])
        self.assertIn("profile-only", override["reason"])
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_explicit_profile_budget_does_not_invent_logprob_flag(self):
        plan = P._build_plan(replace(self.args, profile_max_tokens_per_gpu=4096))
        groups = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        self.assertEqual(P._one_value(groups, "--max-tokens-per-gpu"), "4096")
        self.assertNotIn("--log-probs-max-tokens-per-gpu", {g[0] for g in groups})
        original = _source_argv()
        original[original.index("--max-tokens-per-gpu") + 1] = "4096"
        self._write_source_argv(original)
        unchanged = P._build_plan(replace(self.args, profile_max_tokens_per_gpu=4096))
        self.assertEqual(unchanged["profile_token_budget_override"]["overrides"],
                         [{"flag": "--max-tokens-per-gpu", "old": 4096, "new": 4096, "changed": False}])

    def test_profile_budget_rejects_unsafe_or_unrecorded_limits(self):
        for value, message in [(1024, "context limit"), (8193, "must not increase")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                P._build_plan(replace(self.args, profile_max_tokens_per_gpu=value))
        with self.assertRaisesRegex(ValueError, "zero or positive"):
            replace(self.args, profile_max_tokens_per_gpu=-1)
        original = _source_argv()
        i = original.index("--max-tokens-per-gpu")
        del original[i:i+2]
        self._write_source_argv(original)
        with self.assertRaisesRegex(ValueError, "scalar --max-tokens-per-gpu"):
            P._build_plan(replace(self.args, profile_max_tokens_per_gpu=4096))
        self._write_source_argv(_source_argv() + ["--log-probs-max-tokens-per-gpu", "3072"])
        with self.assertRaisesRegex(ValueError, "must not increase recorded --log-probs"):
            P._build_plan(replace(self.args, profile_max_tokens_per_gpu=4096))

    def test_resume_retains_recipe_and_isolates_outputs(self):
        plan = P._build_plan(self.args)
        groups = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        for flag, value in {
            "--load": str(Path(self.args.checkpoint_root).resolve()), "--start-rollout-id": "50",
            "--num-rollout": "52", "--debug-exit-after-rollout": "2", "--ref-load": "/reference with spaces",
            "--rollout-top-k": "-1", "--rollout-temperature": "1", "--lr": "1e-6",
            "--custom-rm-path": "original.reward", "--profile-step-start": "1", "--profile-step-end": "2",
        }.items():
            self.assertEqual(P._one_value(groups, flag), value)
        flags = {g[0] for g in groups}
        self.assertNotIn("--ckpt-step", flags)
        self.assertNotIn("--ref-ckpt-step", flags)
        self.assertIn("--use-checkpoint-opt-param-scheduler", flags)
        self.assertFalse(flags & {"--save", "--eval-interval", "--finetune", "--no-load-optim", "--no-load-rng"})
        self.assertFalse(any(f.startswith(("--save-", "--eval-", "--wandb-")) for f in flags))
        self.assertEqual(plan["planned_optimizer_steps"], 8)
        self.assertEqual(plan["profiled_rollouts"], [50, 51])
        self.assertEqual(plan["runtime_env"]["env_vars"][P.RUN_ID_ENV], "isolated-test")
        self.assertNotIn("private-key", json.dumps(plan))
        self.assertNotIn("WANDB_API_KEY", plan["runtime_env"]["env_vars"])
        self.assertFalse(Path(self.args.output_dir).exists())
        with self.assertRaises(AssertionError):
            P.U.exec_command_cpu("never execute this")

    def test_frozen_checkpoint9_replays_exactly_two_rollouts_starting10(self):
        root = Path(self.args.checkpoint_root)
        (root / "iter_0000049").rename(root / "iter_0000009")
        (root / "rollout/global_dataset_state_dict_49.pt").rename(root / "rollout/global_dataset_state_dict_9.pt")
        (root / "latest_checkpointed_iteration.txt").write_text("9\n")
        plan = P._build_plan(replace(self.args, checkpoint_iteration=9, profile_max_tokens_per_gpu=4096))
        groups = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        self.assertEqual(P._one_value(groups, "--start-rollout-id"), "10")
        self.assertEqual(P._one_value(groups, "--num-rollout"), "12")
        self.assertEqual(P._one_value(groups, "--debug-exit-after-rollout"), "2")
        self.assertEqual(plan["planned_optimizer_steps"], 8)
        self.assertEqual(plan["profiled_rollouts"], [10, 11])
        self.assertEqual(plan["checkpoint"]["iteration"], 9)

    def test_two_rollouts_reach_verified_profiler_export_boundary(self):
        # Source contract from installed Miles profile_utils.py + actor.py:
        # exactly one prof.step per train call, not per optimizer update.
        # Installed torch profiler schedule reaches RECORD_AND_SAVE -> NONE
        # on that second call; its transition synchronously invokes the handler.
        plan = P._build_plan(self.args)
        groups = P._flag_groups(shlex.split(plan["entrypoint"])[2:])
        start, end = [int(P._one_value(groups, "--profile-step-" + name)) for name in ("start", "end")]
        expected = {"wait": max(start - 1, 0), "warmup": 1 if start > 0 else 0,
                    "active": end - start, "repeat": 1}
        schedule = plan["train_profiler_schedule"]
        self.assertEqual({k: schedule[k] for k in expected}, expected)
        cycle = expected["wait"] + expected["warmup"] + expected["active"]
        self.assertEqual(cycle, 2)
        # Independent state sequence for the verified wait0/warmup1/active1 cycle.
        states = ["WARMUP", "RECORD_AND_SAVE", "NONE"]
        count = int(P._one_value(groups, "--debug-exit-after-rollout"))
        self.assertEqual(states[count - 1:count + 1], ["RECORD_AND_SAVE", "NONE"])
        self.assertNotEqual(states[:2], ["RECORD_AND_SAVE", "NONE"])
        first = int(P._one_value(groups, "--start-rollout-id"))
        stop = int(P._one_value(groups, "--num-rollout"))
        self.assertEqual(stop - first, count)
        self.assertEqual(count, plan["planned_rollouts"])
        self.assertEqual(plan["planned_optimizer_steps"], 4 * count)
        self.assertEqual(schedule["trace_ready_after_replay_rollout"], count)
        self.assertIn("before CPU actor backup", schedule["export_boundary"])
        self.assertEqual((plan["max_runtime_seconds"], plan["max_trace_bytes"]), (4500, 10 * 1024**3))

    def test_checkpoint_identity_detects_change_and_missing_cursor(self):
        first = P._checkpoint_manifest(self.args.checkpoint_root, 49)
        self.args.expected_checkpoint_id = first["id"]
        P._build_plan(self.args)
        (Path(self.args.checkpoint_root) / "iter_0000049/common.pt").write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            P._build_plan(self.args)
        (Path(self.args.checkpoint_root) / "rollout/global_dataset_state_dict_49.pt").unlink()
        with self.assertRaisesRegex(ValueError, "metadata"):
            P._checkpoint_manifest(self.args.checkpoint_root, 49)

    def test_reference_step_is_preserved_without_inheriting_actor_step(self):
        original = _source_argv() + ["--ckpt-step", "49", "--ref-ckpt-step", "5"]
        argv, _ = P._replay_arguments(original, self.args)
        groups = P._flag_groups(argv)
        self.assertNotIn("--ckpt-step", {g[0] for g in groups})
        self.assertEqual(P._one_value(groups, "--ref-ckpt-step"), "5")
        self.assertEqual(P._one_value(groups, "--ref-load"), "/reference with spaces")

    def test_checkpoint_recheck_rejects_advanced_tracker_and_changed_metadata(self):
        plan = P._build_plan(self.args)
        P._verify_checkpoint_unchanged(plan)
        tracker = Path(self.args.checkpoint_root) / "latest_checkpointed_iteration.txt"
        tracker.write_text("50\n")
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            P._verify_checkpoint_unchanged(plan)
        tracker.write_text("49\n")
        (Path(self.args.checkpoint_root) / "iter_0000049/common.pt").write_bytes(b"changed after planning")
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            P._verify_checkpoint_unchanged(plan)

    def test_rejects_model_topology_or_submission_contract_drift(self):
        argv = _source_argv()
        argv[argv.index("--global-batch-size") + 1] = "256"
        with self.assertRaisesRegex(ValueError, "512"):
            P._replay_arguments(argv, self.args)
        with self.assertRaises(ValueError):
            P._parse_submission(["ray job submit -- python3 other.py"], [])
        with self.assertRaisesRegex(ValueError, "command contract"):
            P._parse_submission(["unexpected", "extra", "commands"], [])

    def test_accepts_ray_record_but_rejects_shell_entrypoint(self):
        path = self.root / "ray.json"
        path.write_text(json.dumps({"entrypoint": shlex.join(["python3", "/source/miles/train.py", *_source_argv()]),
                                    "runtime_env": {"env_vars": {"NCCL_CUMEM_ENABLE": "1"}}}))
        self.assertEqual(P._load_source_plan(path)[0], _source_argv())
        path.write_text(json.dumps({"entrypoint": "bash -c 'python3 train.py'", "runtime_env": {"env_vars": {}}}))
        with self.assertRaisesRegex(ValueError, "direct python3"):
            P._load_source_plan(path)

    def _guard_plan_job(self):
        plan = P._build_plan(self.args)
        plan.update(armed_timestamp=0, deadline_timestamp=10, max_trace_bytes=10)
        job = {"type": "SUBMISSION", "submission_id": plan["submission_id"], "status": "RUNNING",
               "entrypoint": plan["entrypoint"], "runtime_env": plan["runtime_env"]}
        return plan, job

    def test_guard_requires_all_three_identity_checks(self):
        plan, job = self._guard_plan_job()
        self.assertEqual(P._guard_action(plan, job, 10, 1), "trace_byte_limit")
        self.assertEqual(P._guard_action(plan, job, 0, 11), "runtime_limit")
        self.assertEqual(P._guard_action(plan, job, 0, 1), "monitor")
        for altered in [
            {**job, "submission_id": "another-job"}, {**job, "entrypoint": "python3 another.py"},
            {**job, "runtime_env": {"env_vars": {P.RUN_ID_ENV: "other-run"}}},
        ]:
            with self.assertRaises(P._IdentityMismatch):
                P._guard_action(plan, altered, 1000, 1000)

    def test_guard_retries_invalid_json_and_stops_only_exact_submission(self):
        plan, job = self._guard_plan_job()

        class API:
            def __init__(self):
                self.reads, self.stops = 0, []

            def list_jobs(self):
                return []

            def get_job(self, _):
                self.reads += 1
                if self.reads == 1:
                    raise ValueError("Temporary invalid JSON")
                return {**job, "status": "STOPPED" if self.stops else "RUNNING"}

            def stop_job(self, submission_id):
                self.stops.append(submission_id)
                return True

        api = API()
        state_path = self.root / "guard.json"
        with patch.object(P.time, "sleep"), patch.object(P, "_trace_bytes", return_value=100):
            P._watch(plan, api, state_path)
        state = json.loads(state_path.read_text())
        self.assertEqual(api.stops, [plan["submission_id"]])
        self.assertEqual(state["api_errors"], 1)
        self.assertEqual(state["terminal_status"], "STOPPED")
        self.assertEqual(state["stop_reason"], "trace_byte_limit")

    def test_busy_cluster_and_trace_symlink_are_rejected(self):
        api = types.SimpleNamespace(list_jobs=lambda: [{"type": "SUBMISSION", "status": "RUNNING",
                                                        "submission_id": "main-learning-run"}])
        with self.assertRaisesRegex(ValueError, "overlap"):
            P._assert_idle(api, "profile-job")
        traces = self.root / "traces"
        traces.mkdir()
        (traces / "trace.gz").write_bytes(b"123456")
        self.assertEqual(P._trace_bytes(traces), 6)
        (traces / "outside").symlink_to(self.root / "source.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            P._trace_bytes(traces)

    def test_trace_scan_io_failure_is_not_silently_counted_as_zero(self):
        traces = self.root / "traces"
        traces.mkdir()
        with patch.object(P.os, "scandir", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                P._trace_bytes(traces)
        plan, job = self._guard_plan_job()
        stopped = []
        api = types.SimpleNamespace(
            list_jobs=lambda: [],
            get_job=lambda _: {**job, "status": "STOPPED" if stopped else "RUNNING"},
            stop_job=lambda submission_id: stopped.append(submission_id) or True,
        )
        state_path = self.root / "scan-error.json"
        with patch.object(P.time, "sleep"), patch.object(P, "_trace_bytes", side_effect=OSError("I/O error")):
            P._watch(plan, api, state_path)
        self.assertEqual(stopped, [plan["submission_id"]])
        self.assertEqual(json.loads(state_path.read_text())["stop_reason"], "trace_scan_error")


if __name__ == "__main__":
    unittest.main()
