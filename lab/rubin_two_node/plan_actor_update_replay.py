"""Print an actor-only replay plan; never starts Ray, engines, or GPU work.

Inputs must be mounted read-only and verified by the outer run operator. A trusted
Miles debug rollout dump is replayed exactly; this planner does not fabricate data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

try:
    from .actor_update_profile import digest_file, validate_config, ENV, RUN_ENV, MIB
except ImportError:
    from actor_update_profile import digest_file, validate_config, ENV, RUN_ENV, MIB


def source_plan(path):
    data = json.loads(Path(path).read_text())
    data = data.get("ray_job", data)
    runtime = data.get("runtime_env", {})
    if set(runtime) - {"env_vars"}:
        raise ValueError("Only recorded environment variables are supported")
    if "argv" in data:
        argv, env = data["argv"], data.get("extra_runtime_env", runtime.get("env_vars", {}))
    else:
        command = shlex.split(data["entrypoint"])
        if len(command) < 3 or Path(command[0]).name != "python3" or Path(command[1]).name != "train.py":
            raise ValueError("Require direct python3 train.py source entrypoint")
        argv, env = command[2:], runtime.get("env_vars", {})
    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("Invalid source argv")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("Invalid source environment")
    parsed = []
    for item in argv:
        if item.startswith("--"):
            flag, equal, value = item.partition("=")
            parsed.append([flag, *([value] if equal else [])])
        elif parsed:
            parsed[-1].append(item)
        else:
            raise ValueError("Expected train arguments only")
    groups = {}
    for flag, *values in parsed:
        if flag in groups and groups[flag] != values:
            raise ValueError("Conflicting duplicate flag: " + flag)
        groups[flag] = values
    return groups, dict(env)


def build_plan(options, now=None):
    now = time.time() if now is None else now
    groups, env = source_plan(options.source_plan)
    required = {"--actor-num-nodes": "1", "--actor-num-gpus-per-node": "4", "--num-gpus-per-node": "4",
                "--tensor-model-parallel-size": "1", "--pipeline-model-parallel-size": "1",
                "--context-parallel-size": "1", "--expert-model-parallel-size": "4",
                "--expert-tensor-parallel-size": "1", "--global-batch-size": "512",
                "--rollout-batch-size": "256", "--n-samples-per-prompt": "8", "--num-steps-per-rollout": "4",
                "--num-rollout": "50", "--megatron-to-hf-mode": "raw"}
    for flag, value in required.items():
        if groups.get(flag) != [value]:
            raise ValueError("Unexpected recorded recipe: " + flag)
    forbidden = {"--load", "--ckpt-step", "--ref-ckpt-step", "--start-rollout-id", "--debug-train-only",
                 "--debug-rollout-only", "--debug-disable-optimizer", "--load-debug-rollout-data",
                 "--custom-megatron-init-path", "--custom-megatron-before-train-step-hook-path",
                 "--dynamic-sampling-filter-path", "--train-backend", "--load-debug-rollout-data-subsample"}
    if forbidden.intersection(groups):
        raise ValueError("Require the unmodified fresh main source plan: " + str(forbidden.intersection(groups)))
    def scalar(flag):
        value = groups.get(flag, [])
        if len(value) != 1:
            raise ValueError("Missing explicit scalar " + flag)
        return value[0]
    prompt_data = getattr(options, "prompt_data_path", "/inputs/train.jsonl")
    if not Path(prompt_data).is_absolute():
        raise ValueError("Prompt data path must be absolute")
    prompt_override = {"old": scalar("--prompt-data"), "new": prompt_data}
    groups["--prompt-data"] = [prompt_data]
    context = int(scalar("--rollout-max-context-len"))
    changes = []
    for flag in ("--max-tokens-per-gpu", "--log-probs-max-tokens-per-gpu"):
        if flag not in groups and flag.startswith("--log-probs"):
            continue
        old = int(scalar(flag))
        if not context <= options.max_tokens_per_gpu <= old:
            raise ValueError("Common token budget must fit context and not increase the recorded budget")
        groups[flag] = [str(options.max_tokens_per_gpu)]
        changes.append({"flag": flag, "old": old, "new": options.max_tokens_per_gpu})
    for value in (options.rollout_sha256, options.train_step_source_sha256, options.initial_model_id):
        if not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("Require explicit SHA256 identities")
    if digest_file(options.rollout_path) != options.rollout_sha256:
        raise ValueError("Rollout SHA256 mismatch")
    if Path(options.output_root).exists():
        raise ValueError("Use a new diagnostic output root")
    config = {"run_id": options.run_id, "deadline_epoch": options.deadline_epoch, "target": [0, 1, 0],
              "world_size": 4, "output_root": options.output_root, "rollout_path": options.rollout_path,
              "rollout_sha256": options.rollout_sha256, "train_step_source_sha256": options.train_step_source_sha256,
              "initial_model_id": options.initial_model_id, "hf_checkpoint": scalar("--hf-checkpoint"),
              "ref_load": scalar("--ref-load"), "capture_seconds": 180, "export_seconds": 120,
              "max_trace_bytes": 512 * MIB, "max_rss_growth_bytes": 8 * 1024 * MIB}
    env[RUN_ENV] = options.run_id
    env["RUBIN_RUN_ID"] = options.run_id
    env["PYTHONPATH"] = "/opt/actor-profile" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    validate_config(config, env, now)
    env[ENV] = json.dumps(config, sort_keys=True)
    remove = {"--save", "--no-save-optim", "--no-save-rng", "--save-trigger-sentinel", "--use-pytorch-profiler",
              "--profile", "--tensorboard-dir", "--use-tensorboard", "--use-wandb", "--disable-wandb-random-suffix",
              "--skip-eval-before-train", "--debug-exit-after-rollout", "--dump-details", "--record-memory-history"}
    removed = []
    for flag in list(groups):
        if flag in remove or flag.startswith(("--save-", "--eval-", "--wandb-", "--profile-", "--memory-snapshot-")):
            removed.append({"flag": flag, "values": groups.pop(flag)})
    groups.update({"--load-debug-rollout-data": [options.rollout_path], "--debug-train-only": [],
                   "--debug-exit-after-rollout": ["1"], "--skip-eval-before-train": [],
                   "--custom-megatron-init-path": ["actor_update_profile.install"]})
    argv = [item for flag, values in groups.items() for item in [flag, *values]]
    return {"schema_version": 1, "run_id": options.run_id, "submission_id": options.run_id, "argv": argv, "runtime_env": {"env_vars": env},
            "entrypoint": shlex.join(["python3", "/opt/miles/train.py", *argv]), "config": config,
            "source_plan_sha256": digest_file(options.source_plan), "hook_sha256": digest_file(Path(__file__).with_name("actor_update_profile.py")),
            "removed": removed, "token_budget_overrides": changes, "prompt_data_path_override": prompt_override,
            "scope": {"rollouts": 1, "optimizer_updates": 4, "captured_update": 1, "profile_ranks": [0],
                      "unprofiled_comparison_updates": [2, 3], "scheduler_horizon_preserved": 50},
            "required_outer_guards": ["exact diagnostic-only Ray/container identity and immutable image/source",
                "read-only verified initial HF/release manifest matching initial_model_id",
                "trusted complete 2048-sample rollout dump: tokens/rewards/loss masks/logprobs verified",
                "original absolute lease deadline and exact-job/container guard, including model loading",
                "four rank packing fingerprints compared between platforms before paired interpretation",
                "retain trace, logs and receipts before stopping exact diagnostic container"],
            "limitations": ["No main actor rollout dump existed; report whether this is newly generated initial-policy data.",
                "One update is representative only of its recorded length/packing mix; rank 0 is not every EP rank.",
                "Synchronization and profiling perturb timings. Compare unprofiled steps 2/3 separately; preceding diagnostic barrier excludes rank0 export skew.",
                "Runtime byte/RSS limits are polled and can overshoot between observations."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-plan", "run-id", "rollout-path", "rollout-sha256", "initial-model-id", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--train-step-source-sha256", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=4096)
    parser.add_argument("--prompt-data-path", default="/inputs/train.jsonl")
    print(json.dumps(build_plan(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
