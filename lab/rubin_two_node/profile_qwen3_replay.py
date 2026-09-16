"""Plan or submit a bounded Qwen3 profile replay on an idle external Ray cluster.

Requires the original launcher's print-only JSON (argv plus extra_runtime_env),
or a Ray job JSON containing entrypoint and runtime_env. Use the same explicit
full checkpoint and metadata fingerprint on both hardware platforms. Checkpoint
files are read only; mount their directory read-only as an additional safeguard.
Three rollouts retain four optimizer updates each. Training profiler counters are
relative to this new actor: the first rollout warms up, the second is recorded.
The recording also includes the first rollout's tail/offload/weight sync.

Args:
  --source-plan: Original resolved-launcher or Ray-job JSON; no shell is evaluated.
  --checkpoint-root / --checkpoint-iteration: Explicit training checkpoint, not
      an automatically selected latest directory. Its tracker must equal the
      requested iteration and remain unchanged; use only a completed run with no
      checkpoint writer. Optimizer/RNG loading stays on. No global --ckpt-step is
      injected, so reference loading retains its own release/numeric tracker.
  --expected-checkpoint-id: Fingerprint printed by the planning invocation; needed
      for execution. It hashes metadata and shard sizes, not full tensor contents.
  --output-dir: New, separate output directory; execution rejects an existing path.
  --ray-address: Dashboard HTTP(S) origin for the already running external cluster.
  --execute-run: Explicitly submit. Default, or --print-only, only prints the plan.
  --max-trace-gib: Default 10; a detached guard stops only this submission on excess.
  --max-runtime-seconds: Default 4500, counted from guard arming, including loading.
  --profile-max-tokens-per-gpu: Default 0 preserves recorded microbatch budgets.
      A positive value explicitly lowers only the training token budget and an
      explicitly recorded log-prob token budget. It must fit the recorded rollout
      context limit and cannot increase either budget. Every override is recorded
      in the plan; it does not modify the main run or its global batch size.
  --guard-plan: Internal detached guard mode; do not use for other Ray submissions.

Example inside a prepared container (planning creates no files or Ray jobs):
  MILES_SCRIPT_EXTERNAL_RAY=1 python3 lab/rubin_two_node/profile_qwen3_replay.py \
      --source-plan /inputs/original-ray-job.json --checkpoint-root /checkpoints \
      --checkpoint-iteration 49 --output-dir /run-output/profile-rubin \
      --ray-address http://127.0.0.1:28265 --print-only
Repeat with --expected-checkpoint-id <printed-id> --execute-run to submit.
Submission success is not workload success: inspect profile-guard.json afterward.

U.execute_train currently kills broad process names even with external Ray. This
script records its commands with shell execution disabled, strictly validates the
single generated submission, and submits it using Ray Jobs SDK with a fixed ID.
No preamble, ray start, shell command, or broad process kill is executed. The
detached guard uses exact submission ID, entrypoint hash and runtime run ID.
Limits are polled, so bytes may overshoot; Ray API outages can delay stopping.
SGLang capture is manual: the printed bounded payload targets one actual TP1
engine during this independent replay, never the main learning job.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError

import typer

import miles.utils.external_utils.command_utils as U
from lab.rubin_two_node.watch_qwen3_run import JobsAPI, TERMINAL, persist, utc


MODEL_TYPE = "qwen3-30B-A3B"
RUN_ID_ENV = "MILES_PROFILE_RUN_ID"
_REMOVED_FLAGS = {
    "--load", "--ckpt-step", "--start-rollout-id", "--num-rollout", "--debug-exit-after-rollout",
    "--no-load-optim", "--no-load-rng", "--finetune", "--override-opt-param-scheduler",
    "--use-checkpoint-opt-param-scheduler", "--profile", "--use-pytorch-profiler", "--profile-target",
    "--profile-step-start", "--profile-step-end", "--profile-ranks", "--record-memory-history",
    "--tensorboard-dir", "--use-tensorboard", "--use-wandb", "--disable-wandb-random-suffix",
    "--skip-eval-before-train", "--save", "--no-save-optim", "--no-save-rng",
}


class _IdentityMismatch(ValueError):
    pass


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 1
    source_plan: str = ""
    checkpoint_root: str = ""
    checkpoint_iteration: int = -1
    expected_checkpoint_id: str = ""
    ray_address: str = "http://127.0.0.1:8265"
    megatron_path: str = "/opt/Megatron-LM"
    max_trace_gib: float = 10.0
    max_runtime_seconds: int = 4500
    profile_max_tokens_per_gpu: int = 0
    poll_seconds: int = 5
    submission_grace_seconds: int = 120
    execute_run: bool = False
    print_only: bool = False
    guard_plan: str = ""

    def __post_init__(self):
        if self.guard_plan:
            return
        if not self.source_plan or not self.checkpoint_root or self.checkpoint_iteration < 0:
            raise ValueError("Specify source-plan, checkpoint-root and a numeric checkpoint-iteration")
        if self.num_nodes != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.run_id):
            raise ValueError("Use one node and a simple unique run ID of at most 80 characters")
        if self.max_trace_gib <= 0 or self.max_runtime_seconds <= 0:
            raise ValueError("Trace and elapsed-time limits must be positive")
        if self.profile_max_tokens_per_gpu < 0:
            raise ValueError("profile-max-tokens-per-gpu must be zero or positive")
        if not 1 <= self.poll_seconds <= 30 or self.submission_grace_seconds < self.poll_seconds:
            raise ValueError("Use a 1-30 second poll and a longer submission grace period")
        if self.execute_run and self.print_only:
            raise ValueError("Choose --execute-run or --print-only")
        if self.extra_env_vars or self.cuda_core_dump:
            raise ValueError("Replay uses the recorded environment; extra overrides/core dumps are not supported")

    @property
    def submission_id(self):
        return f"qwen3-profile-{self.run_id}"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _load_source_plan(path):
    data = json.loads(Path(path).read_text())
    runtime = data.get("runtime_env")
    if runtime is not None and (not isinstance(runtime, dict) or set(runtime) - {"env_vars"}):
        raise ValueError("Only the recorded env_vars runtime environment is supported")
    if "argv" in data:
        argv = data["argv"]
        env = runtime.get("env_vars", {}) if runtime is not None else data.get("extra_runtime_env")
    else:
        command = shlex.split(data.get("entrypoint", ""))
        if len(command) < 3 or Path(command[0]).name != "python3" or Path(command[1]).name != "train.py":
            raise ValueError("Expected a direct python3 train.py entrypoint")
        argv, env = command[2:], (runtime or {}).get("env_vars")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError("Source argv must be a nonempty string list")
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("Source must include the recorded string-valued environment")
    return argv, env


def _flag_groups(argv):
    groups = []
    for token in argv:
        if token.startswith("--"):
            if "=" in token:
                flag, value = token.split("=", 1)
                groups.append([flag, value])
            else:
                groups.append([token])
        elif groups:
            groups[-1].append(token)
        else:
            raise ValueError("Expected train flags, not an executable or shell prefix")
    return groups


def _one_value(groups, name):
    matches = [g[1:] for g in groups if g[0] == name]
    if len(matches) != 1 or len(matches[0]) != 1:
        raise ValueError(f"Expected exactly one scalar {name}")
    return matches[0][0]


def _profile_token_budget(argv, args):
    requested = args.profile_max_tokens_per_gpu
    record = {"enabled": bool(requested), "requested_tokens_per_gpu": requested, "overrides": [],
              "reason": "Default preserves the recorded token budgets."}
    if requested == 0:
        return argv, record
    if requested < 0:
        raise ValueError("profile-max-tokens-per-gpu must be zero or positive")
    groups = _flag_groups(argv)

    def positive_value(flag):
        value = _one_value(groups, flag)
        if not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"Expected a recorded positive integer {flag}")
        return int(value)

    context = positive_value("--rollout-max-context-len")
    if requested < context:
        raise ValueError("Profile token budget must be at least the recorded rollout context limit")
    flags = ["--max-tokens-per-gpu"]
    if any(group[0] == "--log-probs-max-tokens-per-gpu" for group in groups):
        flags.append("--log-probs-max-tokens-per-gpu")
    for flag in flags:
        old = positive_value(flag)
        if requested > old:
            raise ValueError(f"Profile token budget must not increase recorded {flag} {old}")
        record["overrides"].append({"flag": flag, "old": old, "new": requested, "changed": old != requested})
    for group in groups:
        if group[0] in flags:
            group[1] = str(requested)
    record.update(recorded_context_limit_tokens=context,
                  reason="Explicit profile-only common memory-fit microbatch budget; global batch, "
                         "optimizer updates and all other recorded recipe settings are unchanged.")
    return [token for group in groups for token in group], record


def _replay_arguments(argv, args):
    groups = _flag_groups(argv)
    required = {
        "--actor-num-nodes": "1", "--actor-num-gpus-per-node": "4", "--num-gpus-per-node": "4",
        "--rollout-num-gpus-per-engine": "1", "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "1", "--context-parallel-size": "1",
        "--expert-model-parallel-size": "4", "--expert-tensor-parallel-size": "1",
        "--rollout-batch-size": "256", "--n-samples-per-prompt": "8", "--global-batch-size": "512",
        "--num-steps-per-rollout": "4",
    }
    for flag, value in required.items():
        if _one_value(groups, flag) != value:
            raise ValueError(f"This replay expects {flag} {value}; do not silently change the learning recipe")
    for flag in ("--ref-load", "--hf-checkpoint", "--prompt-data"):
        _one_value(groups, flag)
    forbidden = {
        "--debug-disable-optimizer", "--debug-train-only", "--debug-rollout-only",
        "--load-debug-rollout-data", "--dynamic-sampling-filter-path", "--train-backend",
    }
    if any(g[0] in forbidden for g in groups):
        raise ValueError("Expected the real, unfiltered Megatron rollout/training recipe")
    removed, kept = [], []
    for group in groups:
        flag = group[0]
        if flag in _REMOVED_FLAGS or flag.startswith(("--save-", "--eval-", "--wandb-", "--memory-snapshot-")):
            removed.append(flag)
        else:
            kept.extend(group)
    kept.extend([
        "--load", str(Path(args.checkpoint_root).resolve()),
        "--start-rollout-id", str(args.checkpoint_iteration + 1),
        "--num-rollout", str(args.checkpoint_iteration + 4), "--debug-exit-after-rollout", "3",
        "--use-checkpoint-opt-param-scheduler", "--use-pytorch-profiler", "--profile-target", "train_overall",
        "--profile-step-start", "1", "--profile-step-end", "2", "--tensorboard-dir",
        str(Path(args.output_dir).resolve() / "traces/train"),
    ])
    return kept, sorted(set(removed))


def _checkpoint_manifest(root, iteration):
    root = Path(root).resolve()
    directory = root / f"iter_{iteration:07d}"
    tracker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
    if not tracker.isdecimal() or int(tracker) != iteration:
        raise ValueError("The checkpoint tracker must exactly equal the requested completed iteration")
    metadata = [directory / "common.pt", root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"]
    indexes = [p for p in (directory / ".metadata", directory / "metadata.json") if p.is_file()]
    if not indexes:
        raise ValueError("The explicit torch_dist checkpoint has no distributed metadata index")
    metadata.extend(indexes)
    records = []
    for path in metadata:
        if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024**2:
            raise ValueError(f"Missing/empty/oversized checkpoint metadata: {path}")
        records.append({"path": str(path.relative_to(root)), "sha256": _sha256(path.read_bytes())})
    shards = sorted((str(p.relative_to(root)), p.stat().st_size) for p in directory.rglob("*.distcp"))
    if not shards or any(size <= 0 for _, size in shards):
        raise ValueError("Expected nonempty torch_dist checkpoint shards")
    identity = {"iteration": iteration, "metadata": records, "shards": shards}
    return {"root": str(root), "id": _sha256(json.dumps(identity, sort_keys=True).encode()), **identity}


def _verify_checkpoint_unchanged(plan):
    expected = plan["checkpoint"]
    current = _checkpoint_manifest(expected["root"], expected["iteration"])
    if current["id"] != expected["id"]:
        raise ValueError("Checkpoint changed after planning; no submission is permitted")


@contextlib.contextmanager
def _record_commands(env):
    commands = []
    original_exec = U.exec_command_cpu
    previous_env = {key: os.environ.get(key) for key in env}
    try:
        U.exec_command_cpu = lambda command, **kwargs: commands.append(command)
        os.environ.update(env)
        yield commands
    finally:
        U.exec_command_cpu = original_exec
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_submission(commands, expected_argv):
    prefix = "export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "
    matches = [command[len(prefix):] for command in commands if command.startswith(prefix)]
    if len(commands) != 2 or len(matches) != 1:
        raise ValueError("U.execute_train command contract changed; review before executing")
    words = shlex.split(matches[0])
    if words[:3] != ["ray", "job", "submit"] or not words[3].startswith("--runtime-env-json="):
        raise ValueError("Unexpected Ray submission prefix")
    expected = ["--", "python3", str(U.repo_base_dir / "train.py"), *expected_argv]
    if words[4:] != expected:
        raise ValueError("Recorded entrypoint differs from the validated replay argv")
    runtime_env = json.loads(words[3].split("=", 1)[1])
    if set(runtime_env) != {"env_vars"}:
        raise ValueError("Unexpected runtime environment structure")
    return shlex.join(expected[1:]), runtime_env


def _build_plan(args):
    source_argv, source_env = _load_source_plan(args.source_plan)
    model_argv = shlex.split(U.shell_safe_model_args(MODEL_TYPE))
    if source_argv[:len(model_argv)] != model_argv:
        raise ValueError("Recorded model prefix differs from this Miles Qwen3 registry; review source versions")
    profile_argv, token_budget_override = _profile_token_budget(source_argv, args)
    argv, removed = _replay_arguments(profile_argv, args)
    if argv[:len(model_argv)] != model_argv:
        raise ValueError("Replay unexpectedly changed model arguments")
    env = {k: v for k, v in source_env.items() if not k.startswith("WANDB_")}
    env.update({RUN_ID_ENV: args.run_id, "RUBIN_RUN_ID": args.run_id, "WANDB_MODE": "disabled"})
    record_env = {
        "MILES_SCRIPT_EXTERNAL_RAY": "1", "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
        "RAY_ADDRESS": args.ray_address, "NCCL_NVLS_ENABLE": env.get("NCCL_NVLS_ENABLE", "0"),
    }
    with _record_commands(record_env) as commands:
        U.execute_train(
            train_args=shlex.join(argv[len(model_argv):]), config=args, num_gpus_per_node=4,
            megatron_model_type=MODEL_TYPE, megatron_path=args.megatron_path, extra_env_vars=env,
        )
    entrypoint, runtime_env = _parse_submission(commands, argv)
    checkpoint = _checkpoint_manifest(args.checkpoint_root, args.checkpoint_iteration)
    if args.expected_checkpoint_id and args.expected_checkpoint_id != checkpoint["id"]:
        raise ValueError("Checkpoint fingerprint differs from the explicitly expected shared checkpoint")
    output = Path(args.output_dir).resolve()
    if output == Path(checkpoint["root"]) or Path(checkpoint["root"]) in output.parents:
        raise ValueError("Profile output must be outside the checkpoint directory")
    return {
        "schema": 1, "run_id": args.run_id, "submission_id": args.submission_id,
        "ray_address": JobsAPI(args.ray_address).address, "entrypoint": entrypoint,
        "entrypoint_sha256": _sha256(entrypoint.encode()), "runtime_env": runtime_env,
        "checkpoint": checkpoint, "source_plan_sha256": _sha256(Path(args.source_plan).read_bytes()),
        "output_dir": str(output), "trace_dir": str(output / "traces"),
        "max_trace_bytes": int(args.max_trace_gib * 1024**3), "max_runtime_seconds": args.max_runtime_seconds,
        "poll_seconds": args.poll_seconds, "submission_grace_seconds": args.submission_grace_seconds,
        "removed_flags": removed, "planned_rollouts": 3, "optimizer_steps_per_rollout": 4,
        "profile_token_budget_override": token_budget_override,
        "planned_optimizer_steps": 12, "profiling_coverage_known": True,
        "profiled_rollouts": [args.checkpoint_iteration + 1, args.checkpoint_iteration + 2],
        "profile_scope": "First replay rollout tail through second rollout train end; all four trainer ranks",
        "sglang_manual_payload": {
            "output_dir": str(output / "traces/sglang-engine0"), "num_steps": 4,
            "activities": ["CPU", "GPU"], "profile_by_stage": True,
            "with_stack": False, "record_shapes": False, "merge_profiles": False,
        },
        "checkpoint_identity_scope": "Metadata hashes and shard names/sizes; not a full tensor-content hash",
        "checkpoint_selection": "Actor tracker must equal the requested iteration; reference keeps its own selection",
        "limit_scope": "Polling limits can overshoot; Ray API availability is required to stop the exact job",
    }


def _assert_idle(api, submission_id):
    for job in api.list_jobs():
        if job.get("submission_id") == submission_id:
            raise ValueError("Profile submission ID was already used")
        if job.get("type") == "SUBMISSION" and job["status"] not in TERMINAL:
            raise ValueError("An active Ray submission exists; profiling must not overlap the main run")


def _raise_walk_error(error):
    raise error


def _trace_bytes(path):
    size = 0
    for directory, dirs, files in os.walk(path, followlinks=False, onerror=_raise_walk_error):
        for name in dirs + files:
            item = Path(directory) / name
            if item.is_symlink():
                raise ValueError("Trace directory contains a symlink; refuse to follow external paths")
        for name in files:
            try:
                size += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                pass  # A profiler may atomically rename its temporary output.
    return size


def _assert_target(plan, job):
    env = (job.get("runtime_env") or {}).get("env_vars") or {}
    if (job.get("type") != "SUBMISSION" or job.get("submission_id") != plan["submission_id"]
            or env.get(RUN_ID_ENV) != plan["run_id"]
            or _sha256((job.get("entrypoint") or "").encode()) != plan["entrypoint_sha256"]):
        raise _IdentityMismatch("Exact profile submission identity mismatch; no stop is permitted")


def _guard_action(plan, job, trace_bytes, now):
    _assert_target(plan, job)
    if job["status"] in TERMINAL:
        return "terminal"
    if trace_bytes >= plan["max_trace_bytes"]:
        return "trace_byte_limit"
    if now >= plan["deadline_timestamp"]:
        return "runtime_limit"
    return "monitor"


def _persist_guard(path, state):
    try:
        persist(path, state)
    except OSError as exc:
        # A full/unavailable output filesystem must not terminate the guard.
        print(json.dumps({"state_write_error": type(exc).__name__, "state": state["state"]}), flush=True)


def _watch(plan, api, state_path):
    _assert_idle(api, plan["submission_id"])
    state = {"state": "armed", "run_id": plan["run_id"], "submission_id": plan["submission_id"],
             "pid": os.getpid(), "armed_at": utc(plan["armed_timestamp"]),
             "deadline_at": utc(plan["deadline_timestamp"]), "stop_requests": 0, "api_errors": 0}
    persist(state_path, state)
    while True:
        now = time.time()
        try:
            job = api.get_job(plan["submission_id"])
            _assert_target(plan, job)
            try:
                byte_count = _trace_bytes(plan["trace_dir"])
                action = _guard_action(plan, job, byte_count, now)
            except (OSError, ValueError) as exc:
                byte_count = None
                action = "terminal" if job["status"] in TERMINAL else "trace_scan_error"
                state["trace_scan_error"] = str(exc)
            state.update(state="monitoring", ray_status=job["status"], trace_bytes=byte_count)
            if action == "terminal":
                state.update(state="finished", terminal_status=job["status"], completed_at=utc(now))
                _persist_guard(state_path, state)
                return
            if action != "monitor":
                state.update(state="stopping", stop_reason=action, stop_requests=state["stop_requests"] + 1)
                state["last_stop_response"] = api.stop_job(plan["submission_id"])
        except HTTPError as exc:
            if exc.code == 404 and now >= plan["armed_timestamp"] + plan["submission_grace_seconds"]:
                state.update(state="expired_no_job", reason="No exact submission appeared within the grace period")
                _persist_guard(state_path, state)
                return
            state.update(api_errors=state["api_errors"] + 1, last_api_error=type(exc).__name__)
        except _IdentityMismatch as exc:
            state.update(state="rejected", reason=str(exc))
            _persist_guard(state_path, state)
            return
        except Exception as exc:
            state.update(api_errors=state["api_errors"] + 1, last_api_error=type(exc).__name__)
        _persist_guard(state_path, state)
        time.sleep(plan["poll_seconds"])


def _guard(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(plan["output_dir"])
    with (output / "profile-guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _watch(plan, JobsAPI(plan["ray_address"]), output / "profile-guard.json")


def _submit(args, plan):
    # Ray is optional for CPU planning/tests; only execution imports its client.
    from ray.job_submission import JobSubmissionClient

    class _BoundedClient(JobSubmissionClient):
        def _do_request(self, method, endpoint, **kwargs):
            return super()._do_request(method, endpoint, **{**kwargs, "timeout": 20})

    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError("Execution requires MILES_SCRIPT_EXTERNAL_RAY=1 and an existing idle cluster")
    if not args.expected_checkpoint_id:
        raise ValueError("Execution requires the expected checkpoint fingerprint from the reviewed plan")
    _verify_checkpoint_unchanged(plan)
    api = JobsAPI(plan["ray_address"])
    _assert_idle(api, plan["submission_id"])
    output = Path(plan["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    (output / "traces").mkdir()
    now = time.time()
    plan = {**plan, "armed_timestamp": now, "deadline_timestamp": now + plan["max_runtime_seconds"]}
    plan_path = output / "profile-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    with (output / "profile-guard.log").open("x") as log:
        guard = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--guard-plan", str(plan_path)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    state_path = output / "profile-guard.json"
    ready_deadline = time.monotonic() + 30
    while time.monotonic() < ready_deadline:
        if guard.poll() is not None:
            raise RuntimeError("Guard exited before submission; inspect profile-guard.log")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (state.get("state") == "armed" and state.get("pid") == guard.pid
                    and state.get("run_id") == plan["run_id"]
                    and state.get("submission_id") == plan["submission_id"]):
                break
        time.sleep(0.2)
    else:
        raise RuntimeError("Guard did not arm; no job was submitted")
    _assert_idle(api, plan["submission_id"])
    client = _BoundedClient(plan["ray_address"])
    _verify_checkpoint_unchanged(plan)
    submission_id = client.submit_job(
        entrypoint=plan["entrypoint"], runtime_env=plan["runtime_env"], submission_id=plan["submission_id"],
        metadata={"profile_run_id": plan["run_id"], "checkpoint_id": plan["checkpoint"]["id"]},
    )
    if submission_id != plan["submission_id"]:
        raise RuntimeError("Ray returned a different submission ID; inspect before taking any action")
    print(json.dumps({"submitted": submission_id, "guard_pid": guard.pid, "state_path": str(state_path)}))


def execute(args):
    if args.guard_plan:
        _guard(args.guard_plan)
        return
    plan = _build_plan(args)
    if not args.execute_run:
        print(json.dumps(plan, indent=2))
        return
    _submit(args, plan)


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
