"""GLM-5.3-Flash + OpenEnv Terminal-Bench-2 on per-episode Daytona sandboxes, Rubin nodes (4 GPUs each).

Derived from the GLM-5.3-Flash Rubin OpenSandbox recipe launcher used by the earlier experiment (not
part of this tree), with the provider swapped for the native Daytona agent function; the model,
topology, algorithm, memory and backend settings are unchanged. Requires the prepared image, an external Ray cluster
(MILES_SCRIPT_EXTERNAL_RAY=1), the original HF FP8 weights and the frozen task split.

Two modes:
  train (default)      8 nodes: 4 training (TP4/PP4/EP4/CP1) + 4 rollout (two TP8/EP8 engines).
  --debug-rollout-only 2 nodes, one TP8 engine, no optimizer: multi-turn sampling diagnostics only.
                       Batch/concurrency are set explicitly and small; eval is off unless asked.

Credentials: set DAYTONA_API_KEY_FILE (and optionally DAYTONA_API_URL) on the head. Only the file PATH
and the URL are forwarded to rollout workers; --pin-rollout-manager-to-head keeps the agent processes
on the head, so that is the only node that needs to read the file.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

import typer

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, forward_address, provision_provider
from miles.utils.external_utils import command_utils as U

app = typer.Typer()


@dataclass(kw_only=True)
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field()
    num_rollout: int = 20
    hf_checkpoint: str
    prompt_data: str
    prompt_data_sha256: str
    eval_prompt_data: str = ""
    eval_prompt_data_sha256: str = ""
    tasks_dir: str
    output_dir: str = field()
    megatron_path: str
    offload_train_disk_dir: str
    cache_dir: str
    num_nodes: int = 8
    num_gpus_per_node: int = 4
    debug_rollout_only: bool = False
    # Formal values; the debug mode must set each of these explicitly and small.
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    async_max_concurrent_samples: int = 128
    eval_interval: int = 10  # 0 disables evaluation entirely (no baseline, no final)
    daytona_create_concurrency: int = 8
    daytona_create_max_retries: int = 8

    def __post_init__(self):
        if self.num_gpus_per_node != 4:
            raise ValueError("This recipe requires four GPUs per node")
        if self.debug_rollout_only:
            if self.num_nodes != 2:
                raise ValueError("--debug-rollout-only samples on exactly two nodes (one TP8 engine)")
            if self.rollout_batch_size * self.n_samples_per_prompt > 16 or self.async_max_concurrent_samples > 16:
                raise ValueError("debug sampling is a diagnostic: keep prompts x samples and concurrency at or below 16")
            if self.num_rollout > 2:
                raise ValueError("debug sampling runs one or two batches; pass --num-rollout explicitly")
        elif self.num_nodes != 8:
            raise ValueError("Training requires eight nodes: four trainer + four rollout")
        if self.num_rollout <= 0:
            raise ValueError("--num-rollout must be a positive batch/update count")
        if self.async_max_concurrent_samples < self.n_samples_per_prompt:
            raise ValueError("--async-max-concurrent-samples must be at least --n-samples-per-prompt")
        if self.eval_interval and not self.eval_prompt_data:
            raise ValueError("a non-zero --eval-interval needs --eval-prompt-data")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("--run-id must be a nonempty path component")

    @property
    def run_output(self) -> str:
        return str(Path(self.output_dir) / self.run_id)

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt


def _training_args(args: ScriptArgs) -> str:
    checkpoint_args = (
        f"--hf-checkpoint {quote(args.hf_checkpoint)} --ref-load {quote(args.hf_checkpoint)} "
        "--megatron-to-hf-mode raw "
    )
    rollout_args = (
        "--fully-async --pause-generation-mode in_place "
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} --rollout-submission-granularity sample "
        f"--prompt-data {quote(args.prompt_data)} --input-key prompt --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} --n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        "--rollout-max-response-len 8192 --max-seq-len 65536 "
        "--rollout-temperature 0.8 --balance-data "
    )
    eval_args = ""
    if args.eval_interval:
        eval_args = (
            f"--eval-interval {args.eval_interval} --eval-prompt-data tbench2 {quote(args.eval_prompt_data)} "
            "--n-samples-per-eval-prompt 2 --eval-max-response-len 8192 --eval-temperature 0.8 "
        )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path openenv_daytona_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm53 --use-session-server v1 --session-server-port 30000 --session-server-workers 32 "
        "--pin-rollout-manager-to-head "
    )
    if args.debug_rollout_only:
        # Trainer stubs still initialize Megatron's parallel state on the engine's eight GPUs; no model is built.
        # TP8 makes the stub data-parallel size 1, so Megatron's batch divisibility checks accept a single sample.
        perf_args = (
            "--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
            "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
            "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
            "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
            "--data-pad-size-multiplier 1024 --log-probs-chunk-size 16384 "
        )
    else:
        perf_args = (
            "--tensor-model-parallel-size 4 --sequence-parallel --pipeline-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 12 "
            "--context-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
            "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
            "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
            "--data-pad-size-multiplier 1024 --log-probs-chunk-size 16384 "
        )
    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--use-tis --tis-clip-low 0.0 --tis-clip 2.0 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 8 --sglang-tp-size 8 --sglang-ep-size 8 "
        "--sglang-moe-runner-backend triton --sglang-mem-fraction-static 0.7 "
        "--sglang-chunked-prefill-size 8192 --sglang-disable-radix-cache "
        "--sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
    )
    dashboard_args = (
        f"--dump-details {quote(args.run_output + '/dump_details')} "
        "--use-miles-dashboard --dashboard-sglang-scrape-mode direct "
    )
    # Debug mode: Miles computes args.world_size from the actor layout before its debug post-processing
    # rewrites that layout to the engine's 8 GPUs (1 node x 8), so the launcher states that layout itself.
    layout_args = (
        "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --num-gpus-per-node 4 --rollout-num-gpus 8 "
        if args.debug_rollout_only
        else "--actor-num-nodes 4 --actor-num-gpus-per-node 4 --num-gpus-per-node 4 --rollout-num-gpus 16 "
    )
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --allgather-cp "
        "--model-name glm5_next --qkv-format thd "
        f"{layout_args}"
        "--moe-token-dispatcher-type alltoall --update-weight-transfer-mode broadcast "
        "--update-weights-interval 1 --update-weight-buffer-size 1073741824 "
        '--train-env-vars \'{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}\' '
        "--stream-optimizer-state-to-disk --stream-optimizer-state-moment-dtype fp32 "
        f"--offload-train-disk-dir {quote(args.offload_train_disk_dir)} "
    )
    if args.debug_rollout_only:
        misc_args += "--debug-rollout-only "
    else:
        misc_args += "--check-weight-update-equal --check-weight-update-skip-list visual. "
    # No W&B: the recipe's get_default_wandb_args would put WANDB_API_KEY into the logged Ray entrypoint.
    return (
        f"{checkpoint_args}{rollout_args}{eval_args}{agent_args}{perf_args}{grpo_args}"
        f"{optimizer_args}{sglang_args}{dashboard_args}{misc_args}"
    )


def _runtime_env(args: ScriptArgs) -> dict[str, str]:
    openenv_dir = U.repo_base_dir / "examples/experimental/openenv"
    env = {
        "PYTHONPATH": f"{args.megatron_path}:{openenv_dir}:{U.repo_base_dir}",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_DIR": str(Path(args.cache_dir) / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(Path(args.cache_dir) / "torchinductor"),
        "TVM_FFI_CACHE_DIR": str(Path(args.cache_dir) / "tvm_ffi"),
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "4096",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(Path(args.cache_dir) / "nccl_trace"),
        "AGENT_MODEL_NAME": "model",
        "OPENENV_MAX_TURNS": "30",
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": "3600",
        "OPENENV_TB2_TASKS_DIR": args.tasks_dir,
        "OPENENV_LAUNCHER": os.environ.get("OPENENV_LAUNCHER", "miles"),
        "OPENENV_RUN_ID": args.run_id,
        "OPENENV_DAYTONA_CREATE_CONCURRENCY": str(args.daytona_create_concurrency),
        "OPENENV_DAYTONA_CREATE_MAX_RETRIES": str(args.daytona_create_max_retries),
    }
    # Key-file path only (never the value), SDK preflight; an explicit path that is unreadable here is an error.
    key_file = os.environ.get("DAYTONA_API_KEY_FILE", "").strip()
    if not key_file:
        raise ValueError("DAYTONA_API_KEY_FILE must name the head node's private Daytona key file")
    provision_provider(env, PROVIDER_CREDENTIALS["daytona"], arg_path=key_file)
    forward_address(env, "DAYTONA_API_URL", os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"))
    return env


@app.callback()
def _cli():
    pass


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit to the existing Ray cluster; wait for native completion."""
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError("MILES_SCRIPT_EXTERNAL_RAY=1 requires the prepared external Ray cluster")
    datasets = [(args.prompt_data, args.prompt_data_sha256)]
    if args.eval_interval:
        datasets.append((args.eval_prompt_data, args.eval_prompt_data_sha256))
    for path, expected_hash in datasets:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen dataset hash mismatch: {path}")
    if Path(args.run_output).exists():
        raise ValueError("A fresh --run-id is required; output already exists")
    args.create_backend().execute_train(
        train_args=_training_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="glm5.3-flash",
        extra_env_vars=_runtime_env(args),
        megatron_path=args.megatron_path,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    app()
