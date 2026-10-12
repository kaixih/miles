"""GLM-5.3-Flash + OpenSandbox TB2 on eight Rubin nodes (four GPUs each).

Requires the prepared image, an external 8-node Ray cluster, original HF FP8
weights, and the fixed 69/20 task split. See README.md for container preparation.
This recipe does not install packages, apply patches, or save checkpoints.

Args:
    run_id: Fresh output identity.
    num_rollout: Maximum optimizer-update target (default 20).
    hf_checkpoint: Original full-model HF directory, identical on every node.
    prompt_data, eval_prompt_data: Training and held-out JSONL paths.
    prompt_data_sha256, eval_prompt_data_sha256: Expected input SHA256 hashes.
    tasks_dir: Official Terminal-Bench-2 checkout.
    output_dir: Node-local output parent; collect it from every node.
    megatron_path: Image-baked Megatron checkout, readable by the runtime UID.
    offload_train_disk_dir, cache_dir: Fresh node-local directories for this run.

Example (inside the prepared head container):
    MILES_SCRIPT_EXTERNAL_RAY=1 python run_glm5_3_flash_tb2.py train \
        --run-id smoke --num-rollout 3 --hf-checkpoint /inputs/GLM-5.3-Flash \
        --prompt-data /opt/tb2-data/train69.jsonl \
        --eval-prompt-data /opt/tb2-data/eval20.jsonl \
        --prompt-data-sha256 "$TRAIN_SHA" --eval-prompt-data-sha256 "$EVAL_SHA" \
        --tasks-dir /opt/terminal-bench-2 --output-dir /work/output \
        --megatron-path /opt/Megatron-LM \
        --offload-train-disk-dir /work/offload --cache-dir /work/cache

Set OPEN_SANDBOX_API_URL and OPEN_SANDBOX_API_KEY_FILE on the head.
OPEN_SANDBOX_CA_FILE is optional. Only endpoint/file paths are forwarded.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

import typer

from miles.rollout.agentic.credentials import forward_address
from miles.utils.external_utils import command_utils as U

app = typer.Typer()


@dataclass(kw_only=True)
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field()
    num_rollout: int = 20
    hf_checkpoint: str
    prompt_data: str
    eval_prompt_data: str
    prompt_data_sha256: str
    eval_prompt_data_sha256: str
    tasks_dir: str
    output_dir: str = field()
    megatron_path: str
    offload_train_disk_dir: str
    cache_dir: str
    num_nodes: int = 8
    num_gpus_per_node: int = 4

    def __post_init__(self):
        if self.num_nodes != 8 or self.num_gpus_per_node != 4:
            raise ValueError("This recipe requires eight nodes with four GPUs each")
        if self.num_rollout <= 0:
            raise ValueError("--num-rollout must be a positive update count")
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("--run-id must be a nonempty path component")

    @property
    def run_output(self) -> str:
        return str(Path(self.output_dir) / self.run_id)


def _training_args(args: ScriptArgs) -> str:
    # Native HF initialization; omitting save/save-interval disables saving.
    checkpoint_args = (
        f"--hf-checkpoint {quote(args.hf_checkpoint)} --ref-load {quote(args.hf_checkpoint)} "
        "--megatron-to-hf-mode raw "
    )
    rollout_args = (
        "--fully-async --pause-generation-mode in_place "
        "--async-max-concurrent-samples 128 --rollout-submission-granularity sample "
        f"--prompt-data {quote(args.prompt_data)} --input-key prompt --rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 8 --n-samples-per-prompt 8 --global-batch-size 64 "
        "--rollout-max-response-len 8192 --max-seq-len 65536 "
        "--rollout-temperature 0.8 --balance-data "
    )
    eval_args = (
        f"--eval-interval 10 --eval-prompt-data tbench2 {quote(args.eval_prompt_data)} "
        "--n-samples-per-eval-prompt 2 --eval-max-response-len 8192 --eval-temperature 0.8 "
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path openenv_opensandbox_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--tito-model glm53 --use-session-server v1 --session-server-port 30000 --session-server-workers 32 "
        "--pin-rollout-manager-to-head "
    )
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
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --allgather-cp "
        "--model-name glm5_next --qkv-format thd "
        "--actor-num-nodes 4 --actor-num-gpus-per-node 4 --num-gpus-per-node 4 --rollout-num-gpus 16 "
        "--moe-token-dispatcher-type alltoall --update-weight-transfer-mode broadcast "
        "--update-weights-interval 1 --update-weight-buffer-size 1073741824 "
        "--check-weight-update-equal --check-weight-update-skip-list visual. "
        '--train-env-vars \'{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"1"}\' '
        "--stream-optimizer-state-to-disk --stream-optimizer-state-moment-dtype fp32 "
        f"--offload-train-disk-dir {quote(args.offload_train_disk_dir)} "
    )
    return (
        f"{checkpoint_args}{rollout_args}{eval_args}{agent_args}{perf_args}{grpo_args}"
        f"{optimizer_args}{sglang_args}{dashboard_args}{misc_args}"
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)}"
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
        "OPENENV_OPENSANDBOX_LOG_DIR": args.run_output + "/sandbox",
        "OPEN_SANDBOX_API_KEY_FILE": os.environ["OPEN_SANDBOX_API_KEY_FILE"],
    }
    forward_address(env, "OPEN_SANDBOX_API_URL", os.environ["OPEN_SANDBOX_API_URL"])
    if ca_file := os.environ.get("OPEN_SANDBOX_CA_FILE"):
        forward_address(env, "OPEN_SANDBOX_CA_FILE", ca_file)
    return env


@app.callback()
def _cli():
    pass


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit to the existing eight-node Ray cluster; wait for native completion."""
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise ValueError("MILES_SCRIPT_EXTERNAL_RAY=1 requires the prepared eight-node Ray cluster")
    for path, expected_hash in (
        (args.prompt_data, args.prompt_data_sha256),
        (args.eval_prompt_data, args.eval_prompt_data_sha256),
    ):
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
