"""Run two short Qwen3.5-35B-A3B GRPO iterations on two 4-GPU Rubin nodes.

Adapted from scripts/run_qwen3_5_35b_a3b_mtp.py. Existing HF and torch_dist
checkpoints, DAPO data, and an external Ray cluster must already be available.
Run only after the separate two-node communication check has passed.

Args:
  --model-dir / --data-dir: Existing checkpoint and dataset parents.
  --output-dir: Writable run output parent, prepared by the cluster launcher.
  --num-rollout: At least two complete generate/train/weight-sync iterations.
  --dynamic-sampling-filter-path: Optional Miles filter for resampling prompt groups.
  --save-debug-event-data: Optional directory for training and served-weight audit events.
  --save-local-weight-checksum: Hash local parameters/state after each training step;
      requires --save-debug-event-data and adds CPU copies of the tensors.
  --save-debug-rollout-data: Optional sample dump path template containing {rollout_id}.
  --save-debug-trajectory-data: Optional JSONL path template containing {rollout_id};
      requires --save-debug-rollout-data in the selected rollout path.
  --print-only: Print the resolved train.py arguments without starting a job.
  --extra-env-vars: Additional Ray runtime environment as JSON or KEY=value.

Example (inside the prepared head container):
  MILES_SCRIPT_EXTERNAL_RAY=1 python lab/rubin_two_node/run_qwen3_5_35b_a3b_rubin.py \
      --model-dir /models --data-dir /models --output-dir /run-output \
      --megatron-path /opt/Megatron-LM --print-only
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 2
    num_gpus_per_node: int = 4
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/opt/Megatron-LM"
    num_rollout: int = 2
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 2
    rollout_max_response_len: int = 256
    rollout_max_prompt_len: int = 1024
    max_tokens_per_gpu: int = 2048
    sglang_mem_fraction_static: float = 0.35
    sglang_max_running_requests: int = 8
    dynamic_sampling_filter_path: str = ""
    save_debug_event_data: str = ""
    save_local_weight_checksum: bool = False
    save_debug_rollout_data: str = ""
    save_debug_trajectory_data: str = ""
    enable_wandb: bool = False
    print_only: bool = False

    def __post_init__(self):
        if (self.num_nodes, self.num_gpus_per_node) != (2, 4):
            raise ValueError("This recipe requires two nodes with four visible GPUs each")
        if self.num_rollout < 2:
            raise ValueError("Use at least two rollouts to exercise updated weights in generation")
        if self.n_samples_per_prompt < 2 or self.rollout_batch_size < 1:
            raise ValueError("GRPO needs at least two samples per prompt and a positive batch size")
        if self.global_batch_size % 4:
            raise ValueError("Global batch size must be divisible by the TP2/PP1/CP1 data-parallel size 4")
        if self.rollout_max_response_len <= 0 or self.rollout_max_prompt_len <= 0:
            raise ValueError("Prompt and response token limits must be positive")
        if self.save_local_weight_checksum and not self.save_debug_event_data:
            raise ValueError("--save-local-weight-checksum requires --save-debug-event-data")
        if self.save_debug_trajectory_data and not self.save_debug_rollout_data:
            raise ValueError("--save-debug-trajectory-data requires --save-debug-rollout-data")

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def hf_checkpoint(self) -> Path:
        return Path(self.model_dir) / "Qwen3.5-35B-A3B"

    @property
    def ref_checkpoint(self) -> Path:
        return Path(self.model_dir) / "Qwen3.5-35B-A3B_torch_dist"

    @property
    def prompt_data(self) -> Path:
        return Path(self.data_dir) / "dapo-math-17k/dapo-math-17k.jsonl"


def _build_train_args(args: ScriptArgs) -> str:
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} "
        f"--ref-load {shlex.quote(str(args.ref_checkpoint))} "
        "--megatron-to-hf-mode raw "
        # Overrides the existing model registry's --mtp-num-layers 1.
        "--mtp-num-layers 0 "
    )
    rollout_args = (
        f"--prompt-data {shlex.quote(str(args.prompt_data))} "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-prompt-len {args.rollout_max_prompt_len} "
        f"--rollout-max-context-len {args.rollout_max_prompt_len + args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} --balance-data "
    )
    if args.dynamic_sampling_filter_path:
        rollout_args += f"--dynamic-sampling-filter-path {shlex.quote(args.dynamic_sampling_filter_path)} "
    perf_args = (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 512 --linear-attention-backend fla "
    )
    algorithm_args = (
        "--advantage-estimator grpo --entropy-coef 0 "
        "--eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = (
        # Two engines, each contained within one four-GPU node.
        "--rollout-num-gpus-per-engine 4 --sglang-ep-size 1 "
        "--sglang-dtype bfloat16 --sglang-moe-runner-backend triton "
        "--sglang-attention-backend triton --sglang-bf16-gemm-backend torch "
        "--sglang-linear-attn-backend triton --sglang-linear-attn-prefill-backend triton "
        "--sglang-disable-cuda-graph --sglang-disable-piecewise-cuda-graph "
        f"--sglang-context-length {args.rollout_max_prompt_len + args.rollout_max_response_len} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
    )
    misc_args = (
        "--bf16 --attention-dropout 0 --hidden-dropout 0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash --colocate --use-miles-router --object-store-backend ray "
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--update-weights-interval 1 --update-weight-buffer-size 536870912 "
    )
    audit_args = ""
    if args.save_debug_event_data:
        audit_args += f"--save-debug-event-data {shlex.quote(args.save_debug_event_data)} "
    if args.save_local_weight_checksum:
        audit_args += "--save-local-weight-checksum "
    if args.save_debug_rollout_data:
        audit_args += f"--save-debug-rollout-data {shlex.quote(args.save_debug_rollout_data)} "
    if args.save_debug_trajectory_data:
        audit_args += f"--save-debug-trajectory-data {shlex.quote(args.save_debug_trajectory_data)} "
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""
    return " ".join([
        ckpt_args, rollout_args, perf_args, algorithm_args, optimizer_args,
        sglang_args, misc_args, audit_args, wandb_args,
    ])


def _runtime_env() -> dict[str, str]:
    return {
        # MNNVL needs cuMem; override Miles' legacy SGLang-compatible default 0.
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "FLA_CONV_BACKEND": "triton",
        "FLA_DISABLE_BACKEND_DISPATCH": "1",
        "SGLANG_ENABLE_SPEC_V2": "0",
    }


def _validate_inputs(args: ScriptArgs) -> None:
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise RuntimeError("Set MILES_SCRIPT_EXTERNAL_RAY=1 after both containers have joined Ray")
    for path in [args.hf_checkpoint / "config.json", args.prompt_data]:
        if not path.is_file():
            raise FileNotFoundError(path)
    index_path = args.hf_checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name in set(index["weight_map"].values()):
        path = args.hf_checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty HF shard: {path}")
    tracker = args.ref_checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != "release":
        raise RuntimeError(f"Expected an existing converted release checkpoint: {tracker}")
    if not (args.ref_checkpoint / "release").is_dir():
        raise FileNotFoundError(args.ref_checkpoint / "release")


def execute(args: ScriptArgs):
    train_args = _build_train_args(args)
    if args.print_only:
        model_args = U.shell_safe_model_args("qwen3.5-35B-A3B")
        print(json.dumps({
            "train_script": str(U.repo_base_dir / "train.py"),
            "argv": shlex.split(f"{model_args} {train_args}"),
            "extra_runtime_env": U.resolve_extra_env_vars(_runtime_env(), args),
            "external_ray_required": True,
        }, indent=2))
        return
    _validate_inputs(args)
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-35B-A3B",
        megatron_path=args.megatron_path,
        extra_env_vars=_runtime_env(),
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
