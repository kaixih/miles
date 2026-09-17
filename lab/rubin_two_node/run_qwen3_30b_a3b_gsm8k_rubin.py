"""Run Qwen3-30B-A3B GSM8K GRPO learning on one four-GPU Rubin node.

Existing HF and torch_dist checkpoints, prepared GSM8K prompt/label JSONL files,
writable output storage, and a verified external four-GPU Ray cluster are
required. This launcher does not download data, convert models, or start Ray.
The default Qwen3 chat template retains thinking. No prompt-group filter is used.

Args:
  --model-dir / --data-dir: Existing checkpoint and dataset parents.
  --prompt-data-path: Training JSONL; defaults to train.jsonl under data-dir.
  --eval-prompt-data-path: Fixed evaluation subset JSONL; defaults to
      test-fixed-256.jsonl under data-dir. Prepare the chosen subset once.
  --output-dir / --save-dir: Prepared output root and optional checkpoint override.
  --num-rollout: Default 50; each rollout generates 256 prompts times eight samples.
  --global-batch-size: Responses per optimizer step, default 512 (four steps/rollout).
  --save-interval: Default 50; retain optimizer state in the final checkpoint.
      Positive intervals also save the final rollout; 0 disables checkpoints.
  --save-retain-interval: Default 0 leaves native retention disabled. A positive
      value retains checkpoints at multiples of that interval plus the latest;
      1000000 keeps only the latest nonzero checkpoint in this 50-rollout run.
      Megatron prunes after model save, before Miles saves its rollout state.
  --save-trigger-sentinel: Optional existing Miles checkpoint-request file path.
  --eval-interval: Default 10, with evaluation before the first training rollout.
  --no-enable-eval: Disable evaluation for an explicitly bounded execution check.
  --rollout-max-context-len: 0 derives prompt plus response limits (512 + 1024).
  --sglang-enable-cuda-graph: Enable SGLang's default decode CUDA Graph path by
      omitting --sglang-disable-cuda-graph. Default false preserves the old eager
      recipe. Piecewise/prefill CUDA Graph stays explicitly disabled in both
      modes; this option does not enable torch.compile or force graph support.
      Confirm actual capture/replay in SGLang logs before claiming graph use.
  --sglang-moe-runner-backend: BF16 rollout MoE backend: triton (default),
      flashinfer_cutlass, or flashinfer_trtllm. Other rollout settings are unchanged.
  --custom-rm-path: Async adapter for the original verl GSM8K answer scorer.
  --save-debug-event-data / --save-debug-rollout-data: Optional audit artifacts.
  --save-local-weight-checksum: Optional CPU tensor hashes after every optimizer
      step; requires event output and can be expensive for a 200-update run.
  --print-only: Resolve train.py arguments without input checks or job submission.
  --extra-env-vars: Additional Ray environment as JSON or KEY=value.

Example (inside the prepared head container):
  MILES_SCRIPT_EXTERNAL_RAY=1 python3 lab/rubin_two_node/run_qwen3_30b_a3b_gsm8k_rubin.py \
      --model-dir /models --prompt-data-path /inputs/train.jsonl \
      --eval-prompt-data-path /inputs/test-fixed-256.jsonl --output-dir /run-output \
      --megatron-path /opt/Megatron-LM --print-only
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_nodes: int = 1
    num_gpus_per_node: int = 4
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/opt/Megatron-LM"
    prompt_data_path: str = ""
    eval_prompt_data_path: str = ""
    save_dir: str = ""
    num_rollout: int = 50
    rollout_batch_size: int = 256
    n_samples_per_prompt: int = 8
    global_batch_size: int = 512
    rollout_max_prompt_len: int = 512
    rollout_max_response_len: int = 1024
    rollout_max_context_len: int = 0
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    eps_clip_c: float = 10.0
    kl_loss_coef: float = 0.001
    custom_rm_path: str = "lab.rubin_two_node.gsm8k_verl_reward.reward_func"
    max_tokens_per_gpu: int = 8192
    sglang_mem_fraction_static: float = 0.55
    sglang_max_running_requests: int = 128
    sglang_enable_cuda_graph: bool = False
    sglang_moe_runner_backend: str = "triton"
    rollout_num_gpus_per_engine: int = 1
    enable_eval: bool = True
    eval_interval: int = 10
    eval_temperature: float = 1.0
    eval_top_p: float = 0.7
    eval_top_k: int = -1
    n_samples_per_eval_prompt: int = 1
    save_interval: int = 50
    save_retain_interval: int = 0
    save_trigger_sentinel: str = ""
    save_debug_event_data: str = ""
    save_debug_rollout_data: str = ""
    save_local_weight_checksum: bool = False
    enable_wandb: bool = False
    print_only: bool = False

    def __post_init__(self):
        if (self.num_nodes, self.num_gpus_per_node) != (1, 4):
            raise ValueError("This comparison requires one node with four visible GPUs")
        if self.rollout_num_gpus_per_engine not in (1, 2, 4):
            raise ValueError("Rollout engines must use one, two, or four GPUs within this node")
        if self.num_rollout < 1 or self.rollout_batch_size < 1 or self.n_samples_per_prompt < 2:
            raise ValueError("Use positive rollout counts and at least two samples per prompt")
        if self.global_batch_size < 1 or self.global_batch_size % 4:
            raise ValueError("Global response batch must be positive and divisible by data-parallel size 4")
        if (self.rollout_batch_size * self.n_samples_per_prompt) % self.global_batch_size:
            raise ValueError("Each rollout must contain an integer number of global response batches")
        if self.rollout_max_prompt_len <= 0 or self.rollout_max_response_len <= 0:
            raise ValueError("Prompt and response token limits must be positive")
        if self.rollout_max_context_len < 0 or (
            self.rollout_max_context_len
            and self.rollout_max_context_len < self.rollout_max_prompt_len + self.rollout_max_response_len
        ):
            raise ValueError("Context must be zero (automatic) or cover prompt plus response limits")
        for temperature, top_p, top_k in (
            (self.rollout_temperature, self.rollout_top_p, self.rollout_top_k),
            (self.eval_temperature, self.eval_top_p, self.eval_top_k),
        ):
            if temperature < 0 or not 0 < top_p <= 1 or (top_k != -1 and top_k < 1):
                raise ValueError("Sampling requires temperature >= 0, 0 < top-p <= 1, and top-k -1 or positive")
        if self.learning_rate <= 0 or self.adam_eps <= 0 or self.weight_decay < 0 or self.clip_grad <= 0:
            raise ValueError("Optimizer limits must be positive, with nonnegative weight decay")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.eps_clip < 0 or self.eps_clip_high < 0 or self.eps_clip_c <= 1 or self.kl_loss_coef < 0:
            raise ValueError("Invalid PPO clipping or KL coefficient")
        if self.max_tokens_per_gpu < self.context_length or self.sglang_max_running_requests < 1:
            raise ValueError("Token budget must cover one full context and request concurrency must be positive")
        if not 0 < self.sglang_mem_fraction_static < 1:
            raise ValueError("SGLang memory fraction must be in (0, 1)")
        if self.sglang_moe_runner_backend not in ("triton", "flashinfer_cutlass", "flashinfer_trtllm"):
            raise ValueError("MoE backend must be triton, flashinfer_cutlass, or flashinfer_trtllm")
        if self.save_interval < 0 or (self.enable_eval and self.eval_interval < 1):
            raise ValueError("Save interval must be nonnegative; enabled evaluation needs a positive interval")
        if self.save_retain_interval < 0:
            raise ValueError("Checkpoint retention interval must be nonnegative")
        if self.save_retain_interval and not self.save_interval:
            raise ValueError("Checkpoint retention requires checkpoint saving to be enabled")
        if self.save_trigger_sentinel and not self.save_interval:
            raise ValueError("A checkpoint sentinel requires checkpoint saving to be enabled")
        if self.n_samples_per_eval_prompt < 1 or not self.custom_rm_path:
            raise ValueError("Evaluation needs positive sample count and the GSM8K scorer path must be set")
        if self.save_local_weight_checksum and not self.save_debug_event_data:
            raise ValueError("--save-local-weight-checksum requires --save-debug-event-data")

    @property
    def num_steps_per_rollout(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt // self.global_batch_size

    @property
    def context_length(self) -> int:
        return self.rollout_max_context_len or self.rollout_max_prompt_len + self.rollout_max_response_len

    @property
    def hf_checkpoint(self) -> Path:
        return Path(self.model_dir) / MODEL_NAME

    @property
    def ref_checkpoint(self) -> Path:
        return Path(self.model_dir) / f"{MODEL_NAME}_torch_dist"

    @property
    def prompt_data(self) -> Path:
        return Path(self.prompt_data_path) if self.prompt_data_path else Path(self.data_dir) / "train.jsonl"

    @property
    def eval_prompt_data(self) -> Path:
        if self.eval_prompt_data_path:
            return Path(self.eval_prompt_data_path)
        return Path(self.data_dir) / "test-fixed-256.jsonl"

    @property
    def checkpoint_output(self) -> Path:
        return Path(self.save_dir) if self.save_dir else Path(self.output_dir) / "checkpoints"


def _checkpoint_args(args: ScriptArgs) -> str:
    result = (
        f"--hf-checkpoint {shlex.quote(str(args.hf_checkpoint))} "
        f"--ref-load {shlex.quote(str(args.ref_checkpoint))} --megatron-to-hf-mode raw "
    )
    if args.save_interval:
        # train.py also saves the final rollout when an interval is configured.
        result += f"--save {shlex.quote(str(args.checkpoint_output))} --save-interval {args.save_interval} "
    if args.save_retain_interval:
        result += f"--save-retain-interval {args.save_retain_interval} "
    if args.save_trigger_sentinel:
        result += f"--save-trigger-sentinel {shlex.quote(args.save_trigger_sentinel)} "
    return result


def _rollout_args(args: ScriptArgs) -> str:
    return (
        f"--prompt-data {shlex.quote(str(args.prompt_data))} "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        f"--custom-rm-path {shlex.quote(args.custom_rm_path)} --reward-key reward --eval-reward-key reward "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--num-steps-per-rollout {args.num_steps_per_rollout} --balance-data "
        f"--rollout-max-prompt-len {args.rollout_max_prompt_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-context-len {args.context_length} "
        f"--rollout-temperature {args.rollout_temperature} "
        f"--rollout-top-p {args.rollout_top_p} --rollout-top-k {args.rollout_top_k} "
    )


def _eval_args(args: ScriptArgs) -> str:
    if not args.enable_eval:
        return ""
    return (
        f"--eval-interval {args.eval_interval} "
        f"--eval-prompt-data gsm8k {shlex.quote(str(args.eval_prompt_data))} "
        "--eval-input-key prompt --eval-label-key label "
        f"--n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
        f"--eval-temperature {args.eval_temperature} --eval-top-p {args.eval_top_p} --eval-top-k {args.eval_top_k} "
        f"--eval-max-prompt-len {args.rollout_max_prompt_len} "
        f"--eval-max-response-len {args.rollout_max_response_len} --eval-max-context-len {args.context_length} "
    )


def _training_args(args: ScriptArgs) -> str:
    return (
        "--advantage-estimator grpo --calculate-per-token-loss --entropy-coef 0 "
        f"--eps-clip {args.eps_clip} --eps-clip-high {args.eps_clip_high} --eps-clip-c {args.eps_clip_c} "
        f"--use-kl-loss --kl-loss-coef {args.kl_loss_coef} --kl-loss-type low_var_kl --kl-coef 0 "
        f"--optimizer adam --lr {args.learning_rate} --lr-decay-style constant "
        f"--adam-beta1 {args.adam_beta1} --adam-beta2 {args.adam_beta2} --adam-eps {args.adam_eps} "
        f"--weight-decay {args.weight_decay} --clip-grad {args.clip_grad} "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall --recompute-granularity full "
        "--recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} --log-probs-chunk-size 512 "
        "--bf16 --attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
    )


def _build_train_args(args: ScriptArgs) -> str:
    # Decode graphs and piecewise/prefill graphs are separate SGLang paths.
    # Leave the latter off until it has its own explicit, validated recipe.
    graph_args = "" if args.sglang_enable_cuda_graph else "--sglang-disable-cuda-graph "
    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-ep-size 1 --sglang-dtype bfloat16 "
        f"--sglang-moe-runner-backend {args.sglang_moe_runner_backend} "
        "--sglang-attention-backend triton --sglang-bf16-gemm-backend torch "
        f"{graph_args}--sglang-disable-piecewise-cuda-graph "
        f"--sglang-context-length {args.context_length} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
    )
    cluster_args = (
        f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate --use-miles-router --object-store-backend ray "
        "--update-weights-interval 1 --update-weight-buffer-size 536870912 "
    )
    audit_args = ""
    if args.save_debug_event_data:
        audit_args += f"--save-debug-event-data {shlex.quote(args.save_debug_event_data)} "
    if args.save_local_weight_checksum:
        audit_args += "--save-local-weight-checksum "
    if args.save_debug_rollout_data:
        audit_args += f"--save-debug-rollout-data {shlex.quote(args.save_debug_rollout_data)} "
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""
    return " ".join(
        [
            _checkpoint_args(args),
            _rollout_args(args),
            _training_args(args),
            _eval_args(args),
            sglang_args,
            cluster_args,
            audit_args,
            wandb_args,
        ]
    )


def _runtime_env() -> dict[str, str]:
    return {
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_NVLS_ENABLE": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SGLANG_ENABLE_SPEC_V2": "0",
    }


def _validate_inputs(args: ScriptArgs) -> None:
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1":
        raise RuntimeError("Set MILES_SCRIPT_EXTERNAL_RAY=1 after the external four-GPU Ray cluster is ready")
    required = [args.hf_checkpoint / "config.json", args.prompt_data]
    if args.enable_eval:
        required.append(args.eval_prompt_data)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    index = json.loads((args.hf_checkpoint / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        path = args.hf_checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty HF shard: {path}")
    tracker = args.ref_checkpoint / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != "release":
        raise RuntimeError(f"Expected the existing converted release checkpoint: {tracker}")
    if not (args.ref_checkpoint / "release").is_dir():
        raise FileNotFoundError(args.ref_checkpoint / "release")


def execute(args: ScriptArgs):
    train_args = _build_train_args(args)
    if args.print_only:
        model_args = U.shell_safe_model_args(MODEL_TYPE)
        print(
            json.dumps(
                {
                    "train_script": str(U.repo_base_dir / "train.py"),
                    "argv": shlex.split(f"{model_args} {train_args}"),
                    "extra_runtime_env": U.resolve_extra_env_vars(_runtime_env(), args),
                    "optimizer_steps_per_rollout": args.num_steps_per_rollout,
                    "planned_optimizer_steps": args.num_rollout * args.num_steps_per_rollout,
                    "external_ray_required": True,
                },
                indent=2,
            )
        )
        return
    _validate_inputs(args)
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MODEL_TYPE,
        megatron_path=args.megatron_path,
        extra_env_vars=_runtime_env(),
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
