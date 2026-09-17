#!/usr/bin/env python3
"""Build the NEW offline TRTLLM deck from explicitly bound, retained evidence.

Never allocates, runs models, queries remote systems, or edits older reports.
Without complete evidence the deck is a pending/interim artifact. --final is
strict: both raw logs must prove50/200 completion and all four matched new
prefill/decode images must have retained, hash-verified source traces.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "reports/rubin-gb300-qwen3-trtllm-full"
INPUTS = REPO / "outputs/rubin-gb300-qwen3-trtllm-full"
sys.path.insert(0, str(REPO / "lab/rubin_two_node"))
from collect_cudagraph_snapshot import completion_validation
from collect_qwen3_snapshot import scalar_flag

spec = importlib.util.spec_from_file_location("previous_paired_timing_logic", REPO / "reports/rubin-gb300-qwen3-cudagraph/generate.py")
old_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_builder)

EXPERIMENT = "qwen3-trtllm-full-v1"
PLATFORMS = ("rubin", "gb300")
DATASETS = {"train.jsonl": "f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8",
            "test-fixed-256.jsonl": "93ed3ccda6ecd09ce0665d0423bf8720b7db97823b0ce2a1c4d20483f410ce99"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read(path, optional=False):
    if optional and (path is None or not path.exists()):
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("Expected bounded regular JSON: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def bind_experiment(experiment):
    if experiment.get("schema") != "qwen3-trtllm-full-experiment-v1" or experiment.get("experiment_id") != EXPERIMENT:
        raise ValueError("Wrong new-experiment manifest")
    bindings = {x["label"]: x["run_id"] for x in experiment["run_bindings"]}
    if len(experiment["run_bindings"]) != 2 or set(bindings) != set(PLATFORMS):
        raise ValueError("Require exactly two explicit platform/run bindings")
    for label, run_id in bindings.items():
        if not re.fullmatch(r"20260917-" + label + r"-j\d+-trtllm", run_id):
            raise ValueError("Refusing a baseline or unbound run ID")
    return bindings


def check_recipe(metadata):
    if not metadata.get("actual_recipe_available"):
        return False
    recipe = metadata["recipe"]
    expected = {"model": "Qwen3-30B-A3B", "global_batch_size": "512",
                "max_training_tokens_per_gpu": 4096, "max_logprob_tokens_per_gpu": 4096,
                "reward_function": "lab.rubin_two_node.gsm8k_verl_reward.reward_func",
                "sglang_moe_runner_backend": "flashinfer_trtllm", "save_optimizer": False}
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("Actual learning/backend recipe differs from the bound TRTLLM experiment")
    argv = metadata.get("actual_ray_entrypoint_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
        raise ValueError("Actual Ray entrypoint argv is required to verify the displayed recipe")
    integers = {"--num-rollout": 50, "--num-steps-per-rollout": 4,
                "--rollout-batch-size": 256, "--n-samples-per-prompt": 8,
                "--global-batch-size": 512, "--rollout-max-prompt-len": 512,
                "--rollout-max-response-len": 1024, "--rollout-max-context-len": 1536,
                "--rollout-top-k": -1, "--tensor-model-parallel-size": 1,
                "--pipeline-model-parallel-size": 1, "--context-parallel-size": 1,
                "--expert-model-parallel-size": 4, "--expert-tensor-parallel-size": 1,
                "--max-tokens-per-gpu": 4096, "--rollout-num-gpus-per-engine": 1,
                "--sglang-ep-size": 1, "--sglang-context-length": 1536,
                "--actor-num-nodes": 1, "--actor-num-gpus-per-node": 4, "--num-gpus-per-node": 4,
                "--n-samples-per-eval-prompt": 1, "--eval-top-k": -1,
                "--eval-max-prompt-len": 512, "--eval-max-response-len": 1024,
                "--eval-max-context-len": 1536}
    defaults = {"--log-probs-max-tokens-per-gpu": 4096, "--sglang-tp-size": 1, "--sglang-dp-size": 1}
    for flag, expected_value in {**integers, **defaults}.items():
        if scalar_flag(argv, flag, integer=True, default=defaults.get(flag)) != expected_value:
            raise ValueError("Actual displayed recipe differs: " + flag)
    for flag, expected_value in {"--rollout-temperature": 1.0, "--rollout-top-p": 1.0,
                                 "--eval-temperature": 1.0, "--eval-top-p": 0.7, "--lr": 1e-6}.items():
        raw = scalar_flag(argv, flag)
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError("Missing or nonnumeric recipe argument: " + flag) from None
        if not math.isfinite(value) or value != expected_value:
            raise ValueError("Actual displayed recipe differs: " + flag)
    for flag, expected_value in {"--sglang-attention-backend": "triton", "--sglang-bf16-gemm-backend": "torch",
                                 "--sglang-moe-runner-backend": "flashinfer_trtllm", "--sglang-dtype": "bfloat16",
                                 "--custom-rm-path": expected["reward_function"]}.items():
        if scalar_flag(argv, flag) != expected_value:
            raise ValueError("Actual displayed recipe differs: " + flag)
    if (Path(scalar_flag(argv, "--hf-checkpoint", default="")).name != expected["model"]
            or not {"--bf16", "--apply-chat-template", "--colocate", "--no-save-optim"} <= set(argv)
            or any(v == "--dynamic-sampling-filter-path" or v.startswith("--dynamic-sampling-filter-path=") for v in argv)
            or "--fp16" in argv or "--sglang-disable-cuda-graph" in argv
            or scalar_flag(argv, "--sglang-cuda-graph-backend-decode", default="full") != "full"
            or scalar_flag(argv, "--sglang-cuda-graph-backend-prefill") not in (None, "disabled")
            or ("--sglang-disable-piecewise-cuda-graph" not in argv
                and scalar_flag(argv, "--sglang-cuda-graph-backend-prefill") != "disabled")):
        raise ValueError("Actual model/BF16/chat/filter/graph configuration differs from the displayed recipe")
    if (metadata.get("gpus") != 4 or metadata.get("expected_rollouts") != 50
            or metadata.get("optimizer_steps_per_rollout") != 4
            or metadata.get("graph", {}).get("decode_requested") is not True
            or metadata.get("graph", {}).get("prefill_requested") is not False
            or metadata.get("profiling_coverage_known") is not True or metadata.get("profiled_rollouts") != []):
        raise ValueError("GPU/update/graph/profile scope differs from the main experiment")
    if any(metadata.get("input_preparation", {}).get("data", {}).get(k, {}).get("sha256") != v
           for k, v in DATASETS.items()):
        raise ValueError("Main dataset hashes differ from the shared frozen inputs")
    return True


def validate_main(comparison, health, inputs, bindings):
    result = {"complete": False, "runs": {}, "reason": "Awaiting both new main runs."}
    if comparison is None:
        return result
    if comparison.get("schema") != "miles-qwen3-comparison-v1" or comparison.get("experiment_id") != EXPERIMENT:
        raise ValueError("Only the new TRTLLM comparison is accepted")
    labels = [r.get("label") for r in comparison.get("runs", [])]
    if len(labels) != len(set(labels)) or set(labels) - set(PLATFORMS):
        raise ValueError("Unexpected or duplicate platform in comparison")
    if health is not None and (health.get("schema") != "miles-run-health-v1"
            or health.get("comparison_sha256") != sha(inputs / "comparison.json")):
        raise ValueError("Health evidence is stale or mismatched")
    for run in comparison["runs"]:
        label, metadata = run["label"], run["metadata"]
        if metadata.get("run_id") != bindings[label]:
            raise ValueError("Old or unbound main run")
        recipe_ok = check_recipe(metadata)
        log = inputs / label / "logs/qwen3_train.log"
        if not log.is_file() or log.is_symlink() or log.stat().st_size > 1024**3 or sha(log) != run["source_log_sha256"]:
            raise ValueError("Main source log missing, changed, or exceeds1GiB")
        check = completion_validation(run, log.read_bytes())
        if health:
            matches = [r for r in health["runs"] if r["label"] == label and r["run_id"] == bindings[label]]
            if len(matches) != 1 or matches[0]["completion_validation"] != check:
                raise ValueError("Raw-log completion audit disagrees with the collected health receipt")
        result["runs"][label] = {"run_id": bindings[label], "status": metadata.get("status"),
                                  "completed_rollouts": len(run["completed_training_rollouts"]),
                                  "updates": sum(len(r["train_steps"]) for r in run["rows"]),
                                  "recipe_verified": recipe_ok, "validation": check}
    result["complete"] = set(result["runs"]) == set(PLATFORMS) and all(
        r["recipe_verified"] and r["validation"]["status"] == "PASS"
        and r["completed_rollouts"] == 50 and r["updates"] == 200 for r in result["runs"].values())
    if result["complete"]:
        result["reason"] = "Both runs completed50 rollouts and200 real optimizer updates with finite gradients/losses and NORMAL rank outcomes."
    return result


def numerical_summary(comparison):
    result = {}
    for run in (comparison or {}).get("runs", []):
        rows = run["rows"]
        rewards = [r["common"].get("training_reward_mean") for r in rows if r["training_stage_complete"]]
        rewards = [v for v in rewards if finite(v)]
        evals = [{"rollout_id": row["rollout_id"], "accuracy": e["metrics"].get("eval/gsm8k"),
                  "weight_phase": e.get("weight_phase"), "line": e["line"]}
                 for row in rows for e in row["eval"] if finite(e["metrics"].get("eval/gsm8k"))]
        steps = [s for r in rows for s in r["train_steps"]]
        logprob = [s["metrics"].get("train/train_rollout_logprob_abs_diff") for s in steps]
        grad = [s["metrics"].get("train/grad_norm") for s in steps]
        versions = [{"rollout_id": r["rollout_id"], "min": r["metrics"].get("rollout/weight_version/min"),
                     "max": r["metrics"].get("rollout/weight_version/max"),
                     "mixed": r["metrics"].get("rollout/weight_version/mixed_version_ratio")} for r in rows]
        result[run["label"]] = {"first_reward": rewards[0] if rewards else None,
                               "reward_observations": len(rewards),
                               "last10_reward_mean": statistics.mean(rewards[-10:]) if len(rewards) >= 10 else None,
                               "held_out_events": evals,
                               "last_held_out_accuracy": evals[-1]["accuracy"] if evals else None,
                               "logprob_abs_diff_mean": statistics.mean(logprob) if logprob and all(finite(v) for v in logprob) else None,
                               "logprob_observations": sum(finite(v) for v in logprob),
                               "length_observations": sum(finite(r["common"].get("response_length_mean_tokens")) for r in rows),
                               "truncation_observations": sum(finite(r["common"].get("truncated_ratio")) for r in rows),
                               "all_gradients_finite_positive": bool(grad) and all(finite(v) and v > 0 for v in grad),
                               "weight_versions": versions,
                               "all_rollout_versions_clean": bool(versions) and all(v["mixed"] == 0 and v["min"] == v["max"] == v["rollout_id"] + 1 for v in versions)}
    return result


def performance(comparison):
    # Reuse only selection/math code. No old report input is loaded.
    paired = old_builder.paired_timing(comparison)
    throughput = {}
    wanted = set(paired["rollout_ids"])
    for run in (comparison or {}).get("runs", []):
        rows = [r for r in run["rows"] if r["rollout_id"] in wanted]
        durations = [r["common"].get("rollout_seconds") for r in rows]
        rates = [r["common"].get("output_tokens_per_gpu_generation_second") for r in rows]
        throughput[run["label"]] = (sum(t * r for t, r in zip(durations, rates)) / sum(durations)
            if rows and all(finite(t) and t > 0 and finite(r) for t, r in zip(durations, rates)) else None)
    ratios = {}
    for key in ("step", *old_builder.PAIRED_STAGE_KEYS):
        values = [paired["statistics"].get(p, {}).get(key, {}).get("mean_seconds") for p in PLATFORMS]
        ratios[key] = {"gb300_over_rubin": values[1] / values[0], "rubin_time_reduction": 1 - values[0] / values[1]} if all(finite(v) and v > 0 for v in values) else None
    return {"paired": paired, "weighted_output_tokens_per_gpu_generation_second": throughput, "ratios": ratios,
            "selection_caption": "Steady training rounds; startup and checkpoint-related rounds excluded on both systems.",
            "stage_semantics": "Generation, actor update and other stages are separate recorded timers. These bars are not an additive partition of whole-step time."}


def copy_checked(source, expected_sha, output, name):
    path = Path(source)
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or sha(path) != expected_sha:
        raise ValueError("Profile attachment identity/hash mismatch: " + str(path))
    target = output / "evidence" / name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    return {"url": "evidence/" + name, "sha256": expected_sha, "bytes": target.stat().st_size}


def profiles(paths, comparison, bindings, output):
    result = {"complete": False, "matched": False, "runs": {}, "reason": "Awaiting fresh TRTLLM prefill/decode captures."}
    mains = {r["label"]: r for r in (comparison or {}).get("runs", [])}
    for label, path in paths.items():
        data = read(path, optional=True)
        if data is None:
            continue
        if (data.get("schema") != "trtllm-platform-profile-evidence-v1" or data.get("platform") != label
                or data.get("main_run_id") != bindings[label] or label not in mains
                or data.get("image_reference") != mains[label]["metadata"]["image"]
                or data.get("decode_forward_count") != 4 or data.get("correlated_decode_replay_count") != 4):
            raise ValueError("Profile is unbound, old, or lacks four correlated decode replays")
        for stage in ("prefill", "decode"):
            selected = data.get("profiles", {}).get(stage)
            if not selected or not selected.get("png"):
                raise ValueError("A supplied profile must include both rendered stage screenshots")
            row = selected["selected_forward"]
            if row.get("status") != "PAIRED" or row["fields"].get("bs") != 128:
                raise ValueError("Profile must select a complete batch128 paired forward")
            if stage == "decode" and not row.get("decode_replay_proven"):
                raise ValueError("Decode profile lacks correlated graph evidence")
            if stage == "prefill" and (row["stage"] != "EXTEND" or any(row["graph_launch_calls_by_category"].values())):
                raise ValueError("Prefill must be eager and explicitly captured")
            trace = selected["source_trace"]
            selected["attachments"] = {
                "image": copy_checked(selected["png"], selected["image_sha256"], output, f"{label}-{stage}.png"),
                "trace": copy_checked(trace["path"], trace["sha256"], output, f"{label}-{stage}.trace.json.gz")}
        result["runs"][label] = data
    result["complete"] = set(result["runs"]) == set(PLATFORMS)
    if result["complete"]:
        a, b = (result["runs"][p] for p in PLATFORMS)
        keys = ("frozen_input_sha256", "input_ids_sha256", "workload_sha256")
        token_match = all(a.get(k) and a[k] == b.get(k) for k in keys)
        forward_match = all(a["profiles"][s]["selected_forward"]["fields"] == b["profiles"][s]["selected_forward"]["fields"] for s in ("prefill", "decode"))
        result["matched"] = bool(token_match and forward_match)
        result["reason"] = "Same frozen initial-policy requests, output-token budget and selected-forward token counts." if result["matched"] else "Both profiles valid, but request/forward token fields differ; do not claim a matched profile comparison."
    return result


def build(args):
    experiment = read(args.experiment)
    bindings = bind_experiment(experiment)
    comparison = read(args.inputs / "comparison.json", optional=True)
    health = read(args.inputs / "health.json", optional=True)
    validation = validate_main(comparison, health, args.inputs, bindings)
    output = args.output.resolve()
    if output != ROOT / "site" and ROOT not in output.parents:
        raise ValueError("Output must stay inside the NEW report directory")
    marker = output / "report-data.json"
    if output.exists() and any(output.iterdir()) and (not marker.is_file() or read(marker).get("experiment_id") != EXPERIMENT):
        raise ValueError("Refusing to overwrite an unrelated output directory")
    output.mkdir(parents=True, exist_ok=True)
    profile = profiles({"rubin": args.profile_rubin, "gb300": args.profile_gb300}, comparison, bindings, output)
    numerical = numerical_summary(comparison)
    perf = performance(comparison)
    curve_complete = set(numerical) == set(PLATFORMS) and all(
        n["reward_observations"] == n["length_observations"] == n["truncation_observations"] == 50
        and n["logprob_observations"] == 200 and len(n["held_out_events"]) >= 6 for n in numerical.values())
    numerical_complete = validation["complete"] and curve_complete and all(
        n["all_rollout_versions_clean"] for n in numerical.values())
    complete = numerical_complete and profile["complete"] and profile["matched"] and perf["paired"]["status"] == "available"
    if args.final and not complete:
        raise ValueError("Final report requires both50/200 raw-log audits, all correctness curves/six evaluations, clean advancing rollout weight versions, complete main timings and all four new matched profiles")
    result = {"schema": "qwen3-trtllm-full-report-v1", "experiment_id": EXPERIMENT,
              "built_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "FINAL" if args.final else "READY_FOR_REVIEW" if complete else "PENDING_OR_INTERIM",
              "inputs": {"experiment": experiment, "comparison": comparison, "health": health},
              "derived": {"validation": validation, "correctness_curves_complete": curve_complete,
                          "numerical_checks_complete": numerical_complete,
                          "numerical": numerical, "performance": perf, "profiles": profile},
              "sources": {"builder_sha256": sha(Path(__file__)), "experiment_sha256": sha(args.experiment),
                          "comparison_sha256": sha(args.inputs / "comparison.json") if comparison else None,
                          "profile_input_sha256": {p: sha(f) if f and f.exists() else None for p, f in (("rubin", args.profile_rubin), ("gb300", args.profile_gb300))}},
              "limits": ["One recorded run per platform; no repeated-run confidence interval or hardware-only causal claim.",
                         "Learning curves and finite-update checks provide evidence for this recipe, not a general proof of model correctness.",
                         "Training reward and held-out GSM8K accuracy are separate measurements.",
                         "Profiled GPU windows are independent initial-policy diagnostics and include profiler overhead."]}
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("report.css", "report.js", "navigation.js", "plotly-3.1.0.min.js"):
        shutil.copyfile(ROOT / "assets" / name, assets / name)
    raw = json.dumps(result, indent=2, allow_nan=False)
    marker.write_text(raw + "\n")
    safe = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (output / "index.html").write_text((ROOT / "template.html").read_text().replace("__REPORT_DATA__", safe))
    manifest = {str(p.relative_to(output)): {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in output.rglob("*") if p.is_file() and p.name != "asset-manifest.json"}
    (output / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"site": str(output / "index.html"), "status": result["status"], "main_complete": validation["complete"], "profiles_matched": profile["matched"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=INPUTS)
    p.add_argument("--experiment", type=Path, default=ROOT / "experiment.json")
    p.add_argument("--profile-rubin", type=Path)
    p.add_argument("--profile-gb300", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "site")
    p.add_argument("--final", action="store_true")
    print(json.dumps(build(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
