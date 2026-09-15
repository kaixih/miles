#!/usr/bin/env python3
"""Offline Miles metrics for reports; never contacts Ray, GPUs, or the network.

Usage: python summarize_qwen3_runs.py --run rubin=train.log --metadata runs.json -o report.json
Metadata is keyed by run label; record gpus, optimizer_steps_per_rollout,
expected_rollouts, status, profiling_coverage_known, profiled_rollouts,
profiled_optimizer_steps, and actual image/versions/graph/kernels/recipe.
Defaults describe a fresh Qwen3 50-rollout, four-GPU, four-updates/rollout run.
Unknown profiling coverage excludes timings from the unprofiled summary.
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EVENT = re.compile(r"(?<![\w-])(perf|rollout|eval|step) (-?\d+): (\{.*\})")
STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)")
STAGES = ["actor_train", "log_probs", "ref_log_probs", "train", "train_wait", "update_weights", "rollout", "step"]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported metric value: {type(value).__name__}")


def literal_metrics(text):
    if len(text) > 1_000_000:
        raise ValueError("metric payload exceeds 1 MB")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 20_000:
            raise ValueError("metric payload has too many AST nodes")
        # Only nonfinite numeric names are recognized; function calls stay forbidden.
        class Numbers(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.Constant(float(node.id)) if node.id in {"nan", "inf"} else node
        value = ast.literal_eval(Numbers().visit(tree))
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("metrics must be a dictionary with string keys")
    return json_safe(value)


def first(metrics, *keys):
    return next((metrics[k] for k in keys if finite(metrics.get(k))), None)


def divide(numerator, *denominators):
    if not finite(numerator) or any(not finite(x) or x <= 0 for x in denominators):
        return None
    return numerator / math.prod(denominators)


def summarize_run(label, path, metadata):
    config = {"gpus": 4, "optimizer_steps_per_rollout": 4, "expected_rollouts": 50, **metadata}
    steps = config["optimizer_steps_per_rollout"]
    if (type(steps) is not int or steps < 1 or not finite(config["gpus"]) or config["gpus"] <= 0
            or type(config["expected_rollouts"]) is not int or config["expected_rollouts"] < 0):
        raise ValueError("invalid GPU, optimizer-step, or rollout count")
    rows, errors, duplicates, conflicts = {}, [], 0, []
    digest = hashlib.sha256()
    def row(index):
        return rows.setdefault(index, {"rollout_id": index, "metrics": {}, "events": [], "train_steps": [], "eval": []})
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            line = ANSI.sub("", raw.decode("utf-8", errors="replace"))
            match = EVENT.search(line)
            if not match:
                continue
            kind, index, payload = match.groups()
            try:
                metrics = literal_metrics(payload)
            except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
                errors.append({"line": number, "error": str(exc), "payload": payload[:1000]})
                continue
            index = int(index)
            target = row(index // steps if kind == "step" else index)
            stamp = STAMP.search(line)
            event = {"kind": kind, "logged_id": index, "line": number,
                     "timestamp": stamp[1] if stamp else None, "metrics": metrics}
            if kind == "step":
                same = [x for x in target["train_steps"] if x["logged_id"] == index]
                if any(x["metrics"] == metrics for x in same):
                    duplicates += 1
                    continue
                if same:
                    conflicts.append({"rollout_id": target["rollout_id"], "step": index, "line": number})
                target["train_steps"].append(event)
            elif kind == "eval":
                target["eval"].append(event)
            else:
                for key, value in metrics.items():
                    if key in target["metrics"] and target["metrics"][key] != value:
                        conflicts.append({"rollout_id": index, "key": key, "line": number})
                target["metrics"].update(metrics)
                target["events"].append(event)
    profiled = set(config.get("profiled_rollouts", []))
    profiled.update(int(s) // steps for s in config.get("profiled_optimizer_steps", []))
    for index, item in rows.items():
        metrics = item["metrics"]
        item["train_steps"].sort(key=lambda x: x["logged_id"])
        item["optimizer_steps_observed"] = len({x["logged_id"] for x in item["train_steps"]})
        item["training_stage_complete"] = finite(metrics.get("perf/train_time")) and item["optimizer_steps_observed"] == steps
        item["profiled"] = index in profiled if config.get("profiling_coverage_known") is True else None
        item["metric_conflict"] = any(c["rollout_id"] == index for c in conflicts)
        item["unprofiled_timing_eligible"] = (item["profiled"] is False and item["training_stage_complete"]
            and not item["metric_conflict"] and index not in config.get("exclude_timing_rollouts", [0]))
        item["reward_weight_phase"] = "pre_update_for_this_rollout; rollout 0 uses the initial policy in a fresh run"
        train_stamps = [x["timestamp"] for x in item["train_steps"] if x["timestamp"]]
        for evaluation in item["eval"]:
            timestamp = evaluation["timestamp"]
            evaluation["weight_phase"] = ("before_this_update" if timestamp and train_stamps and timestamp < min(train_stamps)
                                           else "after_this_update" if timestamp and train_stamps and timestamp > max(train_stamps)
                                           else "unknown")
        count = first(metrics, "rollout/num_training_samples")
        response = first(metrics, "rollout/response_lengths", "rollout/episode_response_length/mean", "rollout/response_len/mean")
        total = first(metrics, "rollout/total_lengths")
        output_tokens = count * response if count is not None and response is not None else None
        total_tokens = count * total if count is not None and total is not None else None
        item["common"] = {
            "training_reward_mean": first(metrics, "rollout/raw_reward", "rollout/episode_raw_reward"),
            "response_length_mean_tokens": response,
            "truncated_ratio": first(metrics, "rollout/truncated", "rollout/truncated_ratio"),
            "retained_samples": count, "retained_output_tokens_from_mean": output_tokens,
            "retained_input_output_tokens_from_mean": total_tokens,
            "output_tokens_per_gpu_generation_second": first(metrics, "perf/tokens_per_gpu_per_sec"),
            "retained_input_output_tokens_per_gpu_miles_step_second": divide(total_tokens, metrics.get("perf/step_time"), config["gpus"]),
            **{f"{name}_seconds": first(metrics, f"perf/{name}_time") for name in STAGES},
        }
        item["step_timer_identity_error_seconds"] = (metrics["perf/step_time"] - metrics["perf/train_wait_time"] - metrics["perf/train_time"]
            if all(finite(metrics.get(k)) for k in ["perf/step_time", "perf/train_wait_time", "perf/train_time"]) else None)
    ordered = [rows[k] for k in sorted(rows)]
    complete = {r["rollout_id"] for r in ordered if r["training_stage_complete"]}
    start = config.get("start_rollout_id", 0)
    wanted = set(range(start, config["expected_rollouts"]))
    statistics_by_stage = {}
    for stage in STAGES:
        values = [r["common"][f"{stage}_seconds"] for r in ordered if r["unprofiled_timing_eligible"] and finite(r["common"][f"{stage}_seconds"])]
        statistics_by_stage[stage] = {"count": len(values), "mean_seconds": statistics.mean(values) if values else None,
                                      "median_seconds": statistics.median(values) if values else None}
    return {"label": label, "framework": "Miles/SGLang", "source_log": str(Path(path).resolve()),
            "source_log_sha256": digest.hexdigest(), "metadata": config,
            "partial": config.get("status") != "SUCCEEDED" or not wanted.issubset(complete) or bool(errors or conflicts),
            "completed_training_rollouts": sorted(complete), "duplicate_train_lines_removed": duplicates,
            "held_out_eval_events": sum(len(r["eval"]) for r in ordered),
            "parse_errors": errors, "conflicts": conflicts, "rows": ordered,
            "unprofiled_stage_statistics": statistics_by_stage}


def legacy_baseline(path):
    source = json.loads(Path(path).read_text())
    gpus = source["configuration"]["num_nodes"] * source["configuration"]["gpus_per_node"]
    rows = []
    for step in source["steps"]:
        perf, tokens = step["performance"], step["derived_token_counts"]
        rows.append({"rollout_id": step["rollout_step"] - 1, "historical_step": step["rollout_step"], "raw": step,
                     "common": {"training_reward_mean": step["reward_mean"],
                                "response_length_mean_tokens": step["response_length_mean_tokens"],
                                "capacity_clip_ratio_not_engine_truncation": step["response_length_clip_ratio"],
                                "output_tokens_per_gpu_generation_second": divide(tokens["response_tokens"], perf.get("timing_s/gen"), gpus),
                                "input_output_tokens_per_gpu_verl_step_second": divide(tokens["total_input_plus_output_tokens"], perf.get("timing_s/step"), gpus),
                                "actor_train_seconds": perf.get("timing_s/update_actor"),
                                "log_probs_seconds": perf.get("timing_s/old_log_prob"),
                                "ref_log_probs_seconds": perf.get("timing_s/ref"),
                                "rollout_seconds": perf.get("timing_s/gen"), "step_seconds": perf.get("timing_s/step")}})
    return {"label": "historical_verl_gb300", "framework": "VeRL historical extraction", "source": str(path),
            "source_log_sha256": source["source_log_sha256"], "metadata": source["configuration"],
            "metric_semantics": source["metric_semantics"], "rows": rows,
            "held_out_eval_available": False, "profiling_coverage_known": False}


def report(runs, historical=None):
    def events(run):
        return [event for row in run["rows"] for group in ("events", "train_steps", "eval") for event in row[group]]
    keysets = [{key for event in events(run) for key in event["metrics"]} for run in runs]
    shared = sorted(set.intersection(*keysets)) if keysets else []
    config_fields = ["image", "versions", "graph", "kernels", "recipe"]
    return {"schema": "miles-qwen3-comparison-v1", "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "runs": runs, "historical_baseline": historical,
            "shared_raw_metric_count": len(shared), "shared_raw_metrics": shared,
            "shared_metric_observations": {key: {run["label"]: sum(key in e["metrics"] for e in events(run)) for run in runs} for key in shared},
            "configuration_comparison": {key: {run["label"]: run["metadata"].get(key, "UNKNOWN") for run in runs} for key in config_fields},
            "semantics": {
                "alignment": "Miles rollout 0 and historical VeRL step 1 are samples from the pre-update initial policy; optimizer steps are separate.",
                "train_timing": "Miles step_time = train_wait_time + train_time. Nested actor/log-prob/update timers must not be summed as a wall-time partition.",
                "generation_throughput": "Miles tokens_per_gpu_per_sec is retained response tokens / rollout_time / rollout GPUs. Dynamic filtering can exclude generated samples.",
                "legacy_throughput": "Historical VeRL throughput is input+output tokens / whole step seconds / 4 GPUs; never compare it directly to Miles generation throughput.",
                "common_formula_limit": "Mean lengths times retained sample count estimate token totals. Miles wait+train step and VeRL whole-step scopes differ; their rates remain separately named.",
                "learning": "Training rollout reward is not held-out accuracy. Evaluation events are retained separately, including pre/post-update events with the same ID.",
                "performance": "Unprofiled statistics require explicit profiling coverage and exclude rollout 0 by default. Stage-name correspondence alone does not prove identical work.",
                "configurations": "Image, library, graph and kernel differences are reported, not required to match. UNKNOWN values need provenance before causal GPU-speed claims.",
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=LOG")
    parser.add_argument("--metadata", type=Path, help="JSON mapping labels to run metadata; preserved verbatim with defaults")
    parser.add_argument("--verl-baseline", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    runs, labels = [], set()
    for value in args.run:
        label, separator, path = value.partition("=")
        if not separator or not label or label in labels:
            parser.error("--run needs a unique LABEL=LOG")
        labels.add(label)
        runs.append(summarize_run(label, path, metadata.get(label, {})))
    output = report(runs, legacy_baseline(args.verl_baseline) if args.verl_baseline else None)
    args.output.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": [{"label": x["label"], "rows": len(x["rows"]), "partial": x["partial"]} for x in runs]}))


if __name__ == "__main__":
    main()
