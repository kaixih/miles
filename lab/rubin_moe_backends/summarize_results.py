#!/usr/bin/env python3
"""Summarize two-rollout MoE experiments without modifying presentation artifacts."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trtllm-retry-root", type=Path,
                        help="Optional separate PR33743 + post-load fix attempt")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "metrics", args.source / "lab/rubin_two_node/summarize_qwen3_runs.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    baseline = json.loads(args.baseline.read_text())
    reference = {row["rollout_id"]: row for row in baseline["rows"]}
    result = {
        "baseline_log_sha256": baseline["source_log_sha256"],
        "scope": "Two rollouts starting from fresh initial weights per backend; decode graph ON.",
        "limitations": [
            "Two rollouts check execution and early numerical sanity, not convergence.",
            "Historical Triton timings were collected on a different allocation.",
            "Sampling and scheduling can change response lengths; compare tokens/GPU/s too.",
        ],
        "backends": {},
    }
    lines = [
        "# Rubin BF16 MoE: two-rollout comparison", "",
        "Historical Triton comparison uses its first two rollouts from the same initial policy.",
        "Two rollouts are preliminary evidence; no convergence or kernel-level speedup is inferred.", "",
        "| Backend | Rollout | Generation (s) | Response tokens | Tokens/GPU/s | Throughput vs Triton | Reward | Truncation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def table_row(name, row, relative="—"):
        common = row["common"]
        def fmt(value, precision=2):
            return f"{value:.{precision}f}" if metrics.finite(value) else "unavailable"
        return (
            f"| {name} | {row['rollout_id']} | {fmt(common['rollout_seconds'])} | "
            f"{fmt(common['retained_output_tokens_from_mean'], 0)} | "
            f"{fmt(common['output_tokens_per_gpu_generation_second'])} | {relative} | "
            f"{fmt(common['training_reward_mean'], 4)} | {fmt(common['truncated_ratio'], 4)} |"
        )

    for row in baseline["rows"]:
        lines.append(table_row("Triton (historical)", row))
    candidates = [(name, args.root / name) for name in
                  ("flashinfer_cutlass", "flashinfer_trtllm")]
    if args.trtllm_retry_root:
        candidates.append(("flashinfer_trtllm + post-load fix",
                           args.trtllm_retry_root / "flashinfer_trtllm"))
    for backend, directory in candidates:
        receipt_path = directory / "result.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"status": "NOT_COMPLETED"}
        entry = {"receipt": receipt}
        log = directory / "logs/qwen3_train.log"
        if log.exists():
            summary = metrics.summarize_run(backend, log, {
                "expected_rollouts": 2, "status": receipt.get("status"),
                "profiling_coverage_known": True, "profiled_rollouts": [],
            })
            entry["summary"] = summary
            entry["relative_throughput"] = {}
            for row in summary["rows"]:
                index = row["rollout_id"]
                if index not in reference:
                    continue
                measured = row["common"]["output_tokens_per_gpu_generation_second"]
                base = reference[index]["common"]["output_tokens_per_gpu_generation_second"]
                ratio = measured / base if metrics.finite(measured) and base > 0 else None
                entry["relative_throughput"][str(index)] = ratio
                lines.append(table_row(backend, row, f"{ratio:.3f}×" if ratio is not None else "unavailable"))
        result["backends"][backend] = entry
    lines += ["", "## Completion", ""]
    for backend, entry in result["backends"].items():
        receipt = entry["receipt"]
        lines.append(f"- {backend}: {receipt.get('status', 'UNKNOWN')}. {receipt.get('error', '')}")
    if args.trtllm_retry_root:
        lines += ["", "The unmodified PR33743 attempt and the additional post-load fix "
                  "are retained separately. PR33743 alone failed during the initial weight "
                  "update: the final post-load hook tried to pack already blocked weights "
                  "again (FlashInfer expected a 2D expert tile and received 3D)."]
    lines += ["", "## Whole-step timing boundary", "",
              "Generation improvements do not establish a full training-step speedup.", "",
              "| Backend, rollout 1 | Generation (s) | Actor update (s) | Weight sync (s) | Full step (s) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    stage_rows = [("Triton (historical)", reference[1])]
    for backend, entry in result["backends"].items():
        stage_rows += [(backend, row) for row in entry.get("summary", {}).get("rows", [])
                       if row["rollout_id"] == 1]
    for backend, row in stage_rows:
        common = row["common"]
        values = [common.get(key) for key in ("rollout_seconds", "actor_train_seconds",
                                            "update_weights_seconds", "step_seconds")]
        cells = [f"{value:.2f}" if metrics.finite(value) else "unavailable" for value in values]
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    if args.trtllm_retry_root:
        lines += ["", "The corrected TRTLLM run had a slower actor update and weight sync "
                  "in this short comparison. Its generation gain did not translate into "
                  "a lower full-step time. Actor timing variation needs separate investigation; "
                  "this experiment changed the rollout MoE path, not the training kernels."]
    lines += ["", "Both candidates use fresh initial weights, two rollouts, eight optimizer updates, "
              "initial evaluation, and no checkpoint saving. Compare rollout 1 after warmup; "
              "retain rollout 0 as supporting evidence. These are whole-generation measurements, "
              "not isolated MoE kernel timings.", ""]
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    print(args.root / "RESULTS.md")


if __name__ == "__main__":
    main()
