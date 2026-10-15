#!/usr/bin/env python3
"""Collect Figure-5-style per-step series from a Miles driver log (arXiv 2609.08368 §9, Fig. 5).

  (a) wall-clock seconds per training step   : time between successive "Timer update_weights end" lines
      (a step = train on batch N, broadcast); step 0 is the warm-up and is reported but flagged
  (b) rollout-vs-trainer log-prob divergence  : train/train_rollout_logprob_abs_diff per step (+ train_rollout_kl, tis, ess)
  (c) raw task reward per step                : rollout/episode_raw_reward from the rollout-executor perf line,
                                                 with a 9-step moving average
  extras: perf/rollout_time, perf/tokens_per_gpu_per_sec, staleness, weight-version coverage, truncation,
          TITO mismatch rate, loss, grad_norm, log_probs/actor_train/update_weights timer elapsed values.

  python3 fig5_collect.py --log <driver.log> --out <series.json> [--csv <series.csv>] [--png <fig5.png>]
"""

import argparse
import ast
import csv
import json
import re
import statistics
from datetime import datetime

TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
PERF = re.compile(r"metrics\.py:\d+ - perf (\d+): (\{.*\})\s*$")
STEP = re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})\s*$")
TIMER = re.compile(r"Timer (log_probs|actor_train|update_weights) end \(elapsed: ([0-9.]+)s\)")
COLLECTED = re.compile(r"Final collected (\d+) samples")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
KEEP_PERF = ("rollout/episode_raw_reward", "rollout/num_training_samples", "rollout/fully_async/aborted_groups_filtered",
             "rollout/fully_async/stale_groups_filtered", "rollout/fully_async/avg_staleness", "rollout/fully_async/max_staleness",
             "rollout/fully_async/weight_version_sample_coverage", "rollout/weight_version/mean", "rollout/weight_version/mixed_version_ratio",
             "rollout/truncated_ratio", "rollout/response_len/mean", "rollout/response_len/max", "rollout/prefix_cache_hit_rate",
             "rollout/tito_session_mismatch_rate/v1", "rollout/tito_session_mismatch_rate/v1/assistant_text",
             "perf/rollout_time", "perf/tokens_per_gpu_per_sec", "perf/effective_tokens_per_gpu_per_sec", "perf/non_generation_time/mean")
KEEP_STEP = ("train/loss", "train/grad_norm", "train/train_rollout_logprob_abs_diff", "train/train_rollout_kl", "train/tis",
             "train/tis_clipfrac", "train/ess_ratio", "train/pg_clipfrac", "train/ppo_kl")


def parse_time(line):
    m = TS.search(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--png")
    args = ap.parse_args()

    steps = {}
    broadcasts, timers = [], {"log_probs": [], "actor_train": [], "update_weights": []}
    collected = []
    with open(args.log, errors="replace") as fh:
        for raw in fh:
            line = ANSI.sub("", raw.rstrip("\n"))
            ts = parse_time(line)
            if (m := PERF.search(line)):
                d = ast.literal_eval(m.group(2))
                steps.setdefault(int(m.group(1)), {})["perf"] = {k: d.get(k) for k in KEEP_PERF if k in d}
                steps[int(m.group(1))]["collected_at"] = ts.isoformat() if ts else None
            elif (m := STEP.search(line)):
                d = ast.literal_eval(m.group(2))
                steps.setdefault(int(m.group(1)), {})["train"] = {k: d.get(k) for k in KEEP_STEP if k in d}
                steps[int(m.group(1))]["train_metrics_at"] = ts.isoformat() if ts else None
            elif (m := TIMER.search(line)) and "rank00000]" in line:
                timers[m.group(1)].append({"at": ts.isoformat() if ts else None, "elapsed_s": float(m.group(2))})
                if m.group(1) == "update_weights" and ts:
                    broadcasts.append(ts)
            elif (m := COLLECTED.search(line)) and ts:
                collected.append({"at": ts.isoformat(), "samples": int(m.group(1))})

    # broadcasts[0] is the initial HF broadcast; broadcasts[i] (i>=1) closes training step i-1.
    rows = []
    for i in sorted(steps):
        row = {"step": i}
        row.update({f"perf.{k.split('/')[-1] if k.startswith('perf/') else k.replace('rollout/', '')}": v for k, v in steps[i].get("perf", {}).items()})
        row.update({k.replace("train/", "train."): v for k, v in steps[i].get("train", {}).items()})
        if i + 1 < len(broadcasts) and i < len(broadcasts):
            row["step_wall_s"] = (broadcasts[i + 1] - broadcasts[i]).total_seconds()
        for name in ("log_probs", "actor_train", "update_weights"):
            idx = i + (1 if name == "update_weights" else 0)
            if idx < len(timers[name]):
                row[f"{name}_elapsed_s"] = timers[name][idx]["elapsed_s"]
        rows.append(row)
    rewards = [r.get("perf.episode_raw_reward") for r in rows]
    for k, r in enumerate(rows):
        window = [x for x in rewards[max(0, k - 8):k + 1] if x is not None]
        r["raw_reward_ma9"] = sum(window) / len(window) if window else None
    walls = [r["step_wall_s"] for r in rows if r.get("step_wall_s") is not None and r["step"] >= 1]
    divs = [r.get("train.train_rollout_logprob_abs_diff") for r in rows if r.get("train.train_rollout_logprob_abs_diff") is not None]
    summary = {
        "steps_with_train_metrics": len(divs), "steps_with_wall_time": len(walls),
        "step_wall_s_median_excluding_step0": statistics.median(walls) if walls else None,
        "step0_wall_s": next((r["step_wall_s"] for r in rows if r["step"] == 0 and r.get("step_wall_s") is not None), None),
        "logprob_abs_diff_mean": statistics.fmean(divs) if divs else None,
        "raw_reward_first": rewards[0] if rewards else None, "raw_reward_last": rewards[-1] if rewards else None,
        "raw_reward_ma9_first": rows[0]["raw_reward_ma9"] if rows else None, "raw_reward_ma9_last": rows[-1]["raw_reward_ma9"] if rows else None,
        "initial_broadcast_at": broadcasts[0].isoformat() if broadcasts else None, "broadcasts": len(broadcasts),
        "samples_collected_events": len(collected),
    }
    out = {"source_log": args.log, "summary": summary, "steps": rows,
           "definitions": {"step_wall_s": "seconds between the broadcast that closed the previous step and the one that closed this step (Fig. 5a)",
                           "train.train_rollout_logprob_abs_diff": "mean |trainer logprob - rollout logprob| at sampled tokens (Fig. 5b)",
                           "perf.episode_raw_reward": "mean raw task reward of the 64 trajectories trained at this step (Fig. 5c); raw_reward_ma9 = 9-step moving average"}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    if args.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        xs = [r["step"] for r in rows]
        wall = [(r["step"], r["step_wall_s"]) for r in rows if r.get("step_wall_s") is not None and r["step"] >= 1]
        if wall:
            axes[0].plot([w[0] for w in wall], [w[1] for w in wall], marker="o")
            axes[0].axhline(statistics.median([w[1] for w in wall]), ls="--", c="gray", label=f"median {statistics.median([w[1] for w in wall]):.0f} s")
            axes[0].legend()
        axes[0].set_title("(a) wall-clock per training step (step 0 clipped)"); axes[0].set_xlabel("step"); axes[0].set_ylabel("seconds")
        dv = [(r["step"], r["train.train_rollout_logprob_abs_diff"]) for r in rows if r.get("train.train_rollout_logprob_abs_diff") is not None]
        if dv:
            axes[1].plot([d[0] for d in dv], [d[1] for d in dv], marker="o")
            axes[1].axhline(statistics.fmean([d[1] for d in dv]), ls="--", c="gray", label=f"mean {statistics.fmean([d[1] for d in dv]):.4f}")
            axes[1].legend()
        axes[1].set_title("(b) |trainer - rollout| logprob at sampled tokens"); axes[1].set_xlabel("step")
        rw = [(r["step"], r["perf.episode_raw_reward"], r["raw_reward_ma9"]) for r in rows if r.get("perf.episode_raw_reward") is not None]
        if rw:
            axes[2].plot([x[0] for x in rw], [x[1] for x in rw], alpha=0.35, label="per step")
            axes[2].plot([x[0] for x in rw], [x[2] for x in rw], lw=2.5, label="9-step moving avg")
            axes[2].legend()
        axes[2].set_title("(c) raw task reward"); axes[2].set_xlabel("step"); axes[2].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
