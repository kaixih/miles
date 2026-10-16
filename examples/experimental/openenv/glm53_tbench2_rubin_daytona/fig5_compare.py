#!/usr/bin/env python3
"""Overlay two Figure-5 series (fig5_collect.py JSON outputs) on one 1x3 figure and print a summary table.

  python3 fig5_compare.py --series Rubin=fig5-series-rubin.json --series GB300=fig5-series-gb300.json \
      --png fig5-compare.png [--out fig5-compare.json]

Panels, as in fig5_collect.py: (a) wall-clock per training step (step 0 clipped), (b) mean |trainer - rollout|
log-probability at sampled tokens, (c) raw task reward per step with its 9-step moving average.
"""

import argparse
import json
import statistics


def load(label_path):
    label, path = label_path.split("=", 1)
    data = json.load(open(path))
    return label, data["steps"], data.get("summary", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", required=True, help="LABEL=path/to/fig5-series.json (repeatable)")
    ap.add_argument("--png", required=True)
    ap.add_argument("--out", help="optional JSON with the per-series summary numbers")
    args = ap.parse_args()
    series = [load(s) for s in args.series]

    summary = {}
    for label, rows, _ in series:
        walls = [r["step_wall_s"] for r in rows if r.get("step_wall_s") is not None and r["step"] >= 1]
        steady = [r["step_wall_s"] for r in rows if r.get("step_wall_s") is not None and r["step"] >= 5]
        divs = [r["train.train_rollout_logprob_abs_diff"] for r in rows if r.get("train.train_rollout_logprob_abs_diff") is not None]
        rewards = [r["perf.episode_raw_reward"] for r in rows if r.get("perf.episode_raw_reward") is not None]
        ma = [r["raw_reward_ma9"] for r in rows if r.get("raw_reward_ma9") is not None]
        summary[label] = {
            "steps": len(rows),
            "step0_wall_s": next((r["step_wall_s"] for r in rows if r["step"] == 0 and r.get("step_wall_s") is not None), None),
            "step_wall_median_s_steps_1_on": statistics.median(walls) if walls else None,
            "step_wall_median_s_steps_5_on": statistics.median(steady) if steady else None,
            "logprob_abs_diff_mean": statistics.fmean(divs) if divs else None,
            "logprob_abs_diff_first_last": [divs[0], divs[-1]] if divs else None,
            "raw_reward_mean": statistics.fmean(rewards) if rewards else None,
            "raw_reward_ma9_first_last": [ma[0], ma[-1]] if ma else None,
        }
    header = f"{'':28}" + "".join(f"{label:>14}" for label, _, _ in series)
    print(header)
    for key in ("steps", "step0_wall_s", "step_wall_median_s_steps_1_on", "step_wall_median_s_steps_5_on",
                "logprob_abs_diff_mean", "raw_reward_mean"):
        print(f"{key:28}" + "".join(f"{(summary[label][key] if summary[label][key] is not None else float('nan')):>14.4g}" for label, _, _ in series))
    if args.out:
        json.dump({"series": {label: path.split("=", 1)[1] for label, path in zip([s[0] for s in series], args.series)}, "summary": summary},
                  open(args.out, "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for label, rows, _ in series:
        wall = [(r["step"], r["step_wall_s"]) for r in rows if r.get("step_wall_s") is not None and r["step"] >= 1]
        if wall:
            med = statistics.median([w[1] for w in wall])
            axes[0].plot([w[0] for w in wall], [w[1] for w in wall], marker="o", label=f"{label} (median {med:.0f} s)")
        dv = [(r["step"], r["train.train_rollout_logprob_abs_diff"]) for r in rows if r.get("train.train_rollout_logprob_abs_diff") is not None]
        if dv:
            axes[1].plot([d[0] for d in dv], [d[1] for d in dv], marker="o", label=f"{label} (mean {statistics.fmean([d[1] for d in dv]):.4f})")
        rw = [(r["step"], r["perf.episode_raw_reward"], r["raw_reward_ma9"]) for r in rows if r.get("perf.episode_raw_reward") is not None]
        if rw:
            line, = axes[2].plot([x[0] for x in rw], [x[2] for x in rw], lw=2.5, label=f"{label} 9-step moving avg")
            axes[2].plot([x[0] for x in rw], [x[1] for x in rw], alpha=0.3, color=line.get_color())
    axes[0].set_title("(a) wall-clock per training step (step 0 clipped)"); axes[0].set_xlabel("step"); axes[0].set_ylabel("seconds"); axes[0].legend()
    axes[1].set_title("(b) |trainer - rollout| logprob at sampled tokens"); axes[1].set_xlabel("step"); axes[1].legend()
    axes[2].set_title("(c) raw task reward (faint: per step)"); axes[2].set_xlabel("step"); axes[2].set_ylim(0, 1); axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.png, dpi=130)
    print("wrote", args.png)


if __name__ == "__main__":
    main()
