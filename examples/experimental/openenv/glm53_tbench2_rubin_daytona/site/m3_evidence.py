#!/usr/bin/env python3
"""Extract per-update evidence from one node's dump_details (events + dumped samples), as JSON.

Run inside a run container (the head and every trainer node write their own event files):

    python3 m3_evidence.py --dump /work/output/<run-id>/dump_details --out /work/output/<run-id>/evidence-<node>.json

Per rollout id it reports: trainer metric events (train/loss, train/grad_norm, finiteness) by rank,
train_group_step_end controller events, inference-engine weight checksum events (both engines equal,
changed from the previous rollout), rollout-executor metrics (samples, raw reward, aborted/stale
groups, weight-version coverage), eval metrics, and the weight versions recorded in dumped samples.
Missing evidence is reported as absent, never inferred.
"""

import argparse
import json
import math
import socket
from collections import defaultdict
from pathlib import Path


def load_events(events_dir):
    rows = []
    for path in sorted(Path(events_dir).glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_file"] = path.name
            rows.append(row)
    return rows


def finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def sample_weight_versions(rollout_data_dir):
    """rollout_id -> {distinct weight versions, samples, rewards} from dumped batches (torch needed)."""
    out = {}
    paths = sorted(Path(rollout_data_dir).glob("*.pt")) if Path(rollout_data_dir).is_dir() else []
    if not paths:
        return out
    import torch

    for path in paths:
        d = torch.load(path, map_location="cpu", weights_only=False)
        samples = d["samples"] if isinstance(d, dict) else d
        if samples and isinstance(samples[0], list):
            samples = [s for group in samples for s in group]
        versions, rewards, raw = set(), [], []
        tito_mismatch = 0
        for s in samples:
            wv = s.weight_versions if hasattr(s, "weight_versions") else s.get("weight_versions")
            for group in wv or []:
                for seg in group:
                    versions.add(str(seg.get("version")))
            rewards.append(s.reward if hasattr(s, "reward") else s.get("reward"))
            md = s.metadata if hasattr(s, "metadata") else s.get("metadata", {})
            raw.append((md or {}).get("reward"))
            tito_mismatch += bool((md or {}).get("tito_session_mismatch"))
        out[path.stem] = {  # keyed by file (eval_<id>.pt and <id>.pt share rollout ids)
            "file": path.name, "rollout_id": d.get("rollout_id") if isinstance(d, dict) else None,
            "samples": len(samples), "weight_versions": sorted(versions),
            "reward_sum": sum(r for r in rewards if isinstance(r, (int, float))),
            "raw_reward_null": sum(1 for r in raw if r is None), "samples_with_tito_mismatch": tito_mismatch,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = load_events(args.dump / "events")
    by_rollout = defaultdict(lambda: {"train_metrics": {}, "controller": [], "weight_checksums": [], "rollout_metrics": [], "eval_metrics": []})
    other_types = defaultdict(int)
    for row in rows:
        t = row.get("type")
        rid = row.get("rollout_id")
        src = row.get("source") or {}
        if t == "metric":
            m = row.get("metrics") or {}
            if "train/loss" in m or "train/grad_norm" in m:
                by_rollout[rid]["train_metrics"][src.get("rank_within_cell")] = {
                    "timestamp": row.get("timestamp"), "attempt": row.get("attempt"),
                    "loss": m.get("train/loss"), "grad_norm": m.get("train/grad_norm"), "step": m.get("train/step"),
                    "all_finite": finite([v for v in m.values() if isinstance(v, (int, float))]),
                }
            elif any(k.startswith("eval/") for k in m):
                by_rollout[rid]["eval_metrics"].append({"timestamp": row.get("timestamp"), "metrics": m})
            elif src.get("component") == "rollout_executor":
                by_rollout[rid]["rollout_metrics"].append({"timestamp": row.get("timestamp"), "metrics": {
                    k: m[k] for k in m if k in ("rollout/num_training_samples", "rollout/raw_reward_unfiltered", "rollout/episode_raw_reward",
                                                  "rollout/fully_async/aborted_groups_filtered", "rollout/fully_async/stale_groups_filtered",
                                                  "rollout/fully_async/weight_version_sample_coverage", "rollout/fully_async/queue_size",
                                                  "rollout/truncated_ratio", "rollout/step")}})
        elif t == "train_group_step_end":
            by_rollout[rid]["controller"].append({k: row.get(k) for k in ("timestamp", "cell_outcomes", "attempt", "trainer_model_id") if k in row})
        elif t == "inference_engine_weight_checksum":
            by_rollout[rid]["weight_checksums"].append({k: row.get(k) for k in row if k not in ("_file",)})
        else:
            other_types[t] += 1

    checks = {rid: v["weight_checksums"] for rid, v in by_rollout.items() if v["weight_checksums"]}
    updates = []
    for rid in sorted(k for k in by_rollout if isinstance(k, int)):
        v = by_rollout[rid]
        ranks = v["train_metrics"]
        entry = {
            "rollout_id": rid,
            "train_metric_ranks": sorted(r for r in ranks if r is not None),
            "loss": next((m["loss"] for m in ranks.values() if m["loss"] is not None), None),
            "grad_norm": next((m["grad_norm"] for m in ranks.values() if m["grad_norm"] is not None), None),
            "metrics_finite": all(m["all_finite"] for m in ranks.values()) if ranks else None,
            "controller_events": v["controller"],
            "weight_checksum_events": len(v["weight_checksums"]),
            "rollout_metrics": v["rollout_metrics"],
            "eval_metrics": v["eval_metrics"],
        }
        if v["weight_checksums"]:
            cur = v["weight_checksums"][-1]
            prev = checks.get(rid - 1, [None])[-1] if rid - 1 in checks else None
            entry["weight_checksum"] = {"current": cur.get("engine_checksums") or cur.get("checksum"),
                                        "changed_from_previous": (prev is not None and (cur.get("engine_checksums") or cur.get("checksum")) != (prev.get("engine_checksums") or prev.get("checksum")))}
        updates.append(entry)
    report = {
        "node": socket.gethostname(), "dump": str(args.dump), "event_rows": len(rows), "other_event_types": dict(other_types),
        "updates": updates, "dumped_samples": sample_weight_versions(args.dump / "rollout_data"),
        "limits": ["one node's events only; trainer ranks on other nodes write their own files",
                   "absent evidence is unobserved, not failure"],
    }
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("EVIDENCE=" + json.dumps({"node": report["node"], "rollouts": [u["rollout_id"] for u in updates],
                                   "ranks_with_train_metrics": {u["rollout_id"]: len(u["train_metric_ranks"]) for u in updates},
                                   "dumped": {k: (v["samples"], v["weight_versions"]) for k, v in report["dumped_samples"].items()}}, default=str))


if __name__ == "__main__":
    main()
