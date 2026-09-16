#!/usr/bin/env python3
"""Build an offline report from explicit, already collected evidence. No network/GPU access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PLOTLY = "plotly-3.1.0.min.js"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_evidence(path, name):
    if path is None or not path.exists():
        return None, {"section": name, "status": "pending", "path": str(path) if path else None}
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))
    if not isinstance(value, dict):
        raise ValueError(f"{name} input must be a JSON object: {path}")
    return value, {"section": name, "status": "loaded", "path": str(path.resolve()),
                   "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def copy_profile_assets(data, source, destination):
    """Only explicit local attachments are copied. Missing attachments remain pending."""
    if not data:
        return
    if not isinstance(data.get("profiles", []), list):
        raise ValueError("profiles.profiles must be a list")
    for index, item in enumerate(data.get("profiles", [])):
        if not isinstance(item, dict):
            raise ValueError("Each profile must be an object")
        item["attachments"] = {}
        for kind, extensions in [("image", {".png", ".jpg", ".jpeg", ".webp"}),
                                 ("trace", {".json", ".gz", ".zip", ".nsys-rep", ".sqlite", ".html"})]:
            original = item.get(kind)
            if not original:
                continue
            path = Path(original)
            if "://" in original:
                raise ValueError(f"Profile {kind} must be a local file, not a URL")
            if not path.is_absolute():
                path = source.parent / path
            attachment = {"source": str(path), "status": "pending"}
            if path.is_file():
                if path.suffix.lower() not in extensions:
                    raise ValueError(f"Unsupported {kind} attachment extension: {path.suffix}")
                digest = sha256(path)
                filename = f"profile-{index + 1}-{kind}-{digest[:12]}{path.suffix.lower()}"
                target = destination / "evidence" / filename
                target.parent.mkdir(exist_ok=True)
                if path.resolve() != target.resolve():
                    shutil.copyfile(path, target)
                attachment.update(status="available", url=f"evidence/{filename}", sha256=digest,
                                  bytes=path.stat().st_size)
            item["attachments"][kind] = attachment


PAIRED_STAGE_KEYS = ("rollout", "actor_train", "log_probs", "ref_log_probs", "update_weights")


def paired_timing(comparison, source_sha256=None):
    """Derive same-ID stage statistics without mutating original per-run evidence."""
    runs = (comparison or {}).get("runs", [])
    result = {"status": "pending", "reason": None, "comparison_sha256": source_sha256,
              "run_labels": [r.get("label") for r in runs], "rollout_ids": [], "count": 0,
              "statistics": {}, "stage_keys": list(PAIRED_STAGE_KEYS),
              "selection": "Intersection of completed, explicitly unprofiled, timing-eligible IDs; rollout0 and configured exclusions removed.",
              "step_difference_seconds_second_minus_first": None, "largest_observed_stage_gap": None}
    if len(runs) != 2:
        result["reason"] = "Exactly two supplied runs are required for a paired comparison."
        return result
    if len(set(result["run_labels"])) != 2:
        raise ValueError("Paired timing requires distinct run labels")
    selected = []
    for run in runs:
        complete = set(run.get("completed_training_rollouts", []))
        excluded = set(run.get("metadata", {}).get("exclude_timing_rollouts", []))
        eligible = {}
        for row in run.get("rows", []):
            index = row.get("rollout_id")
            if (row.get("training_stage_complete") is True and row.get("profiled") is False
                    and row.get("unprofiled_timing_eligible") is True
                    and index in complete and index not in excluded and index != 0):
                if type(index) is not int or index < 0 or index in eligible:
                    raise ValueError("Invalid or duplicate eligible rollout ID")
                eligible[index] = row
        selected.append(eligible)
    ids = sorted(set(selected[0]) & set(selected[1]))
    result.update(rollout_ids=ids, count=len(ids))
    if not ids:
        result["reason"] = "No shared completed, unprofiled timing-eligible rollout IDs yet."
        return result
    result["status"] = "available"
    # Missing/nonfinite values never silently shorten one run's metric cohort.
    for index, run in enumerate(runs):
        values = {}
        for key in ("step", *PAIRED_STAGE_KEYS):
            samples = [selected[index][i].get("common", {}).get(key + "_seconds") for i in ids]
            missing = [i for i, v in zip(ids, samples) if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)]
            values[key] = {"count": len(ids) if not missing else 0, "required_count": len(ids),
                           "missing_rollout_ids": missing,
                           "mean_seconds": statistics.mean(samples) if not missing else None,
                           "median_seconds": statistics.median(samples) if not missing else None}
        result["statistics"][run["label"]] = values
    first, second = (result["statistics"][r["label"]] for r in runs)
    for key in ("step", *PAIRED_STAGE_KEYS):
        if first[key]["mean_seconds"] is None or second[key]["mean_seconds"] is None:
            # A partially observed stage stays pending on both sides of its bar.
            first[key]["paired_metric_available"] = second[key]["paired_metric_available"] = False
            result["status"] = "partial"
        else:
            first[key]["paired_metric_available"] = second[key]["paired_metric_available"] = True
    if first["step"]["paired_metric_available"]:
        result["step_difference_seconds_second_minus_first"] = second["step"]["mean_seconds"] - first["step"]["mean_seconds"]
    gaps = [{"stage": key, "difference_seconds_second_minus_first": second[key]["mean_seconds"] - first[key]["mean_seconds"]}
            for key in PAIRED_STAGE_KEYS if first[key]["paired_metric_available"]]
    result["largest_observed_stage_gap"] = max(gaps, key=lambda x: abs(x["difference_seconds_second_minus_first"])) if gaps else None
    if result["status"] == "partial":
        result["reason"] = "At least one stage lacks values across the complete shared cohort; that stage is not plotted."
    return result


def build_report(runs=None, build=None, profiles=None, historical=None, output=None, run_health=None):
    output = (output or ROOT / "site").resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, evidence = {}, []
    for name, path in [("comparison", runs), ("build", build), ("profiles", profiles), ("historical", historical), ("run_health", run_health)]:
        value, record = read_evidence(path, name)
        inputs[name] = value
        evidence.append(record)
    comparison = inputs["comparison"]
    if comparison and comparison.get("schema") != "miles-qwen3-comparison-v1":
        raise ValueError("--runs must use schema miles-qwen3-comparison-v1")
    if comparison and not isinstance(comparison.get("runs"), list):
        raise ValueError("Comparison runs must be a list")
    health = inputs["run_health"]
    if health:
        if health.get("schema") != "miles-run-health-v1" or not isinstance(health.get("runs"), list):
            raise ValueError("--run-health must use schema miles-run-health-v1 with a runs list")
        if "comparison_sha256" in health:
            comparison_sha = next(record for record in evidence if record["section"] == "comparison").get("sha256")
            if comparison_sha is None or health["comparison_sha256"] != comparison_sha:
                raise ValueError("Run-health comparison_sha256 must match the supplied --runs file SHA256")
        identities = {(r.get("label"), r.get("metadata", {}).get("run_id")) for r in (comparison or {}).get("runs", [])}
        if any((r.get("label"), r.get("run_id")) not in identities for r in health["runs"]):
            raise ValueError("Run-health records must match a comparison run label and run_id")
    if not inputs["historical"] and comparison:
        inputs["historical"] = comparison.get("historical_baseline")
    copy_profile_assets(inputs["profiles"], profiles, output)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in [PLOTLY, "report.css", "report.js"]:
        src, dst = ROOT / "assets" / name, assets / name
        if not src.is_file():
            raise FileNotFoundError(f"Required offline asset missing: {src}")
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    envelope = {"schema": "rubin-gb300-offline-report-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "inputs": inputs, "provenance": evidence,
                "derived": {"paired_timing": paired_timing(comparison, next(r for r in evidence if r["section"] == "comparison").get("sha256"))},
                "plotly": {"version": "3.1.0", "source": "https://cdn.plot.ly/plotly-3.1.0.min.js",
                           "sha256": sha256(assets / PLOTLY)},
                "notice": "A static snapshot of supplied evidence. No live monitoring or remote reads."}
    payload = json.dumps(envelope, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # Safe inside an application/json script, including adversarial source strings.
    embedded = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (ROOT / "template.html").read_text()
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", embedded))
    (output / "report-data.json").write_text(json.dumps(envelope, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    build_data = inputs.get("build") or {}
    reproduction = build_data.get("reproduction") or {}
    image_reference = (build_data.get("image") or {}).get("reference")
    reproduction_lines = ["Rubin build reproduction entry points", ""]
    reproduction_lines.extend(f"{key}: {value}" for key, value in reproduction.items())
    reproduction_lines.extend(["", "Preserved image pull command:",
                              f"docker pull {image_reference}" if image_reference else "Image reference pending.",
                              "", "Report generation (input filenames are caller supplied):",
                              "python generate.py --runs comparison.json --build rubin-build-report.json "
                              "--profiles profiles.json --output site", "",
                              "Complete source pins, recorded checks, and hashes: report-data.json"])
    (output / "reproduction.txt").write_text("\n".join(reproduction_lines) + "\n")
    (output / "asset-manifest.json").write_text(json.dumps({
        "generated_at": envelope["generated_at"],
        "files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "asset-manifest.json"},
    }, indent=2) + "\n")
    return {"index": str(output / "index.html"), "inputs": evidence,
            "run_count": len(comparison["runs"]) if comparison else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, help="summarize_qwen3_runs.py output JSON")
    parser.add_argument("--build", type=Path, help="Rubin build report JSON")
    parser.add_argument("--profiles", type=Path, help="Profile metadata JSON with local attachments")
    parser.add_argument("--historical", type=Path, help="Optional raw VeRL baseline or normalized historical JSON")
    parser.add_argument("--run-health", type=Path, help="Optional explicit run issue metadata; never inferred from Ray status")
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    print(json.dumps(build_report(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
