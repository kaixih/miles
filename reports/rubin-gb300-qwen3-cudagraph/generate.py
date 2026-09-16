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


def validate_inputs(inputs, provenance):
    experiment = inputs["experiment"]
    if not experiment or experiment.get("schema") != "qwen3-cudagraph-experiment-v1":
        raise ValueError("An explicit qwen3-cudagraph-experiment-v1 manifest is required")
    experiment_id = experiment.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("Experiment identity is required")
    bindings = experiment.get("run_bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("run_bindings must be a list")
    identities = {(b.get("label"), b.get("run_id")) for b in bindings}
    if len(identities) != len(bindings) or any(not a or not b for a, b in identities):
        raise ValueError("Bindings need distinct, nonempty labels and run IDs")
    if len({b[0] for b in identities}) != len(identities):
        raise ValueError("Bind only one new run per platform label")
    previous_ids = set(experiment.get("previous_run_ids", []))
    if any(run_id in previous_ids for _, run_id in identities):
        raise ValueError("Previous eager run IDs cannot be bound as new experiment runs")
    comparison = inputs["comparison"]
    if comparison:
        if comparison.get("schema") != "miles-qwen3-comparison-v1" or not isinstance(comparison.get("runs"), list):
            raise ValueError("--runs must use miles-qwen3-comparison-v1 with a runs list")
        for run in comparison["runs"]:
            identity = (run.get("label"), run.get("metadata", {}).get("run_id"))
            if identity not in identities or identity[1] in previous_ids:
                raise ValueError("Every supplied run must match a NEW exact experiment binding")
        if len({r.get("label") for r in comparison["runs"]}) != len(comparison["runs"]):
            raise ValueError("Duplicate comparison labels")
    health = inputs["run_health"]
    if health:
        if health.get("schema") != "miles-run-health-v1" or not isinstance(health.get("runs"), list):
            raise ValueError("Invalid run-health schema")
        actual = {(r.get("label"), r.get("metadata", {}).get("run_id")) for r in (comparison or {}).get("runs", [])}
        if any((r.get("label"), r.get("run_id")) not in actual for r in health["runs"]):
            raise ValueError("Run-health records must match supplied run identities")
        if "comparison_sha256" in health:
            digest = next(p for p in provenance if p["section"] == "comparison").get("sha256")
            if not digest or health["comparison_sha256"] != digest:
                raise ValueError("Run-health comparison SHA256 mismatch")
    for name in ["profiles", "diagnostics"]:
        data = inputs[name]
        if data and data.get("experiment_id") != experiment_id:
            raise ValueError(f"{name} experiment_id must match this new experiment")
    profiles = inputs["profiles"] or {}
    if profiles and profiles.get("schema") != "qwen3-cudagraph-profiles-v1":
        raise ValueError("Profiles must use qwen3-cudagraph-profiles-v1")
    profile_slots = set()
    for item in profiles.get("profiles", []):
        if (item.get("run_label"), item.get("source_run_id")) not in identities:
            raise ValueError("Profile source_run_id must bind to this experiment's platform run")
        if item.get("stage") not in ("prefill", "decode"):
            raise ValueError("Profile stage must explicitly be prefill or decode")
        for mode in ("decode_graph", "prefill_graph"):
            if mode in item and type(item[mode]) is not bool:
                raise ValueError("Profile graph condition must be a boolean when supplied")
        slot = (item["run_label"], item["stage"])
        if slot in profile_slots:
            raise ValueError("Duplicate profile platform/stage: explicitly select one primary capture")
        profile_slots.add(slot)
    for proof in profiles.get("graph_evidence", []):
        if (proof.get("run_label"), proof.get("source_run_id")) not in identities:
            raise ValueError("Graph proof must bind to this experiment's platform run")
        for field in ["decode_forwards", "decode_graph_replays", "decode_fallbacks", "decode_unknown", "prefill_forwards", "prefill_graph_replays"]:
            value = proof.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Graph evidence {field} must be a nonnegative count")
        counts = [proof.get(field) for field in ["decode_forwards", "decode_graph_replays", "decode_fallbacks", "decode_unknown"]]
        if all(value is not None for value in counts) and counts[0] != sum(counts[1:]):
            raise ValueError("Decode graph counts do not reconcile")
        if proof.get("verified") is True and not proof.get("evidence_refs"):
            raise ValueError("Verified graph counts require explicit evidence references")
    diagnostics = inputs["diagnostics"] or {}
    if diagnostics and diagnostics.get("schema") != "qwen3-cudagraph-diagnostics-v1":
        raise ValueError("Diagnostics must use qwen3-cudagraph-diagnostics-v1")
    for pair in diagnostics.get("pairs", []):
        if (pair.get("platform"), pair.get("source_run_id")) not in identities:
            raise ValueError("Diagnostic pair must bind to a new platform run")
        if pair.get("verified") is not True:
            continue
        off, on = pair.get("off", {}), pair.get("on", {})
        for field in ["image_digest", "initial_model_id", "request_sha256", "context_sha256", "cache_policy", "prefill_graph"]:
            if field not in off or off.get(field) != on.get(field):
                raise ValueError(f"Verified OFF/ON diagnostic mismatch: {field}")
        if off.get("decode_graph") is not False or on.get("decode_graph") is not True:
            raise ValueError("Diagnostic graph modes must be explicitly OFF and ON")
        if not pair.get("evidence_refs"):
            raise ValueError("Verified diagnostic pair needs evidence references")
        for condition in [off, on]:
            if condition.get("timing_scope") != "http_request_prefill_decode_queue_response":
                raise ValueError("Verified diagnostic requires explicit full HTTP prefill+decode timing_scope")
            if "decode_ms" in condition:
                raise ValueError("decode_ms belongs to trace evidence, not HTTP generation timing")
            values = condition.get("generation_seconds", [])
            if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Verified diagnostic needs finite positive raw generation_seconds samples")
            condition["generation_seconds_mean"] = statistics.mean(values)
            condition["generation_seconds_median"] = statistics.median(values)
            condition["sample_count"] = len(values)


def verify_refs(records, source, output):
    """Copy explicitly referenced small receipts, with their supplied hashes checked."""
    for record in records:
        if record.get("verified") is not True:
            continue
        checked = []
        for ref in record.get("evidence_refs", []):
            if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or ":" in ref["path"]:
                raise ValueError("Evidence reference must be a local path plus SHA256")
            path = Path(ref["path"])
            if not path.is_absolute():
                path = source.parent / path
            if not path.is_file() or sha256(path) != ref.get("sha256"):
                raise ValueError("Evidence reference file missing or SHA256 mismatched")
            digest = ref["sha256"]
            target = output / "evidence" / ("receipt-" + digest[:16] + path.suffix)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            checked.append({"source": str(path.resolve()), "sha256": digest, "url": str(target.relative_to(output))})
        record["verified_receipts"] = checked


def build_report(experiment=None, runs=None, profiles=None, diagnostics=None, run_health=None, output=None):
    output = (output or ROOT / "site").resolve()
    old_report = ROOT.parent / "rubin-gb300-qwen3"
    if output == old_report or old_report in output.parents:
        raise ValueError("Refuse to overwrite the preserved original report")
    inputs, provenance = {}, []
    for name, path in [("experiment", experiment or ROOT / "experiment.json"), ("comparison", runs),
                       ("profiles", profiles), ("diagnostics", diagnostics), ("run_health", run_health)]:
        value, record = read_evidence(path, name)
        inputs[name] = value
        provenance.append(record)
    validate_inputs(inputs, provenance)
    output.mkdir(parents=True, exist_ok=True)
    verify_refs((inputs["profiles"] or {}).get("graph_evidence", []), profiles, output)
    verify_refs((inputs["diagnostics"] or {}).get("pairs", []), diagnostics, output)
    copy_profile_assets(inputs["profiles"], profiles, output)
    for item in (inputs["profiles"] or {}).get("profiles", []):
        trace = item.get("attachments", {}).get("trace", {})
        expected = item.get("source_trace_sha256")
        if trace.get("status") == "available" and expected and trace["sha256"] != expected:
            raise ValueError("Profile trace does not match its declared source SHA256")
        if item.get("verified") is True and (trace.get("status") != "available" or not expected):
            raise ValueError("Verified profile requires a local trace and matching source_trace_sha256")
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    for name in [PLOTLY, "report.css", "report.js", "navigation.js"]:
        shutil.copyfile(ROOT / "assets" / name, assets / name)
    comparison_sha = next(p for p in provenance if p["section"] == "comparison").get("sha256")
    envelope = {"schema": "qwen3-cudagraph-offline-slides-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
                "inputs": inputs, "provenance": provenance,
                "derived": {"paired_timing": paired_timing(inputs["comparison"], comparison_sha)},
                "plotly": {"version": "3.1.0", "sha256": sha256(assets / PLOTLY)},
                "notice": "New experiment only. Missing measurements remain pending. No old curves imported."}
    raw = json.dumps(envelope, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    embedded = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (output / "index.html").write_text((ROOT / "template.html").read_text().replace("__REPORT_DATA__", embedded))
    (output / "report-data.json").write_text(json.dumps(envelope, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    (output / "asset-manifest.json").write_text(json.dumps({"generated_at": envelope["generated_at"],
        "files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name != "asset-manifest.json"}}, indent=2) + "\n")
    return {"index": str(output / "index.html"), "run_count": len((inputs["comparison"] or {}).get("runs", [])),
            "experiment_id": inputs["experiment"]["experiment_id"], "inputs": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "experiment.json")
    parser.add_argument("--runs", type=Path, help="NEW comparison JSON, exact run identities bound in experiment.json")
    parser.add_argument("--profiles", type=Path, help="New graph/profile evidence with matching experiment_id")
    parser.add_argument("--diagnostics", type=Path, help="Separate generation-only OFF/ON diagnostics")
    parser.add_argument("--run-health", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    print(json.dumps(build_report(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
