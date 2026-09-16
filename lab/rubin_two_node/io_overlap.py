"""Pure, conservative I/O timing exclusions for summarize_qwen3_runs rows.

The training perf timestamp is log emission, after the train timer and CPU weight
backup. step_time is train_wait + train, not a collection of additive sub-timers.
Reconstruct [last training-perf timestamp - step_time, that timestamp], then also
guard the immediately preceding rollout's interval. This intentionally excludes
one extra boundary interval; it does not claim precise GPU execution boundaries.
Naive Miles log timestamps require the caller's explicit UTC provenance. No I/O,
network, mutation, or inference from rollout-generation perf timestamps occurs.
"""

import datetime as dt
import math

UTC = dt.timezone.utc


def _stamp(value, *, naive_utc=False):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        if not naive_utc:
            raise ValueError("timestamp has no timezone")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("missing or invalid step_seconds")
    return value


def assess_io_overlap(rows, platform, windows, snapshot_utc, *, log_timestamps_are_utc=False):
    """Return row assessments and exclusion IDs without changing caller objects.

windows contain platform, operation, start_utc, end_utc (null while active), and
status. An active window is only observed through snapshot_utc: later row coverage
is unknown, never silently eligible. Exclusions must be UNIONED with existing
warmup/checkpoint/profile exclusions; this helper must never re-enable a row.
"""
    parsed_windows, window_errors = [], []
    try:
        snapshot = _stamp(snapshot_utc)
    except (ValueError, TypeError):
        snapshot = None
        window_errors.append("invalid_snapshot_utc")
    for number, window in enumerate(windows):
        if window.get("platform") != platform:
            continue
        label = window.get("id", f"window-{number}")
        try:
            start = _stamp(window.get("start_utc"))
            active = window.get("end_utc") is None
            if active and window.get("status") not in ("active", "running"):
                raise ValueError("open window is not marked active")
            end = snapshot if active else _stamp(window["end_utc"])
            if end is None or end < start:
                raise ValueError("invalid window bounds")
            parsed_windows.append((label, start, end, active))
        except (ValueError, TypeError, KeyError):
            window_errors.append("invalid_window:" + str(label))

    assessments, intervals = [], {}
    for row in sorted(rows, key=lambda item: item["rollout_id"]):
        index = row["rollout_id"]
        reasons = list(window_errors)
        candidates = [event for event in row.get("events", [])
                      if event.get("kind") == "perf"
                      and "perf/train_time" in event.get("metrics", {})
                      and "perf/step_time" in event.get("metrics", {})]
        interval = None
        try:
            if row.get("training_stage_complete") is not True or not candidates:
                raise ValueError("no completed training perf")
            event = candidates[-1]
            end = _stamp(event.get("timestamp"), naive_utc=log_timestamps_are_utc)
            duration = _seconds(row.get("common", {}).get("step_seconds"))
            if not math.isclose(duration, _seconds(event["metrics"]["perf/step_time"]), rel_tol=1e-9):
                raise ValueError("summary duration differs from last training perf")
            if row.get("metric_conflict"):
                raise ValueError("conflicting metrics")
            interval = (end - dt.timedelta(seconds=duration), end)
        except (ValueError, TypeError, OverflowError):
            reasons.append("unknown_training_interval")
        intervals[index] = interval
        direct, previous = [], []
        prior = intervals.get(index - 1)
        if interval is not None:
            if index - 1 in intervals and prior is None:
                reasons.append("unknown_previous_interval")
            for label, start, end, active in parsed_windows:
                # Closed boundaries conservatively include an exactly touching event.
                if interval[0] <= end and start <= interval[1]:
                    direct.append(label)
                if prior is not None and prior[0] <= end and start <= prior[1]:
                    previous.append(label)
                if active and interval[1] > end:
                    reasons.append("active_window_not_observed_through_row:" + str(label))
        status = "unknown" if reasons else "overlap" if direct or previous else "clear"
        assessments.append({
            "rollout_id": index, "status": status, "exclude_timing": status != "clear",
            "interval_start_utc": interval[0].isoformat() if interval else None,
            "interval_end_utc": interval[1].isoformat() if interval else None,
            "overlapping_window_ids": direct, "previous_interval_guard_window_ids": previous,
            "unknown_reasons": sorted(set(reasons)),
        })
    return {
        "platform": platform, "snapshot_utc": snapshot_utc, "rows": assessments,
        "excluded_rollout_ids": [row["rollout_id"] for row in assessments if row["exclude_timing"]],
        "unknown_rollout_ids": [row["rollout_id"] for row in assessments if row["status"] == "unknown"],
        "policy": "Exclude direct reconstructed-interval overlap and its immediately following rollout; unknown coverage is excluded. Union with existing exclusions.",
    }
