#!/usr/bin/env python3
"""Run an explicit JSON case list, retaining a bounded log and status per process."""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    results = []
    for case in cases:
        name = case["name"]
        assert name and all(c.isalnum() or c in "_-" for c in name)
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("repro.py")), *case["args"]]
        start = time.monotonic()
        with (args.output / f"{name}.log").open("xb") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=limits, start_new_session=True,
                                       env={**os.environ, **case.get("env", {})})
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code = 124
        log_text = (args.output / f"{name}.log").read_text(errors="replace")
        events = []
        for line in log_text.splitlines():
            if line.startswith("{\"event\":"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        result = {**case, "exit_code": code, "elapsed_s": time.monotonic() - start,
                  "events": events, "log": f"{name}.log"}
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"name": name, "exit_code": code, "elapsed_s": result["elapsed_s"],
                          "last_event": events[-1] if events else None}), flush=True)
    print(json.dumps({"event": "matrix_complete", "cases": len(results)}), flush=True)


if __name__ == "__main__":
    main()
