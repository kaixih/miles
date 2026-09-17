#!/usr/bin/env python3
"""Keep the sequential experiment and its final offline report in one tmux command."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--job-id", required=True)
args = parser.parse_args()
root = args.root.resolve(strict=True)
if (root / "runner.lock").exists():
    raise SystemExit("Existing runner lock requires inspection; refusing a second driver")
command = [sys.executable, "-u", str(root / "run_two_backends.py"),
           "--job-id", args.job_id, "--root", str(root), "--source", str(root / "source"),
           "--build-context", str(root / "build-context")]
process = subprocess.Popen(command)
code = process.wait()
summary = subprocess.run([
    sys.executable, "-B", str(root / "summarize_results.py"), "--root", str(root),
    "--source", str(root / "source"), "--baseline", str(root / "triton-baseline-first2.json"),
], check=False)
receipt = {"runner_exit": code, "summary_exit": summary.returncode,
           "runner_pid": process.pid, "finished_unix": time.time()}
(root / "driver-exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
raise SystemExit(code or summary.returncode)
