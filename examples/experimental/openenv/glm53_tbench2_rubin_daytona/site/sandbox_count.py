#!/usr/bin/env python3
"""Append one line per poll with the number and states of Daytona sandboxes under a run label.

Run inside the head container as the ordinary UID while training runs:
  DAYTONA_API_KEY_FILE=/work/private/daytona.key python3 sandbox_count.py --run-id <id> --out <jsonl> [--interval 300] [--once]
Read-only: lists only; never creates or deletes.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OPENENV_DIR = Path("/opt/miles/examples/experimental/openenv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(OPENENV_DIR))
    import tb2_sandbox_daytona as provider
    from daytona import ListSandboxesQuery

    client = provider.make_daytona()
    with args.out.open("a", buffering=1) as out:
        while True:
            try:
                boxes = list(client.list(ListSandboxesQuery(labels={"openenv-run-id": args.run_id}, limit=200)))
                row = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id, "count": len(boxes),
                       "states": dict(Counter(str(getattr(b, "state", "")).replace("SandboxState.", "") for b in boxes)),
                       "tasks": dict(Counter((b.labels or {}).get("openenv-tbench2-task") for b in boxes))}
            except Exception as error:  # noqa: BLE001 - a listing failure is a data point, not a crash
                row = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id, "error": f"{type(error).__name__}: {str(error)[:160]}"}
            out.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
