#!/usr/bin/env python3
"""List, and on request delete, every Daytona sandbox labelled with this run's id.

The native provider labels each sandbox with openenv-run-id (OPENENV_RUN_ID) and deletes it when the
episode ends, but Daytona.delete() returns as soon as the request is accepted. Before the key is
removed, this reconciles the account against the run: every sandbox still carried under the run label
is reported; with --delete it is deleted with wait=True and then fetched again to confirm a 404.

Run as the ordinary UID inside the head container:
  DAYTONA_API_KEY_FILE=/work/private/daytona.key python3 reconcile_sandboxes.py --run-id <id> --out <json> [--delete]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OPENENV_DIR = Path("/opt/miles/examples/experimental/openenv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    sys.path.insert(0, str(OPENENV_DIR))
    import tb2_sandbox_daytona as provider  # the same client construction (key file + URL) the run used
    from daytona import ListSandboxesQuery
    from daytona.common.errors import DaytonaNotFoundError

    client = provider.make_daytona()
    labels = {"openenv-run-id": args.run_id}
    found = []
    for sandbox in client.list(ListSandboxesQuery(labels=labels, limit=100)):
        found.append({"id": sandbox.id, "state": str(getattr(sandbox, "state", "")),
                      "task": (sandbox.labels or {}).get("openenv-tbench2-task"),
                      "created_at": str(getattr(sandbox, "created_at", ""))})
    report = {"utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id, "labels": labels,
              "remaining_before": found, "deleted": [], "credential_logged": False}
    if args.delete:
        for entry in found:
            outcome = {"id": entry["id"], "delete_error": None, "get_after": None}
            try:
                client.delete(client.get(entry["id"]), timeout=300, wait=True)
            except DaytonaNotFoundError:
                pass
            except Exception as error:  # noqa: BLE001
                outcome["delete_error"] = f"{type(error).__name__}: {str(error)[:200]}"
            try:
                after = client.get(entry["id"])
                outcome["get_after"] = f"still present, state={after.state}"
            except DaytonaNotFoundError as error:
                outcome["get_after"] = f"404 (status_code={getattr(error, 'status_code', None)})"
            report["deleted"].append(outcome)
    report["remaining_after"] = [
        {"id": s.id, "state": str(getattr(s, "state", ""))} for s in client.list(ListSandboxesQuery(labels=labels, limit=100))
    ]
    report["clean"] = not report["remaining_after"]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("RECONCILE=" + json.dumps({k: report[k] for k in ("run_id", "clean")} | {"remaining_before": len(found),
                                     "remaining_after": len(report["remaining_after"])}), flush=True)
    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()
