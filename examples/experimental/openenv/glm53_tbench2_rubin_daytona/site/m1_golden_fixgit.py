#!/usr/bin/env python3
"""M1: run the native scan_golden sweep for one task on Daytona and keep the sandbox evidence.

The native sweep (examples/experimental/openenv/scan_golden.py) creates the task sandbox, starts the
OpenEnv TB2 server, pushes the official solution, scores with the canonical tests/test.sh and deletes
the sandbox, but it never prints the sandbox ID and Daytona.delete() returns before the sandbox is gone.
This wrapper records every sandbox the native Daytona client creates (including ones the provider tears
down itself after a failed start) and every delete it requests, runs the sweep unchanged, then asks the
API about each recorded ID (404 expected) and lists what still carries this run's label.

Run as the ordinary UID inside the container with the provider environment set:
  OPENENV_SANDBOX_BACKEND=daytona DAYTONA_API_URL=... DAYTONA_API_KEY_FILE=/work/private/daytona.key
  OPENENV_TB2_TASKS_DIR=/work/terminal-bench-2 OPENENV_LAUNCHER=... OPENENV_RUN_ID=...
  python3 m1_golden_fixgit.py --tasks fix-git --out-dir /work/output/<run>/m1
"""

import argparse
import json
import os
import re
import runpy
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OPENENV_DIR = Path("/opt/miles/examples/experimental/openenv")
REQUIRED_ENV = ("OPENENV_SANDBOX_BACKEND", "DAYTONA_API_KEY_FILE", "OPENENV_TB2_TASKS_DIR", "OPENENV_RUN_ID", "OPENENV_LAUNCHER")


def utc():
    return datetime.now(timezone.utc).isoformat()


def scrub(text):
    """Error text without URLs: signed preview URLs carry a bearer token."""
    return re.sub(r"https?://\S+", "<url>", str(text))[:300]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_status(api_url, key, sandbox_id):
    """HTTP status of GET /sandbox/{id} without following redirects; None on a transport error."""
    request = urllib.request.Request(f"{api_url}/sandbox/{sandbox_id}",
                                     headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception:  # noqa: BLE001 - transport errors are recorded as unknown, not raised
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="fix-git")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    for var in REQUIRED_ENV:
        if not os.environ.get(var):
            sys.exit(f"{var} must be set")
    if os.environ["OPENENV_SANDBOX_BACKEND"] != "daytona":
        sys.exit("this wrapper records Daytona sandboxes only")
    api_url = os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api").rstrip("/")
    if not api_url.startswith("https://"):
        sys.exit("DAYTONA_API_URL must be https")
    key = Path(os.environ["DAYTONA_API_KEY_FILE"]).read_text().strip()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.out_dir / "sandbox-ledger.jsonl"
    ledger = ledger_path.open("x", buffering=1)

    def record(event, **values):
        ledger.write(json.dumps({"utc": utc(), "event": event, **values}, sort_keys=True) + "\n")

    sys.path.insert(0, str(OPENENV_DIR))
    import tb2_sandbox_daytona as provider

    native_make = provider.make_daytona
    created = []  # every sandbox id the client created, in order

    def make_daytona():
        client = native_make()
        if getattr(client, "_m1_wrapped", False):
            return client
        native_create, native_delete = client.create, client.delete

        def create(params, *a, **kw):
            sandbox = native_create(params, *a, **kw)
            created.append(sandbox.id)
            labels = getattr(params, "labels", None) or {}
            # Positive control for the post-check: the same GET must say 200 while the sandbox exists.
            record("created", sandbox_id=sandbox.id, state=str(getattr(sandbox, "state", "")),
                   task=labels.get("openenv-tbench2-task"), run_label=labels.get("openenv-run-id"),
                   get_while_alive_http=api_status(api_url, key, sandbox.id))
            return sandbox

        def delete(sandbox, *a, **kw):
            record("delete_requested", sandbox_id=sandbox.id)
            result = native_delete(sandbox, *a, **kw)
            record("delete_returned", sandbox_id=sandbox.id, state=str(getattr(sandbox, "state", "")))
            return result

        client.create, client.delete, client._m1_wrapped = create, delete, True
        return client

    provider.make_daytona = make_daytona

    golden_out = args.out_dir / "golden.jsonl"
    record("sweep_started", tasks=args.tasks, concurrency=args.concurrency, tasks_dir=os.environ["OPENENV_TB2_TASKS_DIR"],
           run_id=os.environ["OPENENV_RUN_ID"], launcher=os.environ["OPENENV_LAUNCHER"])
    sys.argv = ["scan_golden.py", "--tasks", args.tasks, "--concurrency", str(args.concurrency), "--logs", "--out", str(golden_out)]
    sweep_error = None
    try:
        runpy.run_path(str(OPENENV_DIR / "scan_golden.py"), run_name="__main__")
    except SystemExit as exit_:
        if exit_.code not in (None, 0):
            sweep_error = f"SystemExit({exit_.code})"
    except BaseException as error:  # noqa: BLE001 - keep the evidence even when the sweep dies
        sweep_error = f"{type(error).__name__}: {scrub(error)}"
    record("sweep_finished", error=sweep_error, golden_out=str(golden_out))

    checks = []
    for sandbox_id in created:
        status = None
        for _ in range(36):  # deletion completes shortly after delete() returns; allow up to ~3 minutes
            status = api_status(api_url, key, sandbox_id)
            if status == 404:
                break
            time.sleep(5)
        checks.append({"sandbox_id": sandbox_id, "http_status": status, "absence_confirmed": status == 404})
        record("post_delete_get", sandbox_id=sandbox_id, http_status=status)

    remaining = None
    try:
        from daytona import ListSandboxesQuery

        client = native_make()
        remaining = [{"id": s.id, "state": str(getattr(s, "state", ""))}
                     for s in client.list(ListSandboxesQuery(labels={"openenv-run-id": os.environ["OPENENV_RUN_ID"]}, limit=100))]
    except Exception as error:  # noqa: BLE001
        remaining = f"list failed: {type(error).__name__}: {scrub(error)}"
    record("run_label_sweep", remaining=remaining)
    ledger.close()

    results = [json.loads(line) for line in golden_out.read_text().splitlines()] if golden_out.exists() else []
    leaked = any(key in path.read_text(errors="replace") for path in (ledger_path, golden_out) if path.exists())
    del key
    summary = {
        "utc": utc(), "tasks": args.tasks, "run_id": os.environ["OPENENV_RUN_ID"], "sweep_error": sweep_error,
        "results": results, "sandboxes_created": created, "post_delete_checks": checks, "run_label_remaining": remaining,
        "credential_in_evidence": leaked,
        "passed": (not sweep_error and bool(results) and all(r["reward"] == 1.0 for r in results)
                   and len(created) >= len(results) and all(c["absence_confirmed"] for c in checks)
                   and remaining == [] and not leaked),
    }
    (args.out_dir / "m1-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("M1_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
