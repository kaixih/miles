# Recover one failed standalone diagnostic

`recover_cudagraph_diagnostic.py` is a companion to the reviewed host operator.
It never starts, restarts or stops the main training container. It requires the
main and previous diagnostic containers to be **already stopped**, with exact
IDs, images, normal UID, labels and original bind mounts verified afresh.

For Rubin job2203648, use fresh ID `rubin-j2203648-graph-v2`, order `on,off`, and
the original absolute v1 deadline **2026-09-16T10:26:17.421010Z**. The numeric
deadline from v1's `guard-armed.json` remains authoritative, including any
sub-microsecond precision. A new guard does not create another thirty-minute
budget. Less than ten minutes remaining refuses launch. The original Slurm
node, UID and lease must still match, with more than forty minutes of lease
remaining for safe retention.

The recovery reads the actual retained login-host training log and checks all
50 completed rounds and updates0–199, both exits0, frozen source hashes, the
watchdog-locked SUCCEEDED Ray receipt, and current finished watchdog. It does
not contact a now-stopped Ray cluster. Node checkpoint/watchdog evidence and
login log/exit/source evidence remain separate; no nonexistent node-local log
is assumed. V1 output must already have been copied after stop into
`<prior>/node-output-after-stop`, with every byte size and SHA verified against
`diagnostic-after-stop-retention.json`. V1 files are never overwritten.

On dl3, create a **new** evidence manifest after inspecting the expected v1
receipts. Its schema is `{"files": {"relative-receipt-name": "sha256", ...}}`.
Include exactly these nine files from the prior diagnostic directory:

- `operator-plan.json`
- `main-driver-evidence.json`
- `main-login-retention.json`
- `main-retention.json`
- `main-stopped.json`
- `diagnostic-identity.json`
- `diagnostic-stopped.json`
- `guard-armed.json`
- `diagnostic-after-stop-retention.json`

Example plan command (replace both source hashes with the reviewed versions):

```bash
python3 lab/rubin_two_node/recover_cudagraph_diagnostic.py \
  --config /EXACT_MAIN_ROOT/driver-config.json \
  --prior-directory /EXACT_MAIN_ROOT/diagnostics/rubin-j2203648-graph-v1 \
  --prior-evidence-manifest /EXACT_MAIN_ROOT/recovery-v2-evidence.json \
  --run-id rubin-j2203648-graph-v2 \
  --deadline-utc 2026-09-16T10:26:17.421010Z \
  --wrapper-sha256 REVIEWED_WRAPPER_SHA256 \
  --capture-sha256 REVIEWED_CAPTURE_SHA256 \
  --port 31081
```

Plan mode performs no remote calls or mutations. Add `--execute` only after
review. The recovery uses a new node directory, cache and container; copies only
the two SHA-bound reviewed diagnostic sources; rechecks the initial HF model
inventory/metadata and dataset hash; and performs normal-UID storage probes.
An independent host guard is armed for the exact new container before invoking
the wrapper. Image, model, dataset, mode order and measurement protocol remain
unchanged. Neither main nor prior diagnostic receives a process-control action.

After wrapper completion or error, only the new exact diagnostic is stopped,
then its stable closed output is retained to the new durable directory with
source-before/source-after and destination SHA verification. Failed retention
gets an explicit failure receipt and preserves partial files. The old host
guard may remain alive until its original deadline; its exact old container
ID prevents it from targeting v2. No deletion, lease extension, package upgrade,
main-job submission, or main-container restart is implemented.

The eight CPU tests cover these contracts and the real retained-log parser;
they do not prove native SGLang execution. Actual wrapper completion, graph
replay and prefill/decode trace validity remain runtime evidence.
