# Collecting after the main container stops

Use the local `lab/rubin_two_node/collect_cudagraph_snapshot.py` with its existing
`--config` / `--output` arguments. It imports the adjacent campaign helper
`lab/rubin_trtllm_full/terminal_evidence.py`. No frozen runtime or controller
change is needed. The collector reads retained files on dl3; an expired or
released allocation never permits compute-node access.

For each exact TRTLLM run, the fallback requires these durable run-root files:

- `driver-config.json`, `source-manifest.json`, `train-launch.json`,
  `cudagraph-main-plan.json`, `train_exit.json`, `train-driver-exit.json`, and
  `logs/qwen3_train.log`.
- `campaign-main-completion.json`.
- Under `diagnostics/{platform}-j{job_id}-trtllm-profile-v1/`:
  `operator-plan.json`, `main-driver-evidence.json`,
  `main-login-retention.json`, `main-retention.json`, and
  `main-before/watchdog.json`.

The profile operator retains all of those before stopping the main container.
Therefore later profiling failure does not lose the terminal training proof.
When `main-final/watchdog.json` and `main-final-retention.json` both exist, the
collector additionally verifies that final snapshot and uses it.

Validation binds the full retained Ray **SUCCEEDED** record and its runtime
run ID/submission/entrypoint to the original image/container/source, zero driver
exits, controller completion receipt, SHA-verified closed log, and SHA-verified
terminal watchdog. Incomplete retention cannot supply this fallback; conflicting
or modified evidence fails the transactional collection without replacing the
previous snapshot. A terminal watchdog or 50 curves alone cannot establish Ray
success for the TRTLLM campaign.

Local output retains the source artifacts plus `terminal-evidence-import.json`,
which records every imported artifact's raw SHA256/size and the selected
watchdog/Ray paths. `metadata.json` and the collection receipt identify the
retained terminal source. The existing collector separately recomputes all
50-rollout / 200-update / 800-normal-rank checks from the actual log.

CPU-only verification:

```bash
python3 -B -m unittest discover -s lab/rubin_trtllm_full -p test_terminal_evidence.py
python3 -B -m unittest discover -s lab/rubin_two_node -p test_collect_cudagraph_snapshot.py
```
