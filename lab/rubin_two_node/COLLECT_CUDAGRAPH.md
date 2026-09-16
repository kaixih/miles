# CUDA Graph experiment snapshots

`collect_cudagraph_snapshot.py` is a local, read-only collector for the new
`20260916-{rubin,gb300}-j<job>-cg` experiments. It never launches, stops, or changes
a job, container, watchdog, model, or remote file. The prior eager collector and
its artifacts are separate.

Create an explicit configuration in the new output namespace. Paths are relative
to this configuration file; `null` means the platform has not started:

```json
{
  "rubin": null,
  "gb300": "gb300-driver-config.json"
}
```

Once Rubin's preparation produces its actual driver configuration, replace its
`null` with that file's path. Do not invent a node for a queued allocation.

```sh
python3 -B lab/rubin_two_node/collect_cudagraph_snapshot.py \
  --config outputs/rubin-gb300-qwen3-cudagraph/collector-config.json \
  --output outputs/rubin-gb300-qwen3-cudagraph
```

Each invocation reads small retained JSON artifacts and the raw training log
through `ssh dl3`. Logs are transferred losslessly as compressed appends after
verifying the entire existing prefix; rotation or a changed prefix triggers a
bounded full snapshot. Logs are limited to 1 GiB, each small file to 8 MiB, each
remote collection to 90 seconds (80-second remote alarm). These are collection
limits and do not change experiment deadlines. It reads no model/checkpoint
payloads and performs no remote retention writes.

A live node is accessed only after a fresh Slurm response proves the exact job,
normal UID 28644, single configured node, RUNNING state and unchanged, unexpired
lease. The node independently checks its hostname and remaining lease. The
collector reads the bound host `watchdog.json` without Docker, so the main
container can be stopped after completion. Ray evidence is selected only by exact
`runtime_env.env_vars.RUBIN_RUN_ID`; the collector retains whitelisted job fields,
not the environment. A different owner, expired/replaced allocation or failed
allocation query prevents all node access. Retained terminal evidence remains
usable; a retained RUNNING record alone becomes UNKNOWN rather than fresh health.

All configured platforms must finish validation before existing local evidence is
replaced. A lock prevents concurrent collectors. The shared transaction helper
rolls back ordinary commit failures and installs `comparison.json` last; a host
crash during the short multi-file rename is outside its atomicity guarantee.
One platform can be pending without blocking the other.

Outputs include per-platform exact logs, launch/source/preparation/exit evidence,
`metadata.json`, `ray-job.json` and `watchdog.json` where observed, a transfer and
allocation receipt, `runs.json`, `health.json`, and `comparison.json`. The health
file is bound to the exact comparison SHA. RUNNING is lifecycle status, not a
claim that the job is healthy. The summarizer retains partial rows, raw rewards,
all distinct optimizer updates, evaluation phase uncertainty and parse conflicts.
`health.json` has a separate completion validation: exact 50 completed rollouts,
200 distinct optimizer IDs with finite positive gradients and finite losses, all
800 distinct rank/rollout/step outcomes NORMAL, both driver exits zero, Ray
SUCCEEDED, and no metric parse/conflict errors. It stays PENDING/NOT_PROVEN until
all criteria pass. This does not independently regrade rewards or prove weight
changes, and RUNNING alone never means completed learning.

The submitted Ray entrypoint is the source for graph intent, training/log-prob
budgets, sampler and batch metadata. An omitted log-prob budget inherits the
actual training budget. Only an actual entrypoint with no PyTorch profiler marks
main timing as unprofiled. Save exclusions come from observed save markers for the
actual save path: warmup rollout 0, saved rollouts and their following rollouts.
Preparation's completed input staging and bulk-I/O declaration are retained with
their source files. Graph capture lines and sampled decode log counts are separate
from replay proof; logging samples are not a total-forward count or fallback rate.
The trace diagnostic remains an independent artifact.

Refresh only the new deck:

```sh
python3 -B reports/rubin-gb300-qwen3-cudagraph/generate.py \
  --runs outputs/rubin-gb300-qwen3-cudagraph/comparison.json \
  --run-health outputs/rubin-gb300-qwen3-cudagraph/health.json
```

CPU regression tests (no SSH):

```sh
python3 -B -m unittest discover -s lab/rubin_two_node -p 'test_collect*' -v
```
