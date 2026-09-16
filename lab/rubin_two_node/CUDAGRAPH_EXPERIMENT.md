# Qwen3 CUDA Graph experiment: manifest and evidence protocol

Status: proposed protocol for newly allocated nodes. This document launches no
jobs and does not change the previous experiments, their deadlines, or their
HTML slides. Resolve the choices below into a per-run manifest before launch.

## Questions and comparisons

1. Can the pinned Rubin and upstream GB300 runtimes complete the same 50-rollout
   Qwen3/GSM8K learning recipe with decode CUDA Graph replay enabled?
2. Within each runtime, does replay reduce the recorded host-submission spacing
   seen in the earlier eager decode traces?
3. How do the new systems compare at the same training token budget?

Use **4,096 training and log-prob tokens/GPU on both new runs**. The previous
Rubin A2 used 4,096; the previous GB300 used 8,192. Consequently old GB300 versus
new GB300 is a graph-plus-budget comparison, not an isolated graph speedup.
The previous save cadence also differed: Rubin every 10 rollouts, GB300 every
50. Standardize the new cadence at 50, with the existing deadline save sentinel;
retain and disclose these differences. Available images remain different, so
the cross-platform result does not isolate hardware alone.

For the narrow decode question, the recommended main intervention is **decode
graphs ON, prefill graphs OFF**, preserving the previous prefill choice. If the
operator instead chooses both ON, record that explicitly and keep prefill mode
fixed in the paired decode diagnostic. Removing a disable flag is not evidence
that capture succeeded or that a given forward replayed a graph. Record the
resolved backend and actual execution; do not silently relabel fallback as ON.

There is no retained evidence that the previous graph-off choice followed an
SGLang graph capture failure. It originated in the optional paths disabled by
the Qwen3.5 execution smoke and was carried into Qwen3. It is not evidence of a
Rubin or GB300 hardware limitation.

## Immutable learning inputs

| Field | Required value or evidence |
| --- | --- |
| Layout | One node, four GPUs; record model name, ES status, actual memory and CPU topology |
| Model | Standard Qwen3-30B-A3B, HF revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Initial state | Same HF plus converted `release` checkpoint; fresh optimizer/RNG, no previous training checkpoint |
| Input identity | Verify staged shard inventory/sizes and metadata hashes; previous combined HF/release ID `c94e89d6d72c8131394c6196d04fdb5e681a6768d3ea329dd5dc333401348d46` |
| Data | Same 7,473 GSM8K train rows and same fixed 256 test rows; preserve messages, labels, tokenizer and template |
| Train JSONL | Previous SHA256 `f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8`; verify mounted bytes, and record the fixed-test SHA too |
| Reward | Existing `gsm8k_verl_reward.reward_func`, strict `####`, same source hash; use for train and evaluation; no reward-group filter |
| Sampling | Training T=1, top-p=1, top-k=-1; evaluation T=1, top-p=0.7, top-k=-1 |
| Seeds | Preserve training seed 1234 and rollout seed 42; verify these against resolved source/argv, not assumptions |
| Template/length | Existing Qwen3 default thinking template; prompt 512, response 1024, context 1536 |
| Work | 50 rollouts × 256 prompts × 8 responses; global response batch 512; four updates/rollout, 200 updates total |
| Training | TP1/PP1/CP1/EP4, expert TP1; BF16; same optimizer, GRPO, KL, recomputation and `flash` policy |
| Inference | Four TP1/EP1 engines; Triton attention/MoE, Torch BF16 GEMM, static fraction 0.55, max running requests 128 |
| Memory budget | Explicit 4,096 for training and resolved log-prob budget on both platforms |
| Evaluation/save | Initial + every 10 rollouts evaluation; final save at 50, plus deadline sentinel if needed |

Do not claim seeds guarantee identical generated responses across stacks or
graph modes. Keep source commits, resolved argv and environment allowlists,
data hashes, image digests, Slurm allocation, container identity, Ray submission
ID and UTC deadlines in each manifest. Record graph choices separately for
decode and prefill, including capture batch sizes, padding policy, memory delta
and any fallback reason. The actual runtime graph fields override display labels.

Pinned images, unchanged from the prior experiment:

- Rubin: `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`
- GB300: `radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986`

Record installed versions again. Do not rebuild GB300 to match Rubin, alter
kernel choices, change graph padding, or reduce concurrency to make a capture
pass without recording a separate configuration. The reviewed local launcher
option `--sglang-enable-cuda-graph` removes only the decode disable flag while
retaining explicit piecewise/prefill disable. Its default remains graph-off.
Record the deployed source hash and resolved runtime graph backend: the option
does not force unsupported capture to succeed.

## Small paired diagnostic, separate from the learning curves

Use a bounded initial-policy SGLang OFF/ON pair per platform, with the same image,
model, engine layout, resource limits, Triton/Torch backends and KV-cache policy.
It measures generation only; the manifest still records the common 4,096-token
training budget, but this diagnostic does not exercise optimizer microbatches.
Do not reuse a late-policy trained engine for one side of the pair.

Freeze actual token IDs from a small GSM8K prompt subset and hash the request
list. Submit identical requests/seeds in each condition. A practical short
workload is 128 concurrent requests per TP1 engine, fixed bounded output length
(e.g. 64 new tokens), with early EOS disabled **only in this labelled diagnostic**.
Keep that diagnostic length/EOS policy identical in OFF and ON; never use its
scores as the learning curve or held-out evaluation. Confirm the pinned engine
API supports the exact payload before execution. Use all four engines for the
unprofiled pair when feasible, so host contention resembles the main workload;
trace only one identified engine.

Separate cold model load, graph construction and cache/JIT warmup from measured
requests. Repeat the same bounded request set at least three times per mode;
record all repetitions rather than choosing the best. Record order and warmup
counts. Prefer opposite OFF/ON order on the two platforms, or a bounded reverse
repeat if time permits. Keep prefix-cache state identical by an explicitly
verified reset or by recreating each condition; otherwise report it as a
confound. Do not infer a causal end-to-end Miles gain from this SGLang-only pair.

For decode comparison, select the same actual batch size, per-request decode
token index and context-length distribution. Preserve prompt/output token counts,
active request IDs, token-index/length distributions, queued requests and graph
padding size. Same `bs=128` alone did not establish equal context in the previous
traces. Prefill measurements must be reported separately, with actual input
token count, batch size, chunking and cache-hit state. If exact correspondence
cannot be established, retain the samples but call the comparison diagnostic.

A maximum 20-minute OFF/ON diagnostic budget is a planning limit, not a promised
runtime. Prioritize completing the two main learning runs. Run diagnostic phases
before the mains only if input staging and enough lease budget are confirmed;
otherwise schedule them after terminal mains with an explicit remaining budget.
Do not run the pair concurrently with the main training workload.

## Verify graphs and keep profiling small

Retain startup logs showing successful capture for the selected backend/batch
sizes and its allocated memory. For measured decode forwards, retain runtime
graph-use counters/logs and an actual replay indication, such as correlated
`cudaGraphLaunch`/`cuGraphLaunch` events or graph execution metadata. Absence of a
particular API spelling is not alone failure; reconcile it with the installed
backend. Report the number of decode/prefill forwards, replayed forwards,
fallback forwards and unknown forwards. Keep graph fallback timing separately.

Use unprofiled measurements for throughput. After the main run is terminal, use
one identified TP1 engine in a separate generation-only diagnostic. Arm its
pinned legacy profiler (`SGLANG_PROFILE_V2=0`) once with `num_steps=4`, CPU+GPU
activities, `profile_by_stage=true`, `detailed_annotations=true`,
`with_stack=false`, `record_shapes=false` and `merge_profiles=false`. The first
decode flushes an active prefill capture early, so separate `EXTEND` and `DECODE`
exports do not guarantee four prefill forwards. Classify actual step annotations,
including mixed forwards, batch size and query/context token counts; a tiny
`EXTEND bs=1` sample is not representative long prefill. Keep decode graph ON
and prefill graph OFF explicit. Require the existing single-arm, exact
PID/start-ticks/job checks and ≤180-second capture deadline. The offline
`sglang_graph_capture.py` helper builds payloads and checks retained traces; the
runtime operator must enforce those identity and deadline checks. Confirm actual
replay using a CUDA graph launch on the same CPU PID/TID inside a `DECODE` span
with GPU kernels, or equivalent verified graph execution evidence. Preserve the
same graph policy while profiling; disclose any required profiling option and
keep unprofiled replay evidence primary. Do not send diagnostic requests to the
learning run or treat these short diagnostic timings as learning throughput.

Do not enable four-rank full trainer profiling. The previous trainer export hit
its deadline on Rubin and host-memory OOM on GB300. It is unnecessary for this
decode-launch question. Poll total artifact bytes and host RSS; use a small trace
cap (1 GiB per selected capture is ample relative to the prior <0.5 MiB traces),
while acknowledging polling can overshoot. Preserve partial export and failure
evidence; never count an incomplete gzip as a verified trace. Retain before/after
source and destination hashes for complete captures.

Analyze kernel interval unions and CPU launch correlations, not summed nested
timers or claimed physical GPU idle. Compare OFF/ON within each runtime first.
Keep profiler overhead, clock alignment uncertainty, images and context/routing
differences visible. Separate learning quality, system timers and sampled trace
diagnostics in the new HTML slides.

## Storage, eight-hour allocation and watchdog parameters

Use a fresh unique namespace on each allocated node, owned by UID28644:GID30:

```text
durable: /home/scratch.kaixih_ent/repro/miles-qwen3-cudagraph/<date>-<platform>-j<jobid>/
node:    <verified-local-raid>/miles-kaixih-j<jobid>-cudagraph/
         models/   run/inputs/   run/checkpoints/   cache/   diagnostics/
```

These are templates, not verified destinations. Use the selected node's actual
mounts, free space and write/readback/delete probes before creating the run.
Mount repo and models read-only; `run` and cache writable by the normal UID.
Preserve the CIFS/direct-NFS alias distinction on GB300. The existing orchestrator
supports `--node-repo`, `--node-models`, `--node-run-dir`, `--node-local-output`,
UID/GID and a verified local-to-durable retention route. GB300 needs
`--megatron-path /root/Megatron-LM` in the forwarded launcher arguments and the
already documented source-directory traversal fix. Use explicit nofile65535;
the current orchestrator already emits `--ulimit nofile=65535:65535`.

Finish HF/reference staging and image pull before timed work. No background
model/checkpoint copying during measured rollouts or the OFF/ON pair. Record
unavoidable operational I/O windows and exclude affected intervals plus the
documented following-rollout guard. Do not silently retain those points in
timing curves or paired summaries.

For each actual eight-hour allocation, derive immutable UTC times from its
recorded end `L`: **main soft deadline L−90min; hard deadline L−60min**. This
reserves one hour after the hard deadline for terminal evidence, local checkpoint
verification and retention. Preparation and diagnostics consume the same lease;
they do not reset these times. If a node becomes ready too late, record an
insufficient-budget outcome rather than extending the allocation implicitly.

Arm the existing watcher before submitting the exact run, using the same sentinel
and run ID as the training launcher:

```bash
python3 /opt/miles/lab/rubin_two_node/watch_qwen3_run.py \
  --run-id "$RUN_ID" --ray-address "http://$NODE_IP:28265" \
  --soft-deadline "$SOFT_UTC" --hard-deadline "$HARD_UTC" \
  --lease-deadline "$LEASE_UTC" --num-rollout 50 --poll-seconds 5 \
  --sentinel /run-output/checkpoint-now --save-dir /run-output/checkpoints \
  --state-path /run-output/watchdog.json --log /run-output/logs/qwen3_train.log
```

The orchestrator invokes `--recipe qwen3-gsm8k --num-rollout 50`, with fresh node,
IP, job, image, run/cache paths and container prefix. Use `plan` and launcher
`--print-only` before `bootstrap`/`train`, forwarding `--sglang-enable-cuda-graph`,
`--max-tokens-per-gpu 4096`, save/eval settings and
`--extra-env-vars RUBIN_RUN_ID=<exact-run-id>`. Verify that log-prob budget resolves
to4096 too. Do not use the old job/path defaults. The watcher is a separate
invocation; `orchestrate_rubin.py train` does not arm it automatically.

The watcher requests a synchronous save at the next boundary, then stops only
the locked matching job when the numeric checkpoint advances and Miles removes
the owned sentinel; hard deadline may stop before save completion. Its exit0
means monitoring finished, not50/200 success. Preserve sentinel/state/Ray/driver
records and independently validate checkpoint metadata, shard inventory and
rollout state. Report a saved STOPPED run as partial, never complete.

**Full checkpoint retention needs a separate feasibility decision before launch.**
The previous native checkpoint was about427.6 GB; at the observed slow GB path
of15.56 MiB/s, transfer takes over seven hours. A60-minute reserve cannot promise
that transfer. Verify an adequately fast durable route or a directly writable
durable checkpoint target first; otherwise explicitly record that only local
checkpoint plus small durable evidence is guaranteed. Keep at most the intended
latest checkpoint and enough local space for a save overlap. Do not start a huge
copy with an ETA beyond the original lease or interfere with the other main run.
Small logs/metrics/manifests/traces must be durably retained first; incomplete
checkpoint transfers stay labelled partial with byte counts and receipts.

## Completion and new HTML evidence

Preserve the previous comparison JSON, raw logs, report data and HTML archive.
Create a new comparison/report namespace; do not stitch old and new attempts.
Require fresh Ray `SUCCEEDED`, both launcher/driver exits0,50 completed rollout
IDs0–49 and200 updates per main; check finite positive aggregate gradients and
all expected rank outcomes. Preserve all raw reward/truncation/evaluation values.
Do not claim quality superiority from a one-question fixed-test difference.

Timing plots retain null gaps for exclusions. Stage comparisons use the common
intersection of completed unprofiled eligible IDs, with N and IDs visible.
Report graph replay/fallback fractions alongside throughput, cold capture cost
and capture-memory overhead. Include the within-runtime OFF/ON diagnostic as a
separate panel, explicitly state which prefill mode was held fixed, and disclose
GB's old8,192→new4,096 budget change. The final report must distinguish main
completion, graph enablement/replay proof, diagnostic completion and trace export.

Reviewed implementation references: `run_qwen3_30b_a3b_gsm8k_rubin.py`,
`orchestrate_rubin.py`, `watch_qwen3_run.py`, `capture_sglang_replay.py`,
`PROFILE_INITIAL_POLICY_SOURCE.md`, and the retained initial-profile plans.
