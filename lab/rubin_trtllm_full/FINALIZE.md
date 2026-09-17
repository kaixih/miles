# Finish the retained TRTLLM comparison

Run locally from the Miles workspace. `finalize_campaign.py` is plan-only by
default and binds **Rubin job 2213753** and **GB300 job 2212644**. It rejects the
failed first Rubin run, mismatched images, and a different frozen runtime.

After replacement preparation reaches READY, retain its actual driver config as
`outputs/rubin-gb300-qwen3-trtllm-full/rubin-driver-config.json`; the collector
mapping already names that file. Until this copy is made, a stale config for
job 2212643 correctly fails even the local plan check. Copy the real generated
config from `dl3:/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-rubin-j2213753-trtllm/driver-config.json`;
do not manufacture a config by changing the old job ID.

```bash
python3 lab/rubin_trtllm_full/finalize_campaign.py
```

After both durable controllers report **COMPLETE**, finish with:

```bash
python3 lab/rubin_trtllm_full/finalize_campaign.py --execute
```

The tool does the following, stopping on any failed check:

1. Read the two controllers' completion records, checkpoint PASS receipts and
   retained profiling receipts from **dl3**. Verify the exact run/image identities,
   successful profile terminal state and plan/terminal hashes against each
   diagnostic retention manifest.
2. Download only the manifest-listed files under each
   `diagnostics/<platform>-j<job>-trtllm-profile-v1/node-output/`. Check their sizes
   and SHA256 values locally. No model weights, checkpoint shards, caches or
   source trees are downloaded. Each manifest is bounded to 1,100 MiB.
3. Run the existing `profiling/analyze_profile.py --png` against each fresh local
   profile using its **retained** plan/terminal hashes. The analyzer verifies
   actual CUDA Graph correlation, batch size and workload; it renders prefill
   and decode screenshots separately.
4. Run the existing `collect_cudagraph_snapshot.py` with
   `outputs/rubin-gb300-qwen3-trtllm-full/collector-config.json`. Its terminal
   evidence fallback works after main containers stop or leases end. If an
   allocation remains live, its normal guarded read-only check may also run.
5. Run `build_report.py --final`, then browser QA for all 12 slides. Existing
   completion, curve, matching-profile and numerical checks remain mandatory.

Default outputs:

- Evidence, analyzer results and command logs:
  `outputs/rubin-gb300-qwen3-trtllm-full/finalization/`
- Final HTML: `reports/rubin-gb300-qwen3-trtllm-full/site-final/index.html`
- Screenshots and `browser-checks.json`:
  `reports/rubin-gb300-qwen3-trtllm-full/qa-final/`
- Success marker: `finalization/complete.json`, written only after browser QA.

The existing interim `site/`, earlier decks, source manifests and runtime files
remain untouched. The existing collector refreshes only this campaign's local
main-run evidence. The tool does not launch training, control containers,
release jobs, operate Git, or publish private evidence.

Use the bundled local Node.js and Google Chrome already used by the trace
renderer and slide QA. Browser rendering may require the usual sandbox
approval. Review the four profile screenshots and performance slides visually
after the automated check, then give the user the new HTML path. No uploading or
sharing is implied.

The work/site/QA destinations must be fresh. On failure, inspect `failure.json`
and stage logs and preserve partial evidence. For a deliberate second
finalization attempt, choose fresh `--work`, `--site` and `--qa` directories
inside this campaign/report; do not delete failed evidence or loosen the final
report gates. Node release and any delivery/publishing decisions remain with
the main task.
