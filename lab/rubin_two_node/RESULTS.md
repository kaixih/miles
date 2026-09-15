# Two-node Rubin validation

Registry preservation and the MNNVL gate are complete. The Qwen3.5 training
run is still in progress; do not treat model initialization as a passing run.

## Preserved containers

Both ARM64 images were pushed to `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin`
and independently checked through the GitLab API. The core image was recovered
by digest through Enroot on both newly allocated nodes; the Qwen3.5 extension
was recovered by digest through Docker on both nodes. Digests are in
`registry-images.json`. The extension adds only `fla-core==0.5.2` and
`flash-linear-attention==0.5.2`; 40 native distribution versions and the Torch
ABI remained unchanged.

## MNNVL: PASS

Job `2179787` holds `vr-nvl72-ts2-l11-038-c17` and `c18`, four SM10.7 GPUs each.
The unchanged `mnnvl-comm-smoke` launcher and Python test at commit
`36b106a4a2135e3267541610376af908fbb0b5cc` ran one Pyxis rank per GPU.
`NCCL_MNNVL_ENABLE` was unset and `NCCL_NVLS_ENABLE=0`.

- Ranks 0–3 were on c17 and 4–7 on c18.
- Cross-node edges 3→4 and 7→0 used `P2P/MNNVL`.
- All three message sizes produced exactly `8**25 = 3.777893186295716e+22`.
- There were no NCCL warnings/errors and no data transport edges via `NET/Socket`.

| Message size | Average all-reduce time |
| --- | --- |
| 1 MiB | 0.063 ms |
| 64 MiB | 0.578 ms |
| 512 MiB | 4.318 ms |

The fabric reports Completed/Success, one ClusterUUID and CliqueId32766.
Its health also reports Limited Capacity / Bandwidth Degraded, so these timings
are not a performance baseline.

Runtime checks verified Python `/opt/sglang/bin/python3`, Torch
`2.15.0.dev20260818+cu134`, CUDA13.4 and NCCL2.30.7. IMEX channel2020 was accessible.
Both Pyxis and Docker writer probes used UID28644:GID30; the normal user could
read and delete container-created shared output. All probe artifacts were removed.

## Qwen3.5 execution: in progress

The two-node Ray cluster is ready with 32 CPUs and four GPUs per node.
The first bootstrap exposed an infrastructure configuration error: 352 default
Ray CPUs per node exceeded the original200 worker ports. The corrected
orchestrator limits Ray to32 CPUs and uses600 worker ports; its port check also
handles TIME_WAIT sockets safely. Failed bootstrap logs were retained.

Attempt 1 loaded the training checkpoint, both SGLang HF models, and passed
`/health_generate` on both engines. It was intentionally stopped during initial
weight synchronization because the Miles training default set
`NCCL_CUMEM_ENABLE=0`. Actual cross-node NCCL used `NET/Socket`; 512 MiB update
buckets took about four seconds. Its Ray status is STOPPED, not a successful
training run. The retained zero launcher exit code alone would be misleading.

Attempt 2 sets `NCCL_CUMEM_ENABLE=1` consistently in both container and Miles
runtime environments, while retaining `NCCL_NVLS_ENABLE=0` and MNNVL automatic
selection. It uses the same image, model, parallel layout and memory offload.
All eight training actors have the intended environment; actual cross-node
3→4 and 7→0 links use `P2P/MNNVL`. The initial training communicator logs contain
256 MNNVL edges, no Socket edges and no NCCL warnings.

The training checkpoint loaded successfully. The head SGLang engine loaded its
HF weights in 12.21 seconds and passed `/health_generate`; the worker is still
loading HF shards from NFS. Final rollout, optimizer, weight-update, and Ray
SUCCEEDED status remain to be established. First-attempt HF loading took about
19.5 minutes; process sampling identified NFS-backed mmap page faults as the
bottleneck. A transient `/freeze_gc` connection refusal during head startup was
followed by ready status and successful computation.

## Durable evidence

All detailed logs are under:

`/home/scratch.kaixih_ent/repro/miles-rubin-two-node/20260914-j2179787`

Key files are `mnnvl-default.log`, `mnnvl-default-summary.json`,
`bootstrap-attempt4-cumem.log`, `bootstrap.json`, and `logs/qwen35_train.log`.
`logs/qwen35_train-attempt1-socket.log` and `ray-job-attempt1-socket.json` retain
the intentionally stopped first attempt. `recipe-source-attempt2-cumem.json`
records hashes of the source used at launch.
Original image build and push logs are in the sibling experiment
`/home/scratch.kaixih_ent/repro/miles-rubin-container/20260914-j2170666`.
