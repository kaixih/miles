# GB300: upstream Miles image with TRTLLM hot-refit fixes

This derivative retains the exact upstream image used by the successful GB300
CUDA-graph baseline:

```text
radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986
```

Its recorded runtime is Torch 2.13.0+cu130, CUDA 13.0, Transformer Engine 2.17.0,
FlashInfer 0.6.18, Triton 3.7.1, Ray 2.58.0, and SGLang
0.5.20.dev58+gaea7fb9. SGLang matches the Rubin image's original source commit,
`aea7fb92c047c9c096eae66460acb25dec9ae5a9`, so both exact Python patches apply
without an adapter. This is not derived from the Rubin CUDA 13.4 image.

Only the following SGLang fixes are layered on upstream:

1. [PR #33743](https://github.com/sgl-project/sglang/pull/33743), pinned at
   `89f25a0b9772576fbd427a495a1144f544c90928`, restores canonical BF16 MoE
   weight shape during hot-refit and repacks it after each update.
2. The separately retained final-postload patch makes repeated post-load calls
   idempotent after the bucket hook has already repacked the weights. Its patch
   bytes are identical to the successful Rubin two-rollout experiment.

No FlashInfer, Torch, TE, CUDA, or other native package is installed or rebuilt.
Both stages check all installed distribution metadata, native library bytes,
and Torch ABI before/after. The first stage requires all 11 upstream CPU tests.
The second reproduces the pre-fix double-pack assertion, requires four tests
using real FlashInfer layout functions, then requires the 11 upstream tests
again. BuildKit has no GPU; these tests do not claim GPU execution correctness.

On the allocated ARM GB300 node, from this directory:

```bash
docker pull radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986
DOCKER_BUILDKIT=1 docker build --pull=false --network=none --progress=plain \
  --tag miles-gb300-trtllm-refit:full-20260916 .
```

The actual training process must use the host UID/GID and approved read-only
model/source mounts and writable run/cache mounts. The image build itself only
writes its own image filesystem. Root's common experiment runner handles the
allocation, storage probes, build receipt, and GPU validation.

Provenance inside the image:

- `/opt/miles-moe/provenance.json`: exact upstream base, PR, package/ABI and tests.
- `/opt/miles-moe/followup-idempotent-postload/provenance.json`: final-postload
  patch, real FlashInfer regression tests, package/ABI checks.

`local-validation.json` records local source/checksum/AST validation only.
The build and GPU run receipts must be retained separately when they execute.
