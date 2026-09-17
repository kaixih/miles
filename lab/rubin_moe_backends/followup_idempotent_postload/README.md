# Follow-up: idempotent final post-load after PR33743

The exact PR33743 image reached CUDA graph capture and completed all 116 initial
weight-update buckets. Each bucket's new PR hook repacked BF16 expert weights.
Miles then sent `end_weight_update(run_post_load=True)`, which called the full
`process_weights_after_loading` hook again. Its unconditional packing loop passed
an already blocked per-expert 3D tensor to FlashInfer's 2D row-permutation helper.
This caused `AssertionError: x should be a 2D tensor, not 3` at 00:23:38 UTC.

The separate patch replaces that unconditional loop with PR33743's existing
per-weight canonical-shape-gated repack method. Cold canonical weights still get
packed; already packed weights remain unchanged; mixed canonical/packed weights
are handled independently. No FlashInfer kernel or native package changes.

The original image and PR patch remain intact. This derivative starts from
`sha256:971195a84be7573abaa5add343414633dffd046cc947eaa2ff744c4f5148a95e`.

BuildKit needs a local image tag rather than a bare image ID in `FROM`. After
verifying the parent image ID, build from this directory:

```bash
docker tag sha256:971195a84be7573abaa5add343414633dffd046cc947eaa2ff744c4f5148a95e \
  miles-rubin-trtllm-refit:parent-971195a84be7
DOCKER_BUILDKIT=1 docker build --pull=false --network=none --progress=plain \
  --build-arg PR33743_IMAGE=miles-rubin-trtllm-refit:parent-971195a84be7 \
  --tag miles-rubin-trtllm-refit:postload-j2211281-r2 .
```

Build gates use the installed, real FlashInfer permutation and block-layout
functions on CPU BF16 tensors with Qwen3 TP1 expert geometry (hidden 2048,
intermediate 768). These functions are not mocked. The lightweight model shell
allows the actual full SGLang post-load method to run without a model server.

1. Reproduce the exact double-packing assertion on the unmodified parent.
2. Apply the separate, checksum-pinned patch to an exact Git-blob-matched source.
3. Require four real FlashInfer tests: canonical round-trip, repeated final
   post-load, bucketed updates followed by final post-load matching cold-load
   bytes, and mixed canonical/packed weights.
4. Require all 11 original upstream PR tests to pass without skips.
5. Verify installed native package metadata, native binary content, and Torch ABI
   remain unchanged.

For the actual CUDA TRTLLM configuration, AITER/HIP, CPU-AMX, and NPU post-load
paths are inactive; DeepGEMM is not selected; the optional SwiGLU interleaving
requires the Triton runner, so it does not transform TRTLLM weights.

Provenance lives in `/opt/miles-moe/followup-idempotent-postload/`, including
`before.log`, `after.log`, `cpu-test.log`, native snapshots, and `provenance.json`.
The original `/opt/miles-moe/provenance.json` is retained unchanged. CPU layout
validation does not replace the subsequent GPU rollout and hot-refit validation.

The build passed on c05 at 2026-09-17 00:32:36 UTC (73 seconds). It reproduced
the unmodified failure, passed all four real FlashInfer tests and all 11 original
tests, and confirmed native metadata/binaries/ABI unchanged. Output image:
`sha256:448a37a80616c92c5674489585ae4c3a3edd981d6d442bea121676e0dec3f4ad`.
Retained build evidence is under the original run root as
`trtllm-postload-build-r2.log`, `trtllm-postload-build-r2.json`, and
`trtllm-postload-build-r2-image.json`.
