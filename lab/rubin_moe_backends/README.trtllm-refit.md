# TRTLLM refit derivative image

From the Miles repository root, with the baseline image already cached:

```bash
DOCKER_BUILDKIT=1 docker build --pull=false --network=none --progress=plain \
  --file lab/rubin_moe_backends/Dockerfile.trtllm-refit \
  --tag miles-rubin-trtllm-refit:pr33743-89f25a0 \
  lab/rubin_moe_backends
```

The build context is `lab/rubin_moe_backends`, not the repository root.
`BASE_IMAGE` is an accepted build argument, but the helper rejects any value other
than the exact baseline `miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c`
at `gitlab-master.nvidia.com:5005/kaixih/my_docker_hub`.

This adds the pinned [PR33743](https://github.com/sgl-project/sglang/pull/33743)
Python patch only. It discovers the active editable SGLang source, verifies all
three original Git blobs, applies the exact patch, and requires all 11 upstream
CPU tests to pass. Installed package metadata, native binary hashes, and Torch ABI
must remain identical. No package installation, CUDA compilation, GPU execution,
or network access takes place in the build step.

Evidence is retained in `/opt/miles-moe/provenance.json`, `apply.log`, `cpu-test.log`,
and `native-{before,after}.json`. The latter includes content hashes, so this check
can spend some time reading installed native libraries. Retain the Docker build
log as well: failed builds are rejected before a tagged image is produced.

Local tooling checks passed for source discovery, exact patch application, and
rejection of changed or already-patched sources. The upstream CPU tests run inside
the derivative build; the two-rollout GPU validation remains a separate run.
