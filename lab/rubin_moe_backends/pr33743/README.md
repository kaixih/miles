# SGLang PR #33743 for the existing Rubin image

This directory pins the original upstream PR patch; no manual backport was needed.

- PR: https://github.com/sgl-project/sglang/pull/33743
- PR head: `89f25a0b9772576fbd427a495a1144f544c90928` (`Kh4L/sglang`)
- PR base at retrieval: `17ba2c2e7c7b81f31a8a9e693e7435ab262c16b4`
- Image SGLang revision: `aea7fb92c047c9c096eae66460acb25dec9ae5a9`
- Patch: `pr33743-89f25a0.patch`
- Patch SHA256: `2137ce6f3ec431e70f0ffd4e7e876070255a3e9f1488ece1c5346b1111fa908d`
- PR state at retrieval: open, not merged.

## Changed files

1. `python/sglang/srt/layers/quantization/base_config.py`: default no-op repack hook.
2. `python/sglang/srt/layers/quantization/unquant.py`: restore actual canonical weight data, then rederive the TRTLLM BF16 BlockMajorK layout.
3. `python/sglang/srt/model_executor/model_runner_components/weight_updater.py`: invoke repack on hot-update paths, including bucketed calls, in a finally block.
4. `test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py`: CPU regression coverage with a stand-in permutation.

## Local validation

The fetched source files' Git blob SHA1 values were independently verified. The reconstructed unified patch:

- passes `git apply --check` against the image revision;
- applies to the image revision with exact context and line offsets only (no manual edits);
- applies to the PR base and reproduces all four PR head files byte for byte;
- produces Python files accepted by `ast.parse`.

The local macOS Python has no PyTorch; the upstream CPU unit test was not executed here. No remote edits, builds, or launches were performed for this preparation.

## Apply in the derivative image

Copy the pinned patch to the build context. From the container's SGLang repository root:

```bash
test "$(git rev-parse HEAD)" = aea7fb92c047c9c096eae66460acb25dec9ae5a9
git apply --check /path/to/pr33743-89f25a0.patch
git apply /path/to/pr33743-89f25a0.patch
python test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py -v
```

Use the existing editable SGLang source tree. This PR changes Python code and does not require rebuilding CUDA kernels. A true `git cherry-pick` would need the individual PR commits/merge-base; applying this pinned full PR diff is the equivalent scoped source change without advancing unrelated SGLang code.

## Runtime checks still needed

The clean application proves source compatibility, not kernel availability or correct execution on SM107. The derivative image must still verify:

- the installed FlashInfer TRTLLM BF16 backend can select an SM107-compatible kernel;
- the upstream CPU unit test passes in the actual image;
- cold inference works with the selected Qwen3 model, graph settings, and precision;
- two rollout/training steps exercise a post-training hot weight update and subsequent generation without shape errors or invalid values;
- loss/reward/logprob sanity and throughput use the actual requested configuration.

The patch does not change a forward kernel and does not itself establish a performance gain. It restores and repacks expert data per update RPC, so bucketed refit overhead remains part of the experiment. Do not replace it with a shape-only reshape: untouched experts in other buckets can otherwise retain incorrectly transformed values.

## Evidence layout

- `pr-metadata.json`: connector PR metadata.
- `source-files.json`: exact fetched source contents and upstream blob identities.
- `image-base/`, `pr-base/`, `pr-head/`: immutable fetched source snapshots.
- `image-base-applied/`, `pr-base-applied/`: local application/AST verification copies.
