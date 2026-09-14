# Experimental Miles on Vera Rubin

This image targets core Miles with the Megatron BF16 backend, the default `raw`
weight-conversion path, and SGLang rollout on aarch64 SM107. It is a bringup
candidate until the real hardware smoke and end-to-end training run pass.
Existing example launchers using `--hardware auto` have no Rubin profile;
the first end-to-end run needs explicit training/rollout GPU configuration.

Build from the Miles repository root on a native aarch64 Docker host:

```sh
docker build --platform linux/arm64 -f docker/Dockerfile.rubin \
  --build-arg MAX_JOBS=8 --build-arg NVCC_THREADS=2 -t miles:rubin-core .
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp miles:rubin-core \
  python3 /opt/miles-smoke/verify_rubin.py

docker run --rm --gpus all --ipc=host --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp miles:rubin-core \
  python3 /opt/miles-smoke/verify_rubin_attention.py

docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp miles:rubin-core \
  python3 -m torch.distributed.run --nnodes=1 --nproc-per-node=4 \
  --master-addr=127.0.0.1 --master-port=29500 \
  /opt/miles-smoke/verify_rubin.py --distributed

docker run --rm --gpus all --ipc=host --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MAX_JOBS=4 miles:rubin-core \
  python3 /opt/miles-smoke/verify_rubin_rollout.py
```

The base is `lmsysorg/sglang:nightly-cu134-20260911-a4ff563`, not the ordinary
SGLang v0.5.19 image. It provides CUDA 13.4, Python 3.12, Torch
2.15.0.dev20260818+cu134, CuTe DSL 4.8.0.dev0, and SGL native components built
against that Torch version. The build records their exact versions and ABI,
constrains pip to preserve them, and checks them again before completion.
The base digest is pinned to
`sha256:4af3254b070e087afdf8cabebe895132bb4d1d6cf6fd5b5c60f0a3da07c1b468`.
Editable Miles and Megatron sources live under `/opt` so an ordinary runtime
UID can import them without traversing `/root`.

Keep compiler caches and image layers on node-local storage. For shared logs or
other outputs, mount only the needed directory and use the host UID/GID shown
above. Verify that the normal login user can remove a container-created probe.

Training builds from pinned source: TE 2.19, Apex, FA2 2.8.3.post1, Megatron-LM's
Miles branch, and mbridge. FA2's build flags are patched to C++20 and SM107;
Apex's explicit C++17 flags are updated to C++20. Patches are saved in
`/opt/miles-rubin/`. TE targets `107a` explicitly with CUDA 13.4, avoiding
unneeded Blackwell targets; TE NCCL-EP is disabled for the first BF16 build.
The pip cuDNN headers and libraries are added to compiler search paths. TE's
source and PyTorch build state use a node-local BuildKit cache. Its common
CMake objects live in a temporary directory and may need recompilation after
a failed build.

The SGLang Miles branch is pinned at aea7fb92c047c9c096eae66460acb25dec9ae5a9.
The checkout clears the base repository's inherited GitHub authentication
header and uses the public SGLang remote explicitly. TMS is force-reinstalled from its
pinned source commit even when the base carries the same package version.
Its Python metadata is reconciled to the installed cu134 native versions before
installing with `--no-deps`. Its Rust modules are not rebuilt. Upstream cu134
still marks Rust TreeCore support for this Torch nightly as unfinished; the
rollout smoke must therefore confirm the selected Python/Rust runtime path.

Excluded from this core image:

- Hopper-only FA3 and the old cu130/Torch2.13 CUDA wheels.
- The base's FA4 beta, whose CuTe imports are incompatible with DSL 4.8.
  Its SGLang dependency declaration is also removed. Use cuDNN/FA2 for
  training attention and FlashInfer for the initial rollout validation.
- Mamba/causal-conv and Nemotron hybrid-model kernels.
- FlashQLA, TileLang/tile_kernels, and DeepSeek-V4-specific optional kernels.
- Additional INT4/NVFP4 QAT extensions and optional fast Hadamard kernels.
  The base image's ModelOpt 0.46.1 is retained. Setuptools 79 is used for the
  Megatron build, then restored to 80.9 for ModelOpt and legacy pkg_resources.
- Megatron-Bridge conversion/LoRA paths and Muon; use the default raw conversion
  and Adam for the first smoke.
- Miles' custom Mooncake structured-object-store wheel and custom Rust router
  binary. The base Mooncake is retained and the requirements-provided Python
  router is installed; use the core/default transport or Miles Python router.
- Kubernetes tooling and nccl-tests (these are not needed to compile or import
  core training).

The build-time check covers native-package preservation, native extension
imports, and direct Miles requirements. The separate GPU smoke imports the
actual Miles training actor and checks TE backward, Megatron's selected TE
Adam, Apex Adam, and FA2 attention forward/backward. A separate fresh-process
attention check forces TE/cuDNN and verifies forward plus Q/K/V gradients.
The offline rollout smoke
imports SGLang's FlashInfer backend and compares two BF16 decode calls against
PyTorch SDPA. These do not substitute for a Miles end-to-end rollout, weight
sync, and optimizer-step test.

Sources:
[SGLang cu134 recipe](https://github.com/sgl-project/sglang/blob/a4ff563/docker/Dockerfile.cu134),
[TE 2.19 architecture handling](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/transformer_engine/common/CMakeLists.txt),
[TE build dependencies](https://github.com/NVIDIA/TransformerEngine/blob/v2.19/build_tools/pytorch.py),
[FA2 source](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post1/setup.py).
