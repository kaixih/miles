"""Small, fail-closed build helpers for the pinned Rubin image."""

import argparse
import difflib
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

STATE = Path("/opt/miles-rubin")
PROTECTED_NAMES = {
    "torch", "torchvision", "torchaudio", "triton", "pytorch-triton",
    "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
    "sglang-kernel", "sgl-kernel", "sgl-deep-gemm", "sgl-deep-ep",
    "apache-tvm-ffi", "cuda-python", "cuda-bindings", "cuda-pathfinder",
}
UNPROTECTED_NVIDIA_NAMES = {
    "nvidia-cudnn-frontend", "nvidia-resiliency-ext", "nvidia-modelopt",
    "nvidia-ml-py", "nvidia-dlfw-inspect",
}


def installed():
    return {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}


def snapshot():
    import torch

    assert platform.machine() == "aarch64", platform.machine()
    assert sys.version_info[:2] == (3, 12), sys.version
    assert torch.__version__ == "2.15.0.dev20260818+cu134", torch.__version__
    assert torch.version.cuda == "13.4", torch.version.cuda
    packages = installed()
    assert packages["nvidia-cutlass-dsl"] == "4.8.0.dev0", packages["nvidia-cutlass-dsl"]
    protected = {
        name: version for name, version in packages.items()
        if name in PROTECTED_NAMES
        or (name.startswith("nvidia-") and name not in UNPROTECTED_NVIDIA_NAMES)
    }
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "native-versions.json").write_text(json.dumps(protected, indent=2) + "\n")
    (STATE / "native-constraints.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(protected.items()))
    )
    (STATE / "base-abi.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "python": sys.version, "machine": platform.machine(),
    }, indent=2) + "\n")
    print((STATE / "native-constraints.txt").read_text(), end="")


def verify_native():
    expected = json.loads((STATE / "native-versions.json").read_text())
    current = installed()
    mismatches = {name: (version, current.get(name)) for name, version in expected.items()
                  if current.get(name) != version}
    assert not mismatches, f"Native base packages changed: {mismatches}"
    import torch

    abi = json.loads((STATE / "base-abi.json").read_text())
    assert torch._C._GLIBCXX_USE_CXX11_ABI == abi["cxx11_abi"]
    print(f"Preserved {len(expected)} native base distributions and Torch ABI")


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def checkout(url, destination, commit):
    assert re.fullmatch(r"[0-9a-f]{40}", commit), "Use a complete pinned commit SHA"
    path = Path(destination)
    path.mkdir(parents=True)
    run("git", "init", destination)
    run("git", "remote", "add", "origin", url, cwd=path)
    run("git", "fetch", "--depth=1", "origin", commit, cwd=path)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=path)
    run("git", "submodule", "update", "--init", "--recursive", "--depth=1", cwd=path)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    assert actual == commit, (actual, commit)


def write_patch(path, original, updated, label):
    assert original != updated, f"No changes made for {label}; inspect the pinned source"
    path.write_text(updated)
    patch = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=f"a/{path.name}", tofile=f"b/{path.name}",
    ))
    (STATE / f"{label}.patch").write_text(patch)
    print(patch, end="")


def patch_fa2(root):
    path = Path(root) / "setup.py"
    original = path.read_text()
    target = "arch=compute_100,code=sm_100"
    assert original.count(target) == 1, "FA2 architecture code changed"
    assert '"-std=c++17"' in original, "FA2 compiler flags changed"
    updated = original.replace(target, "arch=compute_107,code=sm_107")
    updated = updated.replace("-std=c++17", "-std=c++20")
    write_patch(path, original, updated, "fa2-sm107-cxx20")


def patch_apex(root):
    path = Path(root) / "setup.py"
    original = path.read_text()
    # Optional Apex groups contain explicit C++17 flags; core groups inherit
    # Torch's C++20 flags. Keep every compiled group compatible with Torch2.15.
    updated = original.replace("-std=c++17", "-std=c++20")
    write_patch(path, original, updated, "apex-cxx20")


def checkout_sglang(commit):
    root = Path("/sgl-workspace/sglang")
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    run("git", "fetch", "--depth=1", "origin", commit, cwd=root)
    run("git", "checkout", "-f", "--detach", commit, cwd=root)
    path = root / "python/pyproject.toml"
    original = path.read_text()
    protected = json.loads((STATE / "native-versions.json").read_text())
    # Preserve requirement markers and extras while replacing ordinary CUDA
    # image pins with the exact cu134 base versions (including build requires).
    pattern = re.compile(r'^(\s*)"([^"\n]+)"(,?.*)$', re.MULTILINE)

    def reconcile(match):
        try:
            req = Requirement(match[2])
        except Exception:
            return match[0]
        name = canonicalize_name(req.name)
        if name not in protected:
            return match[0]
        extras = "[" + ",".join(sorted(req.extras)) + "]" if req.extras else ""
        marker = f"; {req.marker}" if req.marker else ""
        # TOML strings use double quotes; packaging renders marker strings
        # with double quotes too, so use single quotes inside requirement text.
        marker = marker.replace('"', "'")
        return f'{match[1]}"{req.name}{extras}=={protected[name]}{marker}"{match[3]}'

    updated = pattern.sub(reconcile, original)
    import tomllib

    parsed = tomllib.loads(updated)
    assert any("torch==2.15.0.dev20260818+cu134" in dep
               for dep in parsed["project"]["dependencies"])
    write_patch(path, original, updated, "sglang-cu134-metadata")
    (STATE / "sglang-commit.txt").write_text(commit + "\n")


def cudnn_path():
    for entry in sys.path:
        path = Path(entry) / "nvidia/cudnn"
        if (path / "include/cudnn.h").exists():
            print(path)
            return
    raise RuntimeError("The base image lacks pip cuDNN headers; inspect its cuDNN layout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "snapshot", "verify-native", "checkout", "patch-fa2", "patch-apex",
        "checkout-sglang", "cudnn-path",
    ])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    commands = {
        "snapshot": snapshot, "verify-native": verify_native, "checkout": checkout,
        "patch-fa2": patch_fa2, "patch-apex": patch_apex,
        "checkout-sglang": checkout_sglang, "cudnn-path": cudnn_path,
    }
    commands[args.command](*args.args)


if __name__ == "__main__":
    main()
