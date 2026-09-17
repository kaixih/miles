"""Apply only pinned PR33743, check its CPU tests, and preserve native packages."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_IMAGE = (
    "gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:"
    "a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c"
)
PR_HEAD = "89f25a0b9772576fbd427a495a1144f544c90928"
PATCH_SHA256 = "2137ce6f3ec431e70f0ffd4e7e876070255a3e9f1488ece1c5346b1111fa908d"
TEST_PATH = "test/registered/unit/layers/quantization/test_flashinfer_trtllm_bf16_moe_reload.py"
SOURCE_PATHS = (
    "python/sglang/srt/layers/quantization/base_config.py",
    "python/sglang/srt/layers/quantization/unquant.py",
    "python/sglang/srt/model_executor/model_runner_components/weight_updater.py",
)
NATIVE_NAMES = {
    "torch", "torchvision", "torchaudio", "triton", "pytorch-triton",
    "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
    "sglang-kernel", "sgl-kernel", "sgl-deep-gemm", "sgl-deep-ep",
    "transformer-engine", "transformer-engine-torch", "transformer-engine-cu13",
    "flash-attn", "apex", "torch-memory-saver", "apache-tvm-ffi",
    "cuda-python", "cuda-bindings", "cuda-pathfinder",
}
EDITABLE_NATIVE_MODULES = (
    "torch", "triton", "flashinfer", "sgl_kernel", "transformer_engine",
    "flash_attn", "apex", "torch_memory_saver",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def discover_sglang_root() -> Path:
    # Top-level find_spec locates the editable package without importing SGLang.
    spec = importlib.util.find_spec("sglang")
    require(spec is not None, "SGLang is not installed in the active Python.")
    locations = list(spec.submodule_search_locations or [])
    require(len(locations) == 1, f"Ambiguous SGLang package locations: {locations}")
    package = Path(locations[0]).resolve()
    root = package.parent.parent
    require(
        (root / "python/sglang").resolve() == package,
        f"Expected an editable SGLang python/sglang source tree, found {package}",
    )
    require(all((root / name).is_file() for name in SOURCE_PATHS),
            f"Incomplete SGLang source tree at {root}")
    return root


def validate_inputs(root: Path, artifact_dir: Path) -> tuple[dict, Path, dict]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    require(manifest["pr_head"] == PR_HEAD, "Unexpected PR head in manifest")
    patch = artifact_dir / manifest["patch"]
    require(patch.parent.resolve() == artifact_dir.resolve(), "Patch escapes artifact directory")
    require(sha256(patch) == PATCH_SHA256 == manifest["patch_sha256"],
            "Pinned patch checksum mismatch")
    sources = {item["path"]: item for item in manifest["fetched_sources"]
               if item["label"] == "image-base"}
    require(set(sources) == set(SOURCE_PATHS), "Expected exactly three original source blobs")
    before = {}
    for name, entry in sources.items():
        path = root / name
        actual = git_blob_sha(path)
        require(actual == entry["git_blob_sha"],
                f"Refusing changed base file {path}: {actual} != {entry['git_blob_sha']}")
        before[name] = {"git_blob_sha": actual, "sha256": sha256(path)}
    require(not (root / TEST_PATH).exists(), "PR33743 test already exists; refuse double application")
    return manifest, patch, before


def apply_source_patch(root: Path, patch: Path, artifact_dir: Path) -> dict:
    output = []
    for check in (True, False):
        command = ["git", "apply", "--verbose"]
        if check:
            command.append("--check")
        command.append(str(patch.resolve()))
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        output.append("$ " + " ".join(command) + "\n" + result.stdout)
        (artifact_dir / "apply.log").write_text("\n".join(output))
        require(result.returncode == 0, f"git apply failed; see {artifact_dir / 'apply.log'}")
    after = {}
    for name in (*SOURCE_PATHS, TEST_PATH):
        path = root / name
        ast.parse(path.read_text(), filename=str(path))
        after[name] = {"git_blob_sha": git_blob_sha(path), "sha256": sha256(path)}
    return after


def is_native_file(path: Path) -> bool:
    return bool(re.search(r"\.so(?:\.\d+)*$", path.name)) or path.suffix in {
        ".cubin", ".fatbin", ".ptx", ".a", ".dylib", ".dll",
    }


def native_snapshot() -> dict:
    """Record installed metadata and hash native payloads, including editable packages."""
    distributions = []
    native_distributions = []
    native_paths = set()
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        item = {"name": name, "version": dist.version,
                "location": str(Path(dist.locate_file("")).resolve())}
        for metadata_file in ("METADATA", "RECORD", "direct_url.json"):
            text = dist.read_text(metadata_file)
            item[metadata_file + "_sha256"] = (
                hashlib.sha256(text.encode()).hexdigest() if text is not None else None
            )
        distributions.append(item)
        payloads = [Path(dist.locate_file(f)).resolve() for f in (dist.files or [])
                    if is_native_file(Path(str(f)))]
        if payloads or name in NATIVE_NAMES or name.startswith("nvidia-"):
            native_distributions.append(item)
            native_paths.update(payloads)
    # Editable builds may omit the source-tree extension from wheel RECORD.
    for module in EDITABLE_NATIVE_MODULES:
        spec = importlib.util.find_spec(module)
        if spec is None:
            continue
        if spec.origin and is_native_file(Path(spec.origin)):
            native_paths.add(Path(spec.origin).resolve())
        for location in spec.submodule_search_locations or []:
            native_paths.update(path.resolve() for path in Path(location).rglob("*")
                                if path.is_file() and is_native_file(path))
    payloads = {}
    for path in sorted(native_paths):
        payloads[str(path)] = ({"bytes": path.stat().st_size, "sha256": sha256(path)}
                              if path.is_file() else {"missing": True})
    import torch

    abi = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
           "machine": platform.machine(), "python": sys.version}
    order = lambda item: (item["name"], item["version"], item["location"])
    return {"all_distributions": sorted(distributions, key=order),
            "native_distributions": sorted(native_distributions, key=order),
            "native_payloads": payloads, "abi": abi}


def run_cpu_tests(root: Path, artifact_dir: Path) -> dict:
    env = os.environ.copy()
    # SGLang test_utils indexes the first CUDA_VISIBLE_DEVICES character for a
    # port offset. An empty string fails at import; Docker build has no GPUs.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({"PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "disabled"})
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    bootstrap = (
        "import runpy,sys,torch; "
        "assert not torch.cuda.is_available() and torch.cuda.device_count() == 0, "
        "'CPU build tests unexpectedly see a GPU'; "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    command = [sys.executable, "-B", "-c", bootstrap, str(root / TEST_PATH), "-v"]
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False, timeout=600)
    (artifact_dir / "cpu-test.log").write_text(result.stdout)
    print(result.stdout, end="", flush=True)
    count = re.search(r"^Ran (\d+) tests? in ", result.stdout, flags=re.MULTILINE)
    require(result.returncode == 0 and count is not None and int(count[1]) == 11,
            f"PR33743 CPU tests failed or did not run all 11 tests; see {artifact_dir / 'cpu-test.log'}")
    require(re.search(r"^OK\s*$", result.stdout, flags=re.MULTILINE) is not None,
            "Tests skipped or not fully successful; inspect cpu-test.log")
    return {"command": command, "returncode": result.returncode, "tests": int(count[1]),
            "elapsed_seconds": time.monotonic() - started,
            "log_sha256": sha256(artifact_dir / "cpu-test.log")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args()
    require(args.base_image == BASE_IMAGE, "Derivative must use the exact baseline image digest")
    artifact_dir = Path(__file__).resolve().parent
    root = discover_sglang_root()
    manifest, patch, before_source = validate_inputs(root, artifact_dir)
    print(f"Verified the three original SGLang blobs in {root}", flush=True)
    print("Hashing native distributions and binaries before the Python patch", flush=True)
    before_native = native_snapshot()
    write_json(artifact_dir / "native-before.json", before_native)
    print("Applying pinned PR33743 and running its 11 CPU tests", flush=True)
    after_source = apply_source_patch(root, patch, artifact_dir)
    test = run_cpu_tests(root, artifact_dir)
    print("Checking native distributions and binaries are unchanged", flush=True)
    after_native = native_snapshot()
    write_json(artifact_dir / "native-after.json", after_native)
    require(after_native == before_native, "Native/package inventory changed; inspect native-{before,after}.json")
    provenance = {
        "base_image": BASE_IMAGE, "sglang_root": str(root),
        "image_sglang_ref": manifest["image_sglang_ref"],
        "pr_url": manifest["pr_url"], "pr_head": PR_HEAD,
        "patch_sha256": PATCH_SHA256, "apply_script_sha256": sha256(Path(__file__)),
        "source_before": before_source, "source_after": after_source,
        "cpu_tests": test, "native_distributions_unchanged": True,
        "native_payloads_unchanged": True,
        "native_payload_count": len(after_native["native_payloads"]),
        "native_snapshot_sha256": sha256(artifact_dir / "native-after.json"),
        "torch_abi": after_native["abi"],
        "gpu_validation": "not performed during image build",
    }
    write_json(artifact_dir / "provenance.json", provenance)
    print(json.dumps({"status": "passed", "pr_head": PR_HEAD,
                      "cpu_tests": test["tests"], "sglang_root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
