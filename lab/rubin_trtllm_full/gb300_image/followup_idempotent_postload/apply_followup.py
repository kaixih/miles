"""Preserve the original PR image and layer an idempotent final post-load fix."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    artifacts = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("base_apply", "/opt/miles-moe/apply_trtllm_refit.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    original_provenance = Path("/opt/miles-moe/provenance.json")
    parent = json.loads(original_provenance.read_text())
    base.require(parent["pr_head"] == manifest["pr_head"], "Original PR33743 provenance mismatch")
    base.require(parent["base_image"] == manifest["upstream_base_image"],
                 "Follow-up must preserve the pinned upstream GB300 native stack")
    root = base.discover_sglang_root()
    target = root / manifest["path"]
    base.require(base.git_blob_sha(target) == manifest["before_git_blob"], "Unexpected parent unquant.py")
    patch = artifacts / "idempotent-postload.patch"
    base.require(base.sha256(patch) == manifest["patch_sha256"], "Follow-up patch checksum mismatch")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")

    def regression(label, extra):
        result = subprocess.run([sys.executable, "-B", str(artifacts / "test_real_flashinfer_postload.py"), *extra],
                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=300, check=False)
        (artifacts / (label + ".log")).write_text(result.stdout)
        print(result.stdout, end="", flush=True)
        base.require(result.returncode == 0, label + " real FlashInfer regression failed")

    before_native = base.native_snapshot()
    base.write_json(artifacts / "native-before.json", before_native)
    regression("before", ["--expect-double-pack-failure"])
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    base.require(base.git_blob_sha(target) == manifest["after_git_blob"], "Patched source mismatch")
    ast.parse(target.read_text())
    regression("after", [])
    original_tests = base.run_cpu_tests(root, artifacts)
    after_native = base.native_snapshot()
    base.write_json(artifacts / "native-after.json", after_native)
    base.require(before_native == after_native, "Native packages/binaries/ABI changed")
    base.write_json(artifacts / "provenance.json", {
        **manifest, "sglang_root": str(root),
        "original_provenance_sha256": base.sha256(original_provenance),
        "apply_followup_sha256": base.sha256(Path(__file__)),
        "regression_source_sha256": base.sha256(artifacts / "test_real_flashinfer_postload.py"),
        "expected_original_failure_reproduced": True, "real_flashinfer_cpu_tests_passed": 4,
        "original_cpu_tests": original_tests, "native_packages_and_binaries_unchanged": True,
        "gpu_validation": "not performed during build", "native_payload_count": len(after_native["native_payloads"]),
    })
    print("IDEMPOTENT_POSTLOAD_FOLLOWUP_PASS", flush=True)


if __name__ == "__main__":
    main()
