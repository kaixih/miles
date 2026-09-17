#!/usr/bin/env python3
"""Retry a corrected TRT-LLM image within the original allocation deadline."""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--registry-tag", required=True)
    args = parser.parse_args()
    parent = args.parent_root.resolve(strict=True)
    original = json.loads((parent / "state.json").read_text())
    if original["stage"] != "completed_with_failures":
        raise RuntimeError("Original attempt must be terminal and retained")
    failed = original["backends"]["flashinfer_trtllm"]
    if failed["status"] != "FAILED" or failed["ray_status"] != "FAILED":
        raise RuntimeError("Expected the recorded terminal TRT-LLM failure")
    if not args.image.startswith("sha256:") or len(args.image) != 71:
        raise ValueError("Use the immutable corrected image ID")
    if not args.registry_tag.startswith(
        "gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:"
    ):
        raise ValueError("Registry target must remain in the existing repository")
    module_path = parent / "run_two_backends.py"
    if hashlib.sha256(module_path.read_bytes()).hexdigest() != (
        "9591bf1be8ed6b65d6c461ed52e641dff1e943f1365fc831950d76ee17cf0c4d"
    ):
        raise RuntimeError("Original audited runner changed")
    spec = importlib.util.spec_from_file_location("original_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.root.mkdir()  # Do not reuse or overwrite an earlier attempt.
    root = args.root.resolve(strict=True)
    (root / "inputs").mkdir()
    for name, digest in module.INPUT_HASHES.items():
        data = (parent / "inputs" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Dataset changed: {name}")
        (root / "inputs" / name).write_bytes(data)
    shutil.copyfile(parent / "triton-baseline-first2.json", root / "triton-baseline-first2.json")
    runner_args = types.SimpleNamespace(
        job_id=original["job_id"], root=root,
        source=Path(original["source"]), build_context=parent / "build-context",
        base_image=original["base_image"], models=Path("/home/scratch.kaixih_ent/models"),
        registry_tag=args.registry_tag,
    )

    class RetryRunner(module.Runner):
        def orchestrator(self, *values):
            command = super().orchestrator(*values)
            i = command.index("--cache-dir") + 1
            command[i] = str(Path(self.local) / "trtllm-postload-retry-cache")
            return command

    runner = RetryRunner(runner_args)
    runner.node = original["node"]
    runner.local = original["local_root"]
    runner.ip = json.loads((parent / "storage.json").read_text())["node_ip"]
    runner.lease_end = dt.datetime.fromisoformat(original["lease_end_utc"]).timestamp()
    runner.deadline = dt.datetime.fromisoformat(original["deadline_utc"]).timestamp()
    runner.receipt.update(
        parent_attempt=str(parent), corrected_image=args.image, node=runner.node,
        local_root=runner.local, lease_end_utc=original["lease_end_utc"],
        deadline_utc=original["deadline_utc"],
        change="PR33743 plus idempotent final post-load repacking",
    )
    (root / "runner.lock").mkdir()
    module.write_json(root / "runner.lock/identity.json", {
        "pid": os.getpid(), "uid": os.getuid(), "created_utc": module.utc(),
        "job_id": runner.a.job_id,
    })
    try:
        runner.require_own_node()
        if runner.deadline - time.time() < 35 * 60:
            raise RuntimeError("Less than 35 minutes remain in original experiment budget")
        runner.require_idle_gpus()
        provenance_code = (
            "from pathlib import Path; "
            "print(Path('/opt/miles-moe/followup-idempotent-postload/provenance.json').read_text())"
        )
        _, provenance_text = runner.remote([
            "docker", "run", "--rm", "--network", "none", "--user", "28644:30",
            args.image, "python3", "-B", "-c", provenance_code,
        ], timeout=60)
        provenance = json.loads(provenance_text)
        if not (
            provenance.get("expected_original_failure_reproduced") is True
            and provenance.get("real_flashinfer_cpu_tests_passed") == 4
            and provenance.get("native_packages_and_binaries_unchanged") is True
        ):
            raise RuntimeError("Corrected image lacks passing real-FlashInfer build evidence")
        module.write_json(root / "corrected-image-provenance.json", provenance)
        code = """import hashlib,json,pathlib,os
c=json.loads(CONFIG)
assert (os.getuid(),os.getgid())==(28644,30)
local=pathlib.Path(c['local']); assert local.stat().st_uid==28644
cache=local/'trtllm-postload-retry-cache'; cache.mkdir(mode=0o700)
for name in c['models']:
    root=local/'models'/name
    inventory={str(p.relative_to(root)):p.stat().st_size for p in root.rglob('*') if p.is_file()}
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    assert digest==c['models'][name]['inventory_sha256'],name
print(json.dumps({'reused_models_verified':True,'fresh_cache':str(cache)}))
"""
        config = {"local": runner.local, "models": {
            name: json.loads((parent / f"model-{name}.json").read_text())
            for name in module.MODELS
        }}
        _, output = runner.remote(["python3", "-c", code.replace("CONFIG", repr(json.dumps(config)))], timeout=120)
        module.write_json(root / "storage-reuse.json", json.loads(output.strip().splitlines()[-1]))
        runner.backend("flashinfer_trtllm", args.image)
        passed = runner.receipt["backends"]["flashinfer_trtllm"]["status"] == "SUCCEEDED"
        if passed:
            runner.push_image(args.image)
        runner.state("completed" if passed else "completed_with_failures", finished_utc=module.utc())
        return 0 if passed else 1
    except BaseException as exc:
        runner.state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runner.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
