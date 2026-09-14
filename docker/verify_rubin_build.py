"""Build-time native-package, extension-import, and direct-requirement checks."""

import importlib
import importlib.metadata as metadata

from packaging.requirements import Requirement
from packaging.version import Version

from rubin_build import STATE, verify_native


def verify_imports():
    verify_native()
    required = [
        "torch", "flashinfer", "sglang", "transformer_engine.pytorch",
        "apex", "amp_C", "fused_layer_norm_cuda", "flash_attn", "flash_attn_2_cuda",
        "megatron.core", "mbridge", "torch_memory_saver", "ray", "sglang_router", "miles",
    ]
    for name in required:
        importlib.import_module(name)
        print(f"Import passed: {name}")
    assert Version(metadata.version("transformer-engine")).release[:2] == (2, 19)
    assert Version(metadata.version("flash-attn")).base_version == "2.8.3"
    from cutlass.base_dsl.enums import Arch

    assert Arch["sm_107a"]
    problems = []
    for line in (STATE / "miles-requirements.txt").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            problems.append(f"Missing: {req}")
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            problems.append(f"{req}: installed {version}")
    assert not problems, "\n".join(problems)
    print("All direct Miles requirements satisfied")


if __name__ == "__main__":
    verify_imports()
