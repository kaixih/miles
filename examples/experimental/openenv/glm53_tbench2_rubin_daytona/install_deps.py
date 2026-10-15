#!/usr/bin/env python3
"""Install the Daytona + OpenEnv client stack into the run container, without moving the image's stack.

Run inside the container. `plan` and `apply` as root (docker exec --user 0), `verify` as the run UID.

  plan    Resolve the requested packages against a constraints file that pins every package the image
          already has; only openai and mcp may move, to the floors the OpenEnv/fastmcp stack needs.
          Writes a hash-locked requirements file under /opt/tb2-daytona-install. Changes nothing.
  apply   Installs exactly the reviewed lock (--no-deps --require-hashes), installs OpenEnv and
          tbench2_env editable from a copy at /opt/OpenEnv, relocates Megatron-LM to /opt when /root
          is unreadable for the run UID, applies the one-file snapshot-retry patch to /opt/miles.
  verify  Imports what the launcher and the Daytona provider need and records versions and paths.

Inputs staged on /work by the host side: /work/OpenEnv (checkout at the pinned commit),
/work/inputs/initial-snapshot-retry.patch, /work/inputs/known-versions.json.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

MILES = Path("/opt/miles")
MCORE_IMAGE = Path("/root/Megatron-LM")
MCORE_SHARED = Path("/opt/Megatron-LM")
SGLANG = Path("/sgl-workspace/sglang")
OPENENV_STAGED = Path("/work/OpenEnv")
OPENENV = Path("/opt/OpenEnv")
PATCH = Path("/work/inputs/initial-snapshot-retry.patch")
KNOWN = Path("/work/inputs/known-versions.json")
INSTALL = Path("/opt/tb2-daytona-install")

SOURCE_HEADS = {
    MILES: "0d9d726e0f9b9d487c36a0c342012b917820fb04",
    MCORE_IMAGE: "fd15ee20a4f03b03680529baf4b8f7eeed64df1d",
    SGLANG: "6b5f3dfc5a395d6fd56d105c34c8e3542846ca4c",
    OPENENV_STAGED: "38b2a31354d1d4d627894412b9d469e92f4a2c61",
}
# Pins the old run validated, plus the Daytona SDK the key smoke test used.
TARGETS = {"openai": "2.7.2", "mcp": "1.30.0", "fastmcp": "3.4.8", "daytona": "0.224.1"}
# Image packages that may be replaced, and the range that makes the replacement necessary.
COMPATIBLE = {"openai": ">=2.7.2", "mcp": ">=1.24,<2.0"}
EDITABLES = {"openenv": "0.4.2.dev0", "openenv-tbench2-env": "0.1.0"}
NOT_REQUESTED = ["gradio>=4.0.0", "camel-ai>=0.2.90", "docker>=7.1.0"]  # server-side extras the client never imports


def run(argv, **kwargs):
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, timeout=1800, **kwargs)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def packages():
    return {normalized(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}


def git(root, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root), *args], text=True, timeout=60)


def version_ok(version, spec):
    from packaging.specifiers import SpecifierSet

    return SpecifierSet(spec).contains(version, prereleases=True)


def check_sources():
    for root, head in SOURCE_HEADS.items():
        if git(root, "rev-parse", "HEAD").strip() != head:
            raise RuntimeError(f"{root} is not at {head}")
        if git(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"{root} has tracked modifications")
    git(MILES, "apply", "--check", str(PATCH))


def plan(args):
    if os.getuid() != 0 or not Path("/.dockerenv").is_file():
        raise RuntimeError("plan runs as root inside the run container")
    check_sources()
    before = packages()
    if set(before) & set(EDITABLES):
        raise RuntimeError("OpenEnv packages already installed; inspect them instead of reinstalling")
    known = {normalized(k): v for k, v in json.loads(KNOWN.read_text()).items()}
    declared = tomllib.loads((OPENENV_STAGED / "pyproject.toml").read_text())["project"]["dependencies"]

    constraints, replacements = dict(before), {}
    for name, target in TARGETS.items():
        if name not in before:
            constraints[name] = target
        elif name in COMPATIBLE and not version_ok(before[name], COMPATIBLE[name]):
            constraints[name] = target
            replacements[name] = {"before": before[name], "after": target, "required": COMPATIBLE[name]}
        elif name not in COMPATIBLE and before[name] != target:
            raise RuntimeError(f"image has {name}=={before[name]}, pin is {target}; review before changing")
    for name, version in known.items():
        constraints.setdefault(name, version)
    requested = [d for d in declared if d not in NOT_REQUESTED] + [f"{n}=={constraints[n]}" for n in TARGETS]

    INSTALL.mkdir(mode=0o755)
    (INSTALL / "before.json").write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
    constraints_path = INSTALL / "constraints.txt"
    constraints_path.write_text("".join(f"{n}=={v}\n" for n, v in sorted(constraints.items())))
    report_path = INSTALL / "pip-plan.json"
    run([sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report_path),
         "--only-binary=:all:", "-c", str(constraints_path), *requested])
    if packages() != before:
        raise RuntimeError("package inventory changed during planning")

    wheels = INSTALL / "wheels"
    wheels.mkdir()
    locked, planned, downloaded = [], {}, {}
    for item in json.loads(report_path.read_text())["install"]:
        name, version = normalized(item["metadata"]["name"]), item["metadata"]["version"]
        if name in before and name not in replacements and before[name] != version:
            raise RuntimeError(f"resolver changed a frozen image package: {name} {before[name]} -> {version}")
        if name in constraints and constraints[name] != version:
            raise RuntimeError(f"resolver ignored constraint: {name}")
        digest = item["download_info"]["archive_info"].get("hashes", {}).get("sha256")
        if digest:
            locked.append(f"{name}=={version} --hash=sha256:{digest}\n")
        else:
            # The index served no digest; fetch the exact wheel now and lock the bytes we fetched.
            run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--dest", str(wheels),
                 "-c", str(constraints_path), f"{name}=={version}"])
            (wheel,) = [p for p in wheels.glob("*.whl") if normalized(p.name.split("-")[0]) == name]
            digest = sha(wheel)
            downloaded[name] = {"file": str(wheel), "sha256": digest, "report_url": item["download_info"]["url"]}
            locked.append(f"{name} @ {wheel.as_uri()} --hash=sha256:{digest}\n")
        planned[name] = version
    lock = INSTALL / "resolved-requirements.txt"
    lock.write_text("".join(sorted(locked)))
    result = {
        "status": "AWAITING_PLAN_REVIEW", "node": os.uname().nodename, "installer_sha256": sha(__file__), "before": before,
        "constraints_sha256": sha(constraints_path), "pip_plan_sha256": sha(report_path), "lock_sha256": sha(lock),
        "requested": requested, "planned_versions": planned, "permitted_replacements": replacements,
        "new_packages": sorted(set(planned) - set(before)),
        "new_packages_not_in_known_versions": sorted(set(planned) - set(known) - set(before)),
        "wheels_downloaded_for_hashing": downloaded, "editable_installs": EDITABLES, "not_requested": NOT_REQUESTED,
    }
    out = INSTALL / "installation-plan.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "plan": str(out), "plan_sha256": sha(out),
                      "new_packages": result["new_packages"], "replacements": replacements}, indent=2))


def make_world_readable(root):
    """a+rX on a tree copied by root so the run UID can import from it (mirrors the recipe Dockerfile)."""
    for dirpath, dirnames, filenames in os.walk(root):
        for entry in dirnames + filenames:
            path = Path(dirpath) / entry
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o444 | (0o111 if path.is_dir() else 0))
    Path(root).chmod(Path(root).stat().st_mode | 0o555)


def uid_can_read(path, uid, gid):
    probe = subprocess.run([sys.executable, "-I", "-c", f"import os,sys; sys.exit(0 if os.access({str(path)!r}, os.R_OK) else 1)"],
                           user=uid, group=gid, extra_groups=[], timeout=30)
    return probe.returncode == 0


def apply(args):
    if os.getuid() != 0 or not Path("/.dockerenv").is_file():
        raise RuntimeError("apply runs as root inside the run container")
    plan_path = INSTALL / "installation-plan.json"
    if sha(plan_path) != args.approved_plan_sha256:
        raise RuntimeError("the exact reviewed installation-plan SHA is required")
    approved = json.loads(plan_path.read_text())
    before = packages()
    if approved["installer_sha256"] != sha(__file__) or approved["before"] != before:
        raise RuntimeError("installer or package inventory changed after review")
    for name, key in (("constraints.txt", "constraints_sha256"), ("pip-plan.json", "pip_plan_sha256"),
                      ("resolved-requirements.txt", "lock_sha256")):
        if sha(INSTALL / name) != approved[key]:
            raise RuntimeError(f"reviewed input changed: {name}")
    check_sources()
    # Every precondition before the first mutation: a half-applied container cannot be re-run.
    if OPENENV.exists():
        raise RuntimeError(f"{OPENENV} already exists")
    relocate_mcore = not uid_can_read(MCORE_IMAGE / "megatron/core/__init__.py", args.uid, args.gid)
    if relocate_mcore and MCORE_SHARED.exists():
        raise RuntimeError(f"{MCORE_SHARED} already exists")
    if (INSTALL / "apply-started.json").exists():
        raise RuntimeError("apply already started in this container; inspect it instead of re-running")
    (INSTALL / "apply-started.json").write_text(json.dumps({"approved_plan_sha256": args.approved_plan_sha256}) + "\n")

    if approved["planned_versions"]:
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--require-hashes",
             "-r", str(INSTALL / "resolved-requirements.txt")])
    shutil.copytree(OPENENV_STAGED, OPENENV, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    make_world_readable(OPENENV)
    if git(OPENENV, "rev-parse", "HEAD").strip() != SOURCE_HEADS[OPENENV_STAGED] or git(OPENENV, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("copied OpenEnv tree differs from the pinned commit")
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(OPENENV),
         "-e", str(OPENENV / "envs/tbench2_env")])

    mcore = MCORE_IMAGE
    if relocate_mcore:
        shutil.copytree(MCORE_IMAGE, MCORE_SHARED, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        make_world_readable(MCORE_SHARED)
        if git(MCORE_SHARED, "rev-parse", "HEAD").strip() != SOURCE_HEADS[MCORE_IMAGE] or git(MCORE_SHARED, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("relocated Megatron-LM differs from the image checkout")
        mcore = MCORE_SHARED
    for root in (MILES, OPENENV, mcore):
        run(["git", "config", "--system", "--add", "safe.directory", str(root)])

    git(MILES, "apply", str(PATCH))
    changed = git(MILES, "diff", "--name-only", "HEAD", "--").split()
    if changed != ["miles/ray/rollout/server_cell.py"]:
        raise RuntimeError(f"unexpected Miles changes after the snapshot patch: {changed}")

    after = packages()
    expected = before | approved["planned_versions"] | EDITABLES
    if after != expected:
        raise RuntimeError(f"installed packages differ from the reviewed plan: {sorted(set(after.items()) ^ set(expected.items()))}")
    receipt = {
        "status": "INSTALLED_PENDING_RUN_UID_VERIFY", "node": os.uname().nodename, "approved_plan_sha256": args.approved_plan_sha256,
        "added_packages": {n: v for n, v in after.items() if n not in before},
        "changed_existing_packages": {n: {"before": v, "after": after[n]} for n, v in before.items() if after[n] != v},
        "megatron_path": str(mcore), "miles_diff_stat": git(MILES, "diff", "HEAD", "--stat"),
        "server_cell_sha256": sha(MILES / "miles/ray/rollout/server_cell.py"),
    }
    (INSTALL / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


def verify(args):
    receipt = json.loads((INSTALL / "receipt.json").read_text())
    mcore = receipt["megatron_path"]
    sys.path[:0] = [mcore, str(MILES / "examples/experimental/openenv"), str(MILES)]
    expected_paths = {
        "openenv": str(OPENENV / "src/openenv/__init__.py"),
        "tbench2_env": str(OPENENV / "envs/tbench2_env/__init__.py"),
        "megatron.core": f"{mcore}/megatron/core/__init__.py",
        "miles": str(MILES / "miles/__init__.py"),
        "openenv_daytona_agent_function": str(MILES / "examples/experimental/openenv/openenv_daytona_agent_function.py"),
        "tb2_sandbox_daytona": str(MILES / "examples/experimental/openenv/tb2_sandbox_daytona.py"),
        "miles.rollout.agentic.credentials": str(MILES / "miles/rollout/agentic/credentials.py"),
    }
    imports = {}
    for name in [*expected_paths, "daytona", "fastmcp", "openai", "mcp", "typer", "packaging"]:
        module = importlib.import_module(name)
        imports[name] = str(Path(module.__file__).resolve())
        if name in expected_paths and imports[name] != expected_paths[name]:
            raise RuntimeError(f"{name} imported from {imports[name]}, expected {expected_paths[name]}")
    import tbench2_env

    server_src = (Path(tbench2_env.__file__).resolve().parent / "server/tbench2_env_environment.py").read_text()
    if "TB2_WITHHOLD_TESTS" not in server_src or "_require_canonical_verdict" not in server_src:
        raise RuntimeError("tbench2_env server lacks the #1025 canonical-evaluate contract")
    if not (Path(tbench2_env.__file__).resolve().parent / "pyproject.toml").is_file():
        raise RuntimeError("tbench2_env is not an editable checkout install; the sandbox recipe embeds its source")
    from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, Image, Resources  # noqa: F401

    from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provision_provider  # noqa: F401
    from miles.utils.external_utils.model_args_utils import load_model_args

    model_args = load_model_args("glm5.3-flash").split()
    layers = model_args[model_args.index("--num-layers") + 1]
    versions = packages()
    result = {
        "status": "RUN_UID_IMPORTS_PASSED", "uid": os.getuid(), "gid": os.getgid(), "python": sys.executable,
        "pythonpath": sys.path[:3], "imports": imports, "glm53_num_layers": layers,
        "versions": {n: versions.get(n) for n in ("torch", "ray", "transformer-engine", "transformers", "sglang", "triton",
                                                   "openenv", "openenv-tbench2-env", "daytona", "fastmcp", "openai", "mcp")},
        "miles_patched_server_cell_sha256": sha(MILES / "miles/ray/rollout/server_cell.py"),
        "nofile": list(__import__("resource").getrlimit(__import__("resource").RLIMIT_NOFILE)),
    }
    print("VERIFY_JSON=" + json.dumps(result, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "apply", "verify"])
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--uid", type=int, help="UID the run containers execute as (required for apply)")
    parser.add_argument("--gid", type=int, help="GID of that run user (required for apply)")
    args = parser.parse_args()
    if args.mode == "apply" and (args.uid is None or args.gid is None):
        parser.error("--uid and --gid (the run UID/GID) are required for apply")
    if args.mode == "apply" and not args.approved_plan_sha256:
        parser.error("apply requires --approved-plan-sha256")
    os.environ["PIP_CACHE_DIR"] = str(INSTALL / "pip-cache")
    {"plan": plan, "apply": apply, "verify": verify}[args.mode](args)


if __name__ == "__main__":
    main()
