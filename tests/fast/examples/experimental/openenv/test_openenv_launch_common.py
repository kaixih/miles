"""Offline tests for what the launcher itself owns of the credential wiring.

The contract (path-not-value key supply, address forwarding, SDK preflight)
lives in miles.rollout.agentic.credentials and is tested at
tests/fast/rollout/agentic/test_credentials.py. What stays here is the
launcher's own obligation: every backend the example registers must be wired
to a complete credential spec it can act on.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import openenv_launch_common as launch
import openenv_sandbox_common as common
import pytest


def test_every_backend_has_a_credential_spec():
    """A backend registered without credential wiring would fail at launch with
    a KeyError instead of telling the operator what to provision."""
    assert set(launch.PROVIDER_CREDENTIALS) == set(common.AGENT_MODULES)


@pytest.mark.parametrize("backend", sorted(launch.PROVIDER_CREDENTIALS))
def test_every_spec_names_a_launcher_arg(backend):
    """The arg the launcher reads must be declared on the shared config Protocol,
    else a launcher can never override the key-file path."""
    assert launch.PROVIDER_CREDENTIALS[backend]["arg_attr"] in launch.LaunchArgs.__annotations__, backend


@pytest.mark.parametrize(
    ("arg_path", "env_path", "expected_path"),
    [
        (None, "/keys/from-env", "/keys/from-env"),
        ("", "/keys/from-env", "/keys/from-env"),
        ("/keys/from-arg", "/keys/from-env", "/keys/from-arg"),
        (None, None, ""),
    ],
)
def test_key_file_precedence_supports_launchers_without_the_provider_arg(
    monkeypatch, tmp_path, arg_path, env_path, expected_path
):
    args = SimpleNamespace(miles_host_ip="", openenv_sandbox_backend="opensandbox", openenv_tb2_tasks_dir="/tasks")
    if arg_path is not None:
        args.opensandbox_api_key_file = arg_path
    if env_path is None:
        monkeypatch.delenv("OPEN_SANDBOX_API_KEY_FILE", raising=False)
    else:
        monkeypatch.setenv("OPEN_SANDBOX_API_KEY_FILE", env_path)

    env_package = ModuleType("tbench2_env")
    env_package.__file__ = str(tmp_path / "__init__.py")
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "tbench2_env_environment.py").write_text("TB2_WITHHOLD_TESTS\n_require_canonical_verdict\n")
    monkeypatch.setitem(sys.modules, "tbench2_env", env_package)
    provision = Mock()
    monkeypatch.setattr(launch, "provision_provider", provision)

    env: dict[str, str] = {}
    launch.apply_optional_env_vars(env, args)

    provision.assert_called_once_with(env, launch.PROVIDER_CREDENTIALS["opensandbox"], arg_path=expected_path)
    assert env["OPENENV_TB2_TASKS_DIR"] == "/tasks"
