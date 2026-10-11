"""Offline coverage for OpenSandbox credentials, partial creates, and cleanup.

The SDK and HTTP transport are replaced here; no provider calls are made.
"""

import json
import ssl
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import tb2_sandbox_opensandbox as sandbox


class _ApiError(Exception):
    def __init__(self, message="provider error", *, status_code):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """A create response can arrive before SDK initialization fails."""
    settings = sandbox.Settings("https://sandbox.test", "private-test-key", ssl.create_default_context())
    transport = Mock(spec=httpx.BaseTransport)
    transport.handle_request.return_value = httpx.Response(201, json={"id": "sb-123"})
    manager = Mock()
    manager.get_sandbox_info.side_effect = _ApiError(status_code=404)
    manager.list_sandbox_infos.return_value = SimpleNamespace(sandbox_infos=[])
    instance = Mock()
    instance.commands.run.return_value = SimpleNamespace(exit_code=0, error=None)
    instance.get_endpoint.return_value = SimpleNamespace(
        endpoint="https://sandbox.test/v1/sandboxes/sb-123/proxy/8000", headers={}
    )
    health = Mock()
    health.get.return_value = httpx.Response(200)
    health_client = Mock()
    health_client.return_value.__enter__ = Mock(return_value=health)
    health_client.return_value.__exit__ = Mock(return_value=False)

    def create(image, *, connection_config, **kwargs):
        connection_config.transport.handle_request(httpx.Request("POST", "https://sandbox.test/v1/sandboxes"))
        return instance

    sdk_create = Mock(side_effect=create)
    sdk_manager = Mock(return_value=manager)
    for name, attributes in {
        "opensandbox.config": {"ConnectionConfigSync": SimpleNamespace},
        "opensandbox.sync": {
            "SandboxSync": SimpleNamespace(create=sdk_create),
            "SandboxManagerSync": SimpleNamespace(create=sdk_manager),
        },
        "opensandbox.exceptions": {"SandboxApiException": _ApiError},
        "opensandbox.models.sandboxes": {"SandboxFilter": SimpleNamespace},
        "opensandbox.models.execd": {"RunCommandOpts": SimpleNamespace},
        "opensandbox.models.execd_sync": {"ExecutionHandlersSync": SimpleNamespace},
    }.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("OPENENV_OPENSANDBOX_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sandbox, "connection_settings", lambda: settings)
    monkeypatch.setattr(sandbox.httpx, "HTTPTransport", Mock(return_value=transport))
    monkeypatch.setattr(sandbox.httpx, "Client", health_client)
    monkeypatch.setattr(sandbox.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sandbox.recipe, "server_layer_commands", lambda task_dir: ["install-server"])
    monkeypatch.setattr(sandbox.recipe, "task_env_resources", lambda task_dir: (1, 2048, 10240))
    monkeypatch.setattr(sandbox.recipe, "resolve_docker_image", lambda task_dir, override: "task:fixed")
    monkeypatch.setattr(sandbox.recipe, "sandbox_labels", lambda task_dir: {"task": task_dir.name})
    monkeypatch.setattr(sandbox.recipe, "server_cmd", lambda **kwargs: "serve-task")
    return SimpleNamespace(
        settings=settings,
        transport=transport,
        manager=manager,
        instance=instance,
        health=health,
        health_client=health_client,
        sdk_create=sdk_create,
        sdk_manager=sdk_manager,
        create=create,
        log_root=tmp_path,
    )


def _report(runtime):
    reports = list(runtime.log_root.glob("*/lifecycle.json"))
    assert len(reports) == 1
    return json.loads(reports[0].read_text())


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://sandbox.test",
        "https://user:secret@sandbox.test",
        "https://sandbox.test/api",
        "https://sandbox.test?token=secret",
        "https://sandbox.test#fragment",
    ],
)
def test_configuration_requires_a_bare_https_origin(monkeypatch, endpoint):
    monkeypatch.setenv("OPEN_SANDBOX_API_URL", endpoint)
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "private-test-key")
    monkeypatch.delenv("OPEN_SANDBOX_CA_FILE", raising=False)
    with pytest.raises(ValueError, match="HTTPS"):
        sandbox.connection_settings()


def test_settings_repr_does_not_expose_the_api_key(monkeypatch):
    monkeypatch.setenv("OPEN_SANDBOX_API_URL", "https://sandbox.test")
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "private-test-key")
    monkeypatch.delenv("OPEN_SANDBOX_CA_FILE", raising=False)
    settings = sandbox.connection_settings()
    assert "private-test-key" not in repr(settings)
    assert settings.tls.verify_mode == ssl.CERT_REQUIRED
    assert settings.tls.check_hostname


@pytest.mark.parametrize(
    "url",
    [
        "http://sandbox.test/v1/sandboxes",
        "https://other.test/v1/sandboxes",
        "https://sandbox.test:8443/v1/sandboxes",
        "https://user:secret@sandbox.test/v1/sandboxes",
    ],
)
def test_transport_refuses_credentials_leaving_the_configured_origin(runtime, url):
    lease = sandbox._SandboxLease(runtime.settings, "task")
    with pytest.raises(ValueError, match="origin"):
        lease.handle_request(httpx.Request("GET", url, headers={"OPEN-SANDBOX-API-KEY": "private-test-key"}))
    runtime.transport.handle_request.assert_not_called()
    lease.close()


def test_create_post_is_not_repeated_when_its_response_is_lost(runtime):
    lease = sandbox._SandboxLease(runtime.settings, "task")
    lease.manager = runtime.manager
    request = httpx.Request("POST", "https://sandbox.test/v1/sandboxes")
    runtime.transport.handle_request.side_effect = httpx.ReadTimeout("create response lost")
    with pytest.raises(httpx.ReadTimeout):
        lease.handle_request(request)
    with pytest.raises(RuntimeError, match="repeat an ambiguous sandbox create"):
        lease.handle_request(request)
    assert runtime.transport.handle_request.call_count == 1
    lease.close()


def test_sdk_failure_after_create_response_still_deletes_captured_id(runtime):
    def create_then_fail(*args, **kwargs):
        runtime.create(*args, **kwargs)
        raise TimeoutError("SDK endpoint publication timed out")

    runtime.sdk_create.side_effect = create_then_fail
    with pytest.raises(TimeoutError, match="endpoint publication"):
        sandbox.create_task_sandbox(Path("/tasks/task"))
    runtime.manager.kill_sandbox.assert_called_once_with("sb-123")
    runtime.manager.get_sandbox_info.assert_called_once_with("sb-123")
    runtime.health.get.assert_not_called()
    report = _report(runtime)
    assert report["sandbox_id"] == "sb-123"
    assert "SDK endpoint publication timed out" in report["failure"]
    assert report["cleanup"] == {"ids": ["sb-123"], "deletion_confirmed": True}


def test_lost_create_response_reconciles_only_this_attempts_metadata(runtime):
    runtime.transport.handle_request.side_effect = httpx.ReadTimeout("create response lost")

    def list_sandboxes(filters):
        return SimpleNamespace(
            sandbox_infos=[
                SimpleNamespace(id="sb-mine", metadata=dict(filters.metadata)),
                SimpleNamespace(id="sb-neighbor", metadata={"openenv-create-id": "another-attempt"}),
            ]
        )

    runtime.manager.list_sandbox_infos.side_effect = list_sandboxes
    with pytest.raises(httpx.ReadTimeout, match="create response lost"):
        sandbox.create_task_sandbox(Path("/tasks/task"))
    assert runtime.transport.handle_request.call_count == 1
    runtime.manager.kill_sandbox.assert_called_once_with("sb-mine")
    assert _report(runtime)["cleanup"] == {"ids": ["sb-mine"], "deletion_confirmed": True}


def test_failed_evidence_write_after_create_does_not_bypass_deletion(runtime, monkeypatch):
    original_write = Path.write_text

    def disk_full_after_create(path, *args, **kwargs):
        if path.name == "lifecycle.tmp" and runtime.transport.handle_request.call_count:
            raise OSError("evidence disk full")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", disk_full_after_create)
    with pytest.raises(OSError, match="evidence disk full"):
        sandbox.create_task_sandbox(Path("/tasks/task"))
    runtime.manager.kill_sandbox.assert_called_once_with("sb-123")
    runtime.transport.close.assert_called_once()


def test_nonzero_bootstrap_never_reports_ready_and_keeps_redacted_output(runtime):
    calls = 0

    def run(command, *, opts, handlers):
        nonlocal calls
        calls += 1
        if calls == 2:
            handlers.on_stderr(SimpleNamespace(text="bootstrap failed: private-test-key"))
            return SimpleNamespace(exit_code=27, error=None)
        return SimpleNamespace(exit_code=0, error=None)

    runtime.instance.commands.run.side_effect = run
    with pytest.raises(RuntimeError, match="bootstrap-00 failed"):
        sandbox.create_task_sandbox(Path("/tasks/task"))
    runtime.instance.get_endpoint.assert_not_called()
    runtime.health.get.assert_not_called()
    runtime.manager.kill_sandbox.assert_called_once_with("sb-123")
    report = _report(runtime)
    assert not report.get("ready")
    assert report["commands"][-1]["exit_code"] == 27
    (stderr,) = runtime.log_root.glob("*/bootstrap-00.stderr")
    assert stderr.read_text() == "bootstrap failed: [REDACTED]\n"


def test_create_waits_for_server_and_cleanup_confirms_deletion(runtime):
    runtime.health.get.side_effect = [httpx.ConnectError("server is starting"), httpx.Response(200)]
    runtime.manager.get_sandbox_info.side_effect = [SimpleNamespace(id="sb-123"), _ApiError(status_code=404)]
    close, url = sandbox.create_task_sandbox(Path("/tasks/task"))
    assert runtime.sdk_create.call_args.args == ("task:fixed",)
    assert runtime.sdk_create.call_args.kwargs["resource"] == {
        "cpu": "1",
        "memory": "2048Mi",
        "ephemeral-storage": "10240Mi",
    }
    assert "serve-task" in runtime.instance.commands.run.call_args.args[0]
    runtime.instance.get_endpoint.assert_called_once_with(8000)
    assert url == "https://sandbox.test/v1/sandboxes/sb-123/proxy/8000"
    assert runtime.health.get.call_count == 2
    assert _report(runtime)["ready"]
    runtime.manager.kill_sandbox.assert_not_called()
    close()
    close()
    runtime.manager.kill_sandbox.assert_called_once_with("sb-123")
    assert runtime.manager.get_sandbox_info.call_count == 2
    runtime.instance.close.assert_called_once()
    runtime.manager.close.assert_called_once()
    runtime.transport.close.assert_called_once()
    assert _report(runtime)["cleanup"] == {"ids": ["sb-123"], "deletion_confirmed": True}


@pytest.mark.parametrize("delete_fails", [False, True])
def test_cleanup_does_not_claim_success_when_deletion_fails_or_remains_unconfirmed(runtime, delete_fails):
    close, _ = sandbox.create_task_sandbox(Path("/tasks/task"))
    if delete_fails:
        runtime.manager.kill_sandbox.side_effect = _ApiError("delete denied: private-test-key", status_code=500)
        error_type, message = _ApiError, "delete denied"
    else:
        runtime.manager.get_sandbox_info.side_effect = None
        runtime.manager.get_sandbox_info.return_value = SimpleNamespace(id="sb-123")
        error_type, message = RuntimeError, "deletion was not confirmed"
    with pytest.raises(error_type, match=message):
        close()
    report = _report(runtime)
    assert message in report["cleanup_error"]
    assert "private-test-key" not in report["cleanup_error"]
    assert "cleanup" not in report
    runtime.transport.close.assert_called_once()
