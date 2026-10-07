"""CPU-only contract tests for the authenticated OpenEnv transport.

No real keys, network connections, sandbox creation, or GPU dependencies.
The shared backend's agent/scoring tests already cover the unchanged protocol.
"""

import asyncio
import logging
import ssl
from pathlib import Path
from types import SimpleNamespace

import openenv_opensandbox_agent_function as backend_module
import openenv_sandbox_common as common
import pytest
from websockets.asyncio import client as websocket_client
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response


class FakeEnv:
    def __init__(self, base_url, message_timeout_s=60):
        self._ws_url = base_url.replace("https://", "wss://", 1).rstrip("/") + "/ws"
        self._ws = None
        self._ws_loop = None
        self._connect_timeout = 12
        self._max_message_size = 2048
        self._websocket_ping_interval_s = 20
        self._websocket_ping_timeout_s = None
        self.closed = False

    def _start_provider_if_needed(self):
        pass

    async def close(self):
        self.closed = True
        self._ws = None
        self._ws_loop = None

    async def __aenter__(self):
        return await self._connect_async()

    async def __aexit__(self, *args):
        await self.close()


@pytest.fixture
def transport(monkeypatch):
    settings = backend_module.tb2_sandbox_opensandbox.Settings(
        endpoint="https://sandbox.example", api_key="test-only-key", tls=ssl.create_default_context()
    )
    monkeypatch.setattr(backend_module.tb2_sandbox_opensandbox, "connection_settings", lambda: settings)
    state = SimpleNamespace(settings=settings, calls=[], error=None)

    class FakeConnect:
        def __init__(self, url, **kwargs):
            self.url = url
            self.kwargs = kwargs
            state.calls.append(self)

        def __await__(self):
            async def connection():
                if state.error is not None:
                    raise state.error
                return object()

            return connection().__await__()

    monkeypatch.setattr(websocket_client, "connect", FakeConnect)
    return state


def test_authentication_tls_and_openenv_connection_options(transport):
    env_cls = backend_module._authenticated_env_class(FakeEnv)
    env = env_cls(base_url="https://sandbox.example/v1/sandboxes/test/proxy/8000")

    async def connect_twice():
        assert await env._connect_async() is env
        first_connection = env._ws
        assert await env._connect_async() is env
        assert env._ws is first_connection

    asyncio.run(connect_twice())
    assert len(transport.calls) == 1
    connection = transport.calls[0]
    assert connection.url == "wss://sandbox.example/v1/sandboxes/test/proxy/8000/ws"
    assert connection.kwargs["additional_headers"] == {"OPEN-SANDBOX-API-KEY": "test-only-key"}
    assert connection.kwargs["ssl"] is transport.settings.tls
    assert connection.kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert connection.kwargs["ssl"].check_hostname
    assert connection.kwargs["proxy"] is None
    assert connection.kwargs["open_timeout"] == 12
    assert connection.kwargs["max_size"] == 2048
    assert connection.kwargs["ping_interval"] == 20
    assert connection.kwargs["ping_timeout"] is None
    assert not connection.kwargs["logger"].isEnabledFor(logging.DEBUG)


def test_redirects_are_never_followed_with_credentials(transport):
    env = backend_module._authenticated_env_class(FakeEnv)(base_url="https://sandbox.example")
    asyncio.run(env._connect_async())
    redirect = InvalidStatus(Response(302, "Found", Headers({"Location": "wss://other.example/ws"})))
    assert transport.calls[0].process_redirect(redirect) is redirect


@pytest.mark.parametrize("url", ["https://other.example", "https://sandbox.example:444", "http://sandbox.example"])
def test_rejects_another_origin_before_sending_credentials(transport, url):
    env = backend_module._authenticated_env_class(FakeEnv)(base_url=url)
    with pytest.raises(ConnectionError, match="ValueError"):
        asyncio.run(env._connect_async())
    assert not transport.calls
    assert env.closed


def test_connection_failure_does_not_render_api_key(transport):
    transport.error = RuntimeError("request header OPEN-SANDBOX-API-KEY: test-only-key")
    env = backend_module._authenticated_env_class(FakeEnv)(base_url="https://sandbox.example")
    with pytest.raises(ConnectionError) as error:
        asyncio.run(env._connect_async())
    assert "test-only-key" not in str(error.value)
    assert error.value.__suppress_context__
    assert env.closed


def test_reconnects_when_the_original_event_loop_has_closed(transport):
    env = backend_module._authenticated_env_class(FakeEnv)(base_url="https://sandbox.example")
    asyncio.run(env._connect_async())
    first_connection = env._ws
    asyncio.run(env._connect_async())
    assert len(transport.calls) == 2
    assert env._ws is not first_connection


@pytest.mark.parametrize("connection_fails", [False, True])
def test_shared_episode_lifecycle_always_deletes_sandbox(monkeypatch, transport, connection_fails):
    monkeypatch.setattr(common, "_agent_function", lambda: SimpleNamespace(MESSAGE_TIMEOUT_S=1200))
    closed = []
    backend = backend_module.OpenSandboxBackend(
        provider="OpenSandbox",
        logger=logging.getLogger("test"),
        is_throttle=lambda exc: False,
        start_sandbox=lambda task_id, tasks_dir: (lambda: closed.append(task_id), "https://sandbox.example"),
    )
    if connection_fails:
        transport.error = RuntimeError("upgrade failed")

    async def episode():
        async with backend.episode_env(FakeEnv, {"task_id": "regex-log"}) as env:
            assert isinstance(env, FakeEnv)
            raise LookupError("episode failed")

    expected_error = ConnectionError if connection_fails else LookupError
    with pytest.raises(expected_error):
        asyncio.run(episode())
    assert closed == ["regex-log"]


def test_start_hook_passes_task_and_returns_its_cleanup(monkeypatch):
    seen = []

    def close_fn():
        seen.append("closed")

    def create(task_dir):
        seen.append(task_dir)
        return close_fn, "https://sandbox.example"

    monkeypatch.setattr(backend_module.tb2_sandbox_opensandbox, "create_task_sandbox", create)
    close, url = backend_module._start_sandbox("regex-log", "/tasks")
    close()
    assert seen == [Path("/tasks/regex-log"), "closed"]
    assert close is close_fn
    assert url == "https://sandbox.example"
