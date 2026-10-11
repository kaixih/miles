"""OpenSandbox connection wiring; shared loop behavior is tested in sandbox_common."""

import asyncio
import logging
import ssl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("connection_fails", [False, True])
def test_episode_connects_with_authentication_and_releases_sandbox(monkeypatch, transport, connection_fails):
    monkeypatch.setattr(common, "_agent_function", lambda: SimpleNamespace(MESSAGE_TIMEOUT_S=1200))
    close = Mock()
    create = Mock(return_value=(close, "https://sandbox.example/v1/sandboxes/test/proxy/8000"))
    monkeypatch.setattr(backend_module.tb2_sandbox_opensandbox, "create_task_sandbox", create)
    monkeypatch.setenv("OPENENV_TB2_TASKS_DIR", "/tasks")
    if connection_fails:
        transport.error = RuntimeError("request header OPEN-SANDBOX-API-KEY: test-only-key")

    async def episode():
        async with backend_module.BACKEND.episode_env(FakeEnv, {"task_id": "regex-log"}) as env:
            assert env._ws is not None

    if connection_fails:
        with pytest.raises(ConnectionError) as error:
            asyncio.run(episode())
        assert "test-only-key" not in str(error.value)
        assert error.value.__suppress_context__
    else:
        asyncio.run(episode())
    create.assert_called_once_with(Path("/tasks/regex-log"))
    close.assert_called_once_with()
    assert len(transport.calls) == 1
    connection = transport.calls[0]
    assert connection.url == "wss://sandbox.example/v1/sandboxes/test/proxy/8000/ws"
    assert connection.kwargs["additional_headers"] == {"OPEN-SANDBOX-API-KEY": "test-only-key"}
    assert connection.kwargs["ssl"] is transport.settings.tls
    assert connection.kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert connection.kwargs["ssl"].check_hostname
    assert connection.kwargs["proxy"] is None
    assert not connection.kwargs["logger"].isEnabledFor(logging.DEBUG)


def test_redirects_are_never_followed_with_credentials(transport):
    env = backend_module._authenticated_env_class(FakeEnv)(base_url="https://sandbox.example")
    asyncio.run(env._connect_async())
    redirect = InvalidStatus(Response(302, "Found", Headers({"Location": "wss://other.example/ws"})))
    assert transport.calls[0].process_redirect(redirect) is redirect


def test_rejects_another_origin_before_sending_credentials(transport):
    env = backend_module._authenticated_env_class(FakeEnv)(base_url="https://other.example")
    with pytest.raises(ConnectionError, match="ValueError"):
        asyncio.run(env._connect_async())
    assert not transport.calls
    assert env.closed
