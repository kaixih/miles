"""Miles TB2 episodes on an authenticated OpenSandbox server proxy.

The task image and OpenEnv server are prepared by ``tb2_sandbox_opensandbox``.
The shared Miles loop still owns reset, commands, canonical evaluation, and
training records. This module adds the proxy's authentication and TLS to the
OpenEnv WebSocket, plus the provider's ordinary sandbox lifecycle hooks.

Requires OpenEnv revision 38b2a31354d1d4d627894412b9d469e92f4a2c61. Its client
has no public headers / SSL options, so the small connection override below
uses that revision's private connection fields. Keep this pin until OpenEnv
exposes those options. No OpenEnv protocol or scoring code is replaced.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import openenv_sandbox_common as common
import tb2_sandbox_opensandbox

if TYPE_CHECKING:
    from openenv.core.env_client import EnvClient

logger = logging.getLogger(__name__)
# WebSocket DEBUG traces include handshake headers. Never send the API key to
# a process-wide DEBUG logger, even when rollout diagnostics enable it.
_websocket_logger = logging.Logger("openenv.opensandbox.websocket", level=logging.WARNING)


def _authenticated_env_class(env_cls: type["EnvClient"]) -> type["EnvClient"]:
    # OpenEnv and websockets are optional until this provider runs an episode.
    from websockets.asyncio.client import connect

    settings = tb2_sandbox_opensandbox.connection_settings()

    class NoRedirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception:
            # websockets otherwise forwards additional_headers when following
            # redirects. The API key must only reach the configured origin.
            return exc

    class AuthenticatedEnv(env_cls):
        async def _connect_async(self):
            loop = asyncio.get_running_loop()
            if self._ws is not None:
                if self._ws_loop is loop:
                    return self
                self._ws = None
                self._ws_loop = None

            try:
                self._start_provider_if_needed()
                assert self._ws_url is not None
                settings.assert_origin(self._ws_url)
                self._ws = await NoRedirectConnect(
                    self._ws_url,
                    additional_headers={"OPEN-SANDBOX-API-KEY": settings.api_key},
                    ssl=settings.tls,
                    proxy=None,
                    open_timeout=self._connect_timeout,
                    max_size=self._max_message_size,
                    ping_interval=self._websocket_ping_interval_s,
                    ping_timeout=self._websocket_ping_timeout_s,
                    logger=_websocket_logger,
                )
                self._ws_loop = loop
            except Exception as exc:
                await self.close()
                # Third-party exceptions can retain request headers. Keep the
                # failure category without rendering a credential-bearing cause.
                raise ConnectionError(f"OpenSandbox WebSocket connection failed ({type(exc).__name__})") from None
            return self

    return AuthenticatedEnv


class OpenSandboxBackend(common.SandboxBackend):
    @asynccontextmanager
    async def episode_env(self, env_cls: type["EnvClient"], metadata: dict[str, object]):
        async with super().episode_env(_authenticated_env_class(env_cls), metadata) as env:
            yield env


def _start_sandbox(task_id: str, tasks_dir: str) -> tuple[Callable[[], None], str]:
    return tb2_sandbox_opensandbox.create_task_sandbox(Path(tasks_dir) / task_id)


def _is_throttle_error(exc: BaseException) -> bool:
    return common.throttle_text(exc)


BACKEND = OpenSandboxBackend(
    provider="OpenSandbox",
    start_sandbox=_start_sandbox,
    is_throttle=_is_throttle_error,
    logger=logger,
    **common.backend_knobs("OPENSANDBOX"),
)

run_episode = BACKEND.run_episode
run = BACKEND.run
