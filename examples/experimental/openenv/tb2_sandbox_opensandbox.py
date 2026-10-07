"""OpenSandbox materialization of the existing TB2 OpenEnv server recipe.

This initial backend installs the server layer at sandbox startup. It is for
functional validation; training should use measured, cached startup rather
than assuming Daytona's image-build cache exists on another provider.

Required: OPEN_SANDBOX_API_URL and OPEN_SANDBOX_API_KEY[_FILE]. Optional:
OPEN_SANDBOX_CA_FILE supplies a private CA without disabling TLS validation;
OPENENV_OPENSANDBOX_LOG_DIR retains lifecycle and bootstrap evidence.
"""

import json
import logging
import os
import shlex
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import tb2_sandbox_recipe as recipe

from miles.rollout.agentic.credentials import resolve_provider_api_key

logger = logging.getLogger(__name__)
# The authenticated OpenEnv client also uses this version's connection seam.
OPENENV_REVISION = "38b2a31354d1d4d627894412b9d469e92f4a2c61"


@dataclass(frozen=True)
class Settings:
    endpoint: str
    api_key: str = field(repr=False)
    tls: ssl.SSLContext = field(repr=False)

    def assert_origin(self, url: str) -> None:
        expected = urlsplit(self.endpoint)
        actual = urlsplit(url)
        if (
            actual.scheme not in ("https", "wss")
            or actual.hostname != expected.hostname
            or (actual.port or 443) != (expected.port or 443)
            or actual.username is not None
            or actual.password is not None
            or actual.fragment
        ):
            raise ValueError("OpenSandbox endpoint escaped its configured HTTPS origin")


def connection_settings() -> Settings:
    endpoint = os.environ.get("OPEN_SANDBOX_API_URL", "").rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query:
        raise ValueError("OPEN_SANDBOX_API_URL must be an explicit HTTPS service origin")
    ca_file = os.environ.get("OPEN_SANDBOX_CA_FILE") or None
    settings = Settings(
        endpoint=endpoint,
        api_key=resolve_provider_api_key(
            "OPEN_SANDBOX_API_KEY", "OPEN_SANDBOX_API_KEY_FILE", "~/.config/opensandbox/api_key"
        ),
        tls=ssl.create_default_context(cafile=ca_file),
    )
    settings.assert_origin(endpoint)
    return settings


class _SandboxLease(httpx.BaseTransport):
    """Own one create attempt, its HTTP transport, evidence, and deletion."""

    def __init__(self, settings: Settings, task_id: str):
        self.settings = settings
        self.inner = httpx.HTTPTransport(verify=settings.tls, retries=0)
        self.sandbox = None
        self.manager = None
        self.identifier = None
        self.create_attempted = False
        self.closed = False
        self.metadata = {"openenv-create-id": uuid4().hex}
        log_root = os.environ.get("OPENENV_OPENSANDBOX_LOG_DIR")
        self.log_dir = Path(log_root) / self.metadata["openenv-create-id"] if log_root else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=False)
        self.report = {"task_id": task_id, "metadata": self.metadata, "http": [], "commands": []}

    def save(self, *, best_effort: bool = False) -> None:
        if self.log_dir is not None:
            data = json.dumps(self.report, indent=2).replace(self.settings.api_key, "[REDACTED]")
            path = self.log_dir / "lifecycle.json"
            try:
                path.with_suffix(".tmp").write_text(data + "\n")
                path.with_suffix(".tmp").replace(path)
            except OSError:
                if not best_effort:
                    raise
                # Never lose the SDK's create response or skip deletion just
                # because its diagnostic output filesystem became unavailable.
                logger.warning("Could not persist OpenSandbox lifecycle evidence")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.settings.assert_origin(str(request.url))
        creating = request.method == "POST" and request.url.path == "/v1/sandboxes"
        if creating:
            if self.create_attempted:
                raise RuntimeError("Refusing to repeat an ambiguous sandbox create")
            self.create_attempted = True
        response = self.inner.handle_request(request)
        self.report["http"].append(
            {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if creating and 200 <= response.status_code < 300:
            response.read()
            self.identifier = response.json()["id"]
            self.report["sandbox_id"] = self.identifier
        self.save(best_effort=True)
        return response

    def command(self, name: str, command: str, timeout_s: float) -> None:
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.execd_sync import ExecutionHandlersSync

        started = time.monotonic()
        record = {"name": name, "started_at": datetime.now(timezone.utc).isoformat()}
        self.report["commands"].append(record)
        self.save()

        def output(stream, message):
            if self.log_dir is not None:
                with (self.log_dir / f"{name}.{stream}").open("a") as file:
                    file.write(message.text.replace(self.settings.api_key, "[REDACTED]") + "\n")

        result = self.sandbox.commands.run(
            command,
            opts=RunCommandOpts(timeout=timedelta(seconds=timeout_s), working_directory="/", uid=0, gid=0),
            handlers=ExecutionHandlersSync(
                on_stdout=lambda message: output("stdout", message),
                on_stderr=lambda message: output("stderr", message),
            ),
        )
        record.update(
            seconds=round(time.monotonic() - started, 3),
            exit_code=result.exit_code,
            error=result.error.model_dump() if result.error else None,
        )
        self.save()
        if result.exit_code != 0 or result.error is not None:
            raise RuntimeError(f"OpenSandbox {name} failed; inspect bootstrap logs")

    def close(self) -> None:
        # SDK clients do not own this supplied transport. This callback owns
        # deletion too; Sandbox.close() alone only closes client-side handles.
        if self.closed:
            return
        from opensandbox.exceptions import SandboxApiException
        from opensandbox.models.sandboxes import SandboxFilter

        try:
            if self.sandbox is not None and self.log_dir is not None:
                if any(command["name"] == "start-server" for command in self.report["commands"]):
                    try:
                        data = self.sandbox.files.read_bytes("/tmp/openenv-server.log")
                        (self.log_dir / "openenv-server.log").write_bytes(
                            data.replace(self.settings.api_key.encode(), b"[REDACTED]")
                        )
                    except Exception as exc:
                        self.report["server_log_capture_error"] = type(exc).__name__
            ids = [self.identifier] if self.identifier else []
            if self.create_attempted and not ids:
                page = self.manager.list_sandbox_infos(SandboxFilter(metadata=self.metadata, page_size=10))
                ids = [
                    item.id
                    for item in page.sandbox_infos
                    if item.metadata and item.metadata.get("openenv-create-id") == self.metadata["openenv-create-id"]
                ]
            for identifier in ids:
                try:
                    self.manager.kill_sandbox(identifier)
                except SandboxApiException as exc:
                    if exc.status_code != 404:
                        raise
                for _ in range(12):
                    try:
                        self.manager.get_sandbox_info(identifier)
                    except SandboxApiException as exc:
                        if exc.status_code == 404:
                            break
                        raise
                    time.sleep(2)
                else:
                    raise RuntimeError("OpenSandbox deletion was not confirmed")
            self.report["cleanup"] = {"ids": ids, "deletion_confirmed": bool(ids)}
            if self.create_attempted and not ids:
                self.report["cleanup"]["note"] = "No matching create found; TTL remains the orphan backstop"
        except Exception as exc:
            self.report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.save(best_effort=True)
            if self.sandbox is not None:
                self.sandbox.close()
            if self.manager is not None:
                self.manager.close()
            self.inner.close()
            self.closed = True


def create_task_sandbox(task_dir: Path, *, docker_image: str | None = None):
    """Return (delete_callback, authenticated-server-proxy URL) for one task.

    Per-attempt creation is never retried. TTL bounds leaks after node/process
    loss; the shared backend additionally reaps a create cancelled mid-bootstrap.
    """
    # Only a selected provider needs its SDK installed.
    from opensandbox.config import ConnectionConfigSync
    from opensandbox.sync import SandboxManagerSync, SandboxSync

    task_dir = Path(task_dir)
    commands = recipe.server_layer_commands(task_dir)  # fail before spending quota
    settings = connection_settings()
    lease = _SandboxLease(settings, task_dir.name)
    config = ConnectionConfigSync(
        domain=settings.endpoint,
        protocol="https",
        api_key=settings.api_key,
        use_server_proxy=True,
        transport=lease,
        disable_metrics=True,
        request_timeout=timedelta(seconds=60),
    )
    lease.manager = SandboxManagerSync.create(connection_config=config)
    cpus, memory, storage = recipe.task_env_resources(task_dir)
    resource = {"cpu": str(cpus), "memory": f"{memory}Mi", "ephemeral-storage": f"{storage}Mi"}
    image = recipe.resolve_docker_image(task_dir, docker_image)
    lease.metadata.update(recipe.sandbox_labels(task_dir))
    lease.report.update(image=image, resource=resource, ttl_seconds=3600, openenv_revision=OPENENV_REVISION)
    lease.save()
    try:
        lease.sandbox = SandboxSync.create(
            image,
            connection_config=config,
            resource=resource,
            metadata=lease.metadata,
            timeout=timedelta(seconds=3600),
            ready_timeout=timedelta(seconds=240),
            entrypoint=["tail", "-f", "/dev/null"],
        )
        deadline = time.monotonic() + 1800
        # Keep the runtime used by the server and the authenticated client in
        # agreement. No official task files or scoring code are rewritten.
        constraint = f"openenv @ https://github.com/huggingface/OpenEnv/archive/{OPENENV_REVISION}.tar.gz\n"
        lease.command("constraints", f"printf %s {shlex.quote(constraint)} > /tmp/openenv-constraints.txt", 30)
        for i, command in enumerate(commands):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OpenSandbox server bootstrap exceeded 1800 seconds")
            lease.command(
                f"bootstrap-{i:02}",
                "UV_CONSTRAINT=/tmp/openenv-constraints.txt bash -euo pipefail -c " + shlex.quote(command),
                min(900, remaining),
            )
        command = recipe.server_cmd(default_task_id=task_dir.name)
        lease.command(
            "start-server", f"nohup bash -c {shlex.quote(command)} > /tmp/openenv-server.log 2>&1 < /dev/null &", 30
        )
        endpoint = lease.sandbox.get_endpoint(8000)
        if endpoint.headers:
            raise ValueError("Additional OpenSandbox endpoint headers are not supported by this backend")
        url = endpoint.endpoint if "://" in endpoint.endpoint else f"https://{endpoint.endpoint}"
        settings.assert_origin(url)
        # Health checks need the same authentication and CA as the websocket.
        with httpx.Client(
            verify=settings.tls, headers={"OPEN-SANDBOX-API-KEY": settings.api_key}, follow_redirects=False, timeout=10
        ) as client:
            for _ in range(60):
                try:
                    response = client.get(f"{url}/health")
                except httpx.TransportError:
                    time.sleep(2)
                    continue
                if response.status_code == 200:
                    lease.report["ready"] = True
                    lease.save()
                    return lease.close, url
                if response.status_code in (301, 302, 307, 308, 401, 403):
                    raise RuntimeError(f"OpenEnv health endpoint rejected access ({response.status_code})")
                time.sleep(2)
        raise TimeoutError("OpenEnv server did not become healthy")
    except BaseException as exc:
        lease.report["failure"] = f"{type(exc).__name__}: {exc}"
        lease.save(best_effort=True)
        try:
            lease.close()
        except Exception:
            logger.exception("Failed to reclaim this attempt's sandbox; its TTL remains armed")
        raise
