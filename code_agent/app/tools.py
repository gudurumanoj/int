from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.models import ToolResult


MAX_TOOL_OUTPUT_BYTES = 64 * 1024
MAX_READ_BYTES = 256 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _bounded(data: bytes) -> str:
    suffix = b"\n...[output truncated]"
    if len(data) > MAX_TOOL_OUTPUT_BYTES:
        data = data[: MAX_TOOL_OUTPUT_BYTES - len(suffix)] + suffix
    return data.decode("utf-8", errors="replace")


class ToolValidationError(ValueError):
    pass


class SandboxRunner(Protocol):
    async def run(
        self,
        run_id: str,
        workspace: Path,
        command: str,
        timeout_seconds: float,
    ) -> tuple[str, str, int | None, str]: ...

    async def cancel(self, run_id: str) -> None: ...


class WorkspaceFiles:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, relative_path: str) -> Path:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ToolValidationError("Workspace paths must be relative")
        candidate = (self.root / requested).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolValidationError("Path escapes the run workspace") from exc
        return candidate

    def read_file(self, relative_path: str) -> str:
        path = self._path(relative_path)
        if not path.is_file():
            raise ToolValidationError("File does not exist")
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise ToolValidationError("File is too large to read")
        return data.decode("utf-8")

    def write_file(self, relative_path: str, content: str) -> int:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Re-resolve after mkdir so newly exposed links cannot redirect the write.
        path = self._path(relative_path)
        encoded = content.encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return len(encoded)


@dataclass(slots=True)
class _ActiveProcess:
    process: asyncio.subprocess.Process
    container_name: str


class DockerRunner:
    """Runs a command in a networkless, resource-limited Docker container."""

    def __init__(self, image: str):
        self.image = image
        self._active: dict[str, _ActiveProcess] = {}
        self._lock = asyncio.Lock()

    async def run(
        self,
        run_id: str,
        workspace: Path,
        command: str,
        timeout_seconds: float,
    ) -> tuple[str, str, int | None, str]:
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        name_hash = hashlib.sha256(run_id.encode()).hexdigest()[:16]
        container_name = f"code-agent-{name_hash}"
        mount = f"type=bind,source={workspace},target=/workspace"
        arguments = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--cpus",
            "0.5",
            "--memory",
            "256m",
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            self.image,
            "sh",
            "-lc",
            command,
        ]
        process_options: dict[str, Any] = {}
        if os.name == "nt":
            process_options["creationflags"] = (
                subprocess_creation_new_process_group()
            )
        else:
            process_options["start_new_session"] = True
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_options,
        )
        async with self._lock:
            self._active[run_id] = _ActiveProcess(process, container_name)
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
                status = "succeeded" if process.returncode == 0 else "failed"
                return (
                    _bounded(stdout),
                    _bounded(stderr),
                    process.returncode,
                    status,
                )
            except TimeoutError:
                await self.cancel(run_id)
                return "", "Command timed out", None, "timed_out"
            except asyncio.CancelledError:
                await self.cancel(run_id)
                raise
        finally:
            async with self._lock:
                current = self._active.get(run_id)
                if current is not None and current.process is process:
                    self._active.pop(run_id, None)

    async def cancel(self, run_id: str) -> None:
        async with self._lock:
            active = self._active.get(run_id)
        if active is None or active.process.returncode is not None:
            return
        await self._remove_container(active.container_name)
        await self._terminate_process_tree(active.process)

    @staticmethod
    async def _remove_container(container_name: str) -> None:
        try:
            remover = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(remover.wait(), timeout=5)
        except (FileNotFoundError, TimeoutError):
            pass

    @staticmethod
    async def _terminate_process_tree(
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=5)
            except (FileNotFoundError, TimeoutError):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                await asyncio.wait_for(process.wait(), timeout=2)
            except (ProcessLookupError, TimeoutError):
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGKILL)
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()


def subprocess_creation_new_process_group() -> int:
    # Kept behind a function because this constant is Windows-only.
    import subprocess

    return subprocess.CREATE_NEW_PROCESS_GROUP


class ToolRegistry:
    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 file inside the run workspace.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Atomically write a UTF-8 workspace file.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": (
                    "Run a shell command in the networkless Docker workspace."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def __init__(
        self,
        workspace_root: Path,
        runner: SandboxRunner,
        shell_timeout_seconds: float = 60,
    ):
        self.workspace_root = workspace_root
        self.runner = runner
        self.shell_timeout_seconds = shell_timeout_seconds

    @staticmethod
    def validate(name: str, arguments: dict[str, Any]) -> None:
        expected: dict[str, dict[str, type]] = {
            "read_file": {"path": str},
            "write_file": {"path": str, "content": str},
            "shell": {"command": str},
        }
        schema = expected.get(name)
        if schema is None:
            raise ToolValidationError(f"Unknown tool: {name}")
        if set(arguments) != set(schema):
            raise ToolValidationError(f"Invalid arguments for {name}")
        for key, expected_type in schema.items():
            if not isinstance(arguments[key], expected_type):
                raise ToolValidationError(f"{key} must be a string")
            if arguments[key] == "":
                raise ToolValidationError(f"{key} must not be empty")

    @staticmethod
    def requires_approval(name: str) -> bool:
        return name == "shell"

    async def execute(
        self,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        started = _now()
        workspace = self.workspace_root / run_id
        files = WorkspaceFiles(workspace)
        try:
            if name == "read_file":
                stdout = files.read_file(arguments["path"])
                details = {"path": arguments["path"], "operation": "read"}
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    status="succeeded",
                    stdout=stdout,
                    side_effect_summary=details,
                    started_at=started,
                    finished_at=_now(),
                )
            if name == "write_file":
                count = files.write_file(
                    arguments["path"], arguments["content"]
                )
                details = {
                    "path": arguments["path"],
                    "operation": "atomic_write",
                    "bytes_written": count,
                }
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    status="succeeded",
                    stdout=f"Wrote {count} bytes",
                    side_effect_summary=details,
                    started_at=started,
                    finished_at=_now(),
                )
            stdout, stderr, exit_code, status = await self.runner.run(
                run_id,
                workspace,
                arguments["command"],
                self.shell_timeout_seconds,
            )
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                status=status,  # type: ignore[arg-type]
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                side_effect_summary={"sandbox": "docker", "network": "none"},
                started_at=started,
                finished_at=_now(),
            )
        except (OSError, UnicodeError, ToolValidationError) as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                status="failed",
                stderr=str(exc),
                error_code="tool_error",
                started_at=started,
                finished_at=_now(),
            )
