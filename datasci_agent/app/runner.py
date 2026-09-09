from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from .config import Settings
from .models import RunnerResult
from .storage import Storage


class StepRunner(Protocol):
    image: str

    async def run(
        self,
        run_id: str,
        plan_version: int,
        step_id: str,
        code: str,
        dataset_path: str,
    ) -> RunnerResult: ...

    async def cancel(self, run_id: str) -> None: ...


class DockerRunner:
    """Illustrative sandbox. Tests inject a fake and never invoke Docker."""

    def __init__(self, settings: Settings, storage: Storage):
        self.image = settings.docker_image
        self.cpus = settings.docker_cpus
        self.memory = settings.docker_memory
        self.timeout = settings.docker_timeout_seconds
        self.storage = storage
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def run(
        self,
        run_id: str,
        plan_version: int,
        step_id: str,
        code: str,
        dataset_path: str,
    ) -> RunnerResult:
        workspace, output = self.storage.runner_directories(
            run_id, plan_version, step_id
        )
        code_path = workspace / "step.py"
        code_path.write_text(code, encoding="utf-8")

        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            "128",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,src={Path(dataset_path).resolve()},dst=/input/data.csv,readonly",
            "--mount",
            f"type=bind,src={workspace.resolve()},dst=/workspace,readonly",
            "--mount",
            f"type=bind,src={output.resolve()},dst=/output",
            "--workdir",
            "/workspace",
            self.image,
            "python",
            "-I",
            "/workspace/step.py",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._processes[run_id] = process
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return RunnerResult(status="timed_out", error={"type": "timeout"})
        finally:
            self._processes.pop(run_id, None)

        stdout = stdout_bytes.decode("utf-8", errors="replace")[-20_000:]
        if process.returncode != 0:
            return RunnerResult(
                status="failed",
                stdout=stdout,
                error={"type": "nonzero_exit", "returncode": process.returncode},
            )
        try:
            artifacts = self._collect_outputs(output)
        except ValueError as exc:
            return RunnerResult(
                status="failed",
                stdout=stdout,
                error={"type": "invalid_output", "message": str(exc)},
            )
        return RunnerResult(status="succeeded", stdout=stdout, artifacts=artifacts)

    async def cancel(self, run_id: str) -> None:
        process = self._processes.get(run_id)
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()

    @staticmethod
    def _collect_outputs(output: Path) -> dict[str, str]:
        allowed = {".txt", ".json", ".csv", ".md"}
        result: dict[str, str] = {}
        total_bytes = 0
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed:
                raise ValueError(f"output type is not allowlisted: {path.name}")
            size = path.stat().st_size
            total_bytes += size
            if size > 5_000_000 or total_bytes > 20_000_000:
                raise ValueError("sandbox output limit exceeded")
            relative = path.relative_to(output).as_posix()
            result[relative] = path.read_text(encoding="utf-8")
        return result
