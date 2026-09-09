from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

from .util import canonical_json, new_id, sha256_bytes


class Storage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.datasets_dir = self.root / "datasets"
        self.runs_dir = self.root / "runs"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_csv_filename(filename: str) -> str:
        if Path(filename).name != filename or not filename.lower().endswith(".csv"):
            raise ValueError("filename must be a plain .csv filename")
        if filename in {".csv", "..csv"}:
            raise ValueError("invalid CSV filename")
        return filename

    def store_dataset(self, filename: str, content: str) -> dict[str, Any]:
        filename = self.validate_csv_filename(filename)
        raw = content.encode("utf-8")
        digest = sha256_bytes(raw)
        path = self.datasets_dir / digest[:2] / f"{digest}.csv"
        if path.exists():
            if sha256_bytes(path.read_bytes()) != digest:
                raise RuntimeError("immutable dataset path has unexpected content")
        else:
            self._atomic_write(path, raw)
            try:
                path.chmod(0o444)
            except OSError:
                pass
        return {
            "filename": filename,
            "sha256": digest,
            "path": str(path),
            "size_bytes": len(raw),
        }

    def profile_csv(self, path: str, max_rows: int = 10_000) -> dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV must have a header")
        columns: dict[str, list[str]] = {name: [] for name in reader.fieldnames}
        row_count = 0
        for row in reader:
            row_count += 1
            if row_count <= max_rows:
                for name in reader.fieldnames:
                    columns[name].append((row.get(name) or "").strip())

        profile_columns: dict[str, dict[str, Any]] = {}
        for name, values in columns.items():
            non_null = [value for value in values if value != ""]
            numeric: list[float] = []
            for value in non_null:
                try:
                    numeric.append(float(value))
                except ValueError:
                    numeric = []
                    break
            profile_columns[name] = {
                "dtype": "number" if numeric and len(numeric) == len(non_null) else "string",
                "nulls_in_sample": len(values) - len(non_null),
                "nunique_in_sample": len(set(non_null)),
                "mean": round(statistics.fmean(numeric), 6) if numeric else None,
                "std": (
                    round(statistics.stdev(numeric), 6)
                    if len(numeric) > 1
                    else None
                ),
            }
        return {
            "row_count": row_count,
            "scanned_rows": min(row_count, max_rows),
            "sampling_policy": f"first_{max_rows}_rows",
            "columns": profile_columns,
        }

    def save_code(
        self, run_id: str, plan_version: int, step_id: str, code: str
    ) -> dict[str, Any]:
        path = (
            self.runs_dir
            / run_id
            / f"plan-{plan_version}"
            / self._safe_name(step_id)
            / "proposal.py"
        )
        return self._write_ref(path, code.encode("utf-8"), "code", "code")

    def save_step_artifacts(
        self,
        run_id: str,
        plan_version: int,
        step_id: str,
        outputs: dict[str, str],
        code_ref: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        directory = (
            self.runs_dir
            / run_id
            / f"plan-{plan_version}"
            / self._safe_name(step_id)
        )
        artifacts = [code_ref]
        for name, content in sorted(outputs.items()):
            safe_name = self._safe_name(name)
            path = directory / "artifacts" / safe_name
            artifacts.append(
                self._write_ref(path, content.encode("utf-8"), "step_output")
            )
        manifest_data = {
            "run_id": run_id,
            "plan_version": plan_version,
            "step_id": step_id,
            "artifacts": [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                }
                for item in artifacts
            ],
        }
        checkpoint_path = directory / "checkpoint.json"
        checkpoint_raw = canonical_json(manifest_data).encode("utf-8")
        self._atomic_write(checkpoint_path, checkpoint_raw)
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": sha256_bytes(checkpoint_raw),
        }
        return artifacts, checkpoint

    def save_final(
        self, run_id: str, report_markdown: str, manifest_data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        directory = self.runs_dir / run_id / "final"
        report = self._write_ref(
            directory / "report.md",
            report_markdown.encode("utf-8"),
            "report",
            "report",
        )
        manifest = self._write_ref(
            directory / "reproducibility.json",
            (json.dumps(manifest_data, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
            "reproducibility_manifest",
            "manifest",
        )
        return report, manifest

    def runner_directories(
        self, run_id: str, plan_version: int, step_id: str
    ) -> tuple[Path, Path]:
        root = (
            self.runs_dir
            / run_id
            / f"plan-{plan_version}"
            / self._safe_name(step_id)
            / "sandbox"
        )
        workspace = root / "workspace"
        output = root / "output"
        workspace.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        return workspace, output

    def _write_ref(
        self,
        path: Path,
        raw: bytes,
        kind: str,
        id_prefix: str = "art",
    ) -> dict[str, Any]:
        self._atomic_write(path, raw)
        return {
            "id": new_id(id_prefix),
            "kind": kind,
            "path": str(path),
            "sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        }

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
        if not cleaned or cleaned in {".", ".."}:
            raise ValueError("invalid artifact name")
        return cleaned[:160]

    @staticmethod
    def _atomic_write(path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
