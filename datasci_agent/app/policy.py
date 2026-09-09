from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .models import AnalysisStep, WorkerProposal


MANDATORY_APPROVAL_TAGS = {"target_change", "row_drop", "package_install"}
NETWORK_MODULES = {"requests", "httpx", "socket", "urllib", "ftplib"}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    risk_tags: list[str]
    checks: list[str]
    warnings: list[str]


def statistical_validity_checklist(
    config: dict[str, object], columns: set[str]
) -> dict[str, list[str]]:
    checks = [
        "Split data before imputation, scaling, feature selection, or tuning.",
        "Fit transforms only on training folds and apply them to validation/test folds.",
        (
            f"Keep the final {config.get('holdout_fraction', 0.2):.0%} holdout "
            "untouched until model selection is complete."
        ),
    ]
    warnings: list[str] = []
    time_column = config.get("time_column")
    group_column = config.get("group_column")
    target = config.get("target_column")
    if time_column:
        checks.append(
            f"Use chronological/walk-forward splits based on '{time_column}'."
        )
    else:
        warnings.append(
            "No time column declared; temporal leakage cannot be ruled out."
        )
    if group_column:
        checks.append(f"Keep groups from '{group_column}' in only one split.")
    else:
        warnings.append(
            "No group column declared; repeated-entity leakage cannot be ruled out."
        )
    if target:
        if target not in columns:
            warnings.append(f"Configured target '{target}' is absent from the dataset.")
        warnings.append(
            f"Audit features for target-derived or post-outcome leakage into '{target}'."
        )
    else:
        warnings.append("Target is not frozen; predictive analysis needs confirmation.")
    return {"checks": checks, "warnings": warnings}


def validate_proposal(
    step: AnalysisStep, proposal: WorkerProposal, config: dict[str, object]
) -> ValidationResult:
    try:
        tree = ast.parse(proposal.code)
    except SyntaxError as exc:
        raise ValueError(f"generated code has invalid syntax: {exc}") from exc

    errors: list[str] = []
    detected_tags = set(step.risk_tags) | set(proposal.risk_tags)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.split(".")[0] in NETWORK_MODULES for name in names):
                errors.append("network client imports are forbidden")
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in {"eval", "exec", "compile"}:
                errors.append(f"{call_name}() is forbidden")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target = str(config.get("target_column") or "")
            if target and any(_assigned_column(item) == target for item in targets):
                detected_tags.add("target_change")

    lowered = proposal.code.lower()
    if re.search(r"\b(pip|uv|conda)\b.{0,40}\binstall\b", lowered, re.DOTALL):
        detected_tags.add("package_install")
    if re.search(r"\.(dropna|drop_duplicates)\s*\(", lowered) or re.search(
        r"\.drop\s*\([^)]*axis\s*=\s*0", lowered
    ):
        detected_tags.add("row_drop")
    target = str(config.get("target_column") or "")
    if target and re.search(
        rf"(rename|assign|drop)\s*\([^)]*['\"]{re.escape(target)}['\"]", proposal.code
    ):
        detected_tags.add("target_change")
    if errors:
        raise ValueError("; ".join(sorted(set(errors))))

    return ValidationResult(
        risk_tags=sorted(detected_tags),
        checks=[
            "Python parsed successfully.",
            "No network-client imports or dynamic eval/exec calls were found.",
            "Proposal hash will be bound to any approval and execution.",
        ],
        warnings=[],
    )


def requires_approval(risk_tags: list[str], approval_policy: str) -> bool:
    if MANDATORY_APPROVAL_TAGS.intersection(risk_tags):
        return True
    return approval_policy == "always"


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _assigned_column(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None
