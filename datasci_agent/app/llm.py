from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from .config import Settings
from .models import AnalysisPlan, AnalysisStep, WorkerProposal


PLANNER_PROMPT_VERSION = "planner-v1"
WORKER_PROMPT_VERSION = "worker-v1"


class DataScienceLLM:
    """Two logical roles backed by one configured chat-completions client."""

    def __init__(self, settings: Settings):
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        self.client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    async def plan(
        self,
        question: str,
        datasets: list[dict[str, Any]],
        profiles: list[dict[str, Any]],
        config: dict[str, Any],
        based_on_event_seq: int,
    ) -> AnalysisPlan:
        schema = AnalysisPlan.model_json_schema()
        content = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the planner role in a data-science agent. Return only "
                        "strict JSON matching the supplied schema. Produce a small, "
                        "topologically ordered step DAG. Preserve an untouched holdout, "
                        "split by time/group when configured, fit transforms on training "
                        "folds only, and flag target changes, row drops, and package "
                        "installs in risk_tags."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt_version": PLANNER_PROMPT_VERSION,
                            "question": question,
                            "datasets": datasets,
                            "profiles": profiles,
                            "config": config,
                            "required_plan_fields": schema,
                            "required_values": {
                                "version": 1,
                                "parent_version": None,
                                "based_on_event_seq": based_on_event_seq,
                            },
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
        plan = AnalysisPlan.model_validate_json(content)
        if (
            plan.version != 1
            or plan.parent_version is not None
            or plan.based_on_event_seq != based_on_event_seq
        ):
            raise ValueError("planner returned incorrect plan version metadata")
        if len(plan.steps) > config["max_steps"]:
            raise ValueError("planner exceeded max_steps")
        return plan

    async def propose_step(
        self,
        step: AnalysisStep,
        dataset_path: str,
        committed_artifacts: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> WorkerProposal:
        content = await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the worker role. Return only strict JSON matching the "
                        "WorkerProposal schema. Emit Python code for exactly one step. "
                        "Read the CSV from /input/data.csv and write only allowlisted "
                        "text, JSON, CSV, or Markdown outputs under /output. Never use "
                        "network access. Never install packages. Split before transforms "
                        "and fit transforms on training folds only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt_version": WORKER_PROMPT_VERSION,
                            "step": step.model_dump(mode="json"),
                            "dataset_runtime_path": "/input/data.csv",
                            "dataset_storage_ref": dataset_path,
                            "committed_artifacts": committed_artifacts,
                            "config": config,
                            "schema": WorkerProposal.model_json_schema(),
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
        return WorkerProposal.model_validate_json(content)

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return content
