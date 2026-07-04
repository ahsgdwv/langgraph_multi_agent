"""Skill 基类，入参与 AgentState 对齐。"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from skills.models import SkillOutput, format_skill_markdown

if TYPE_CHECKING:
    from state import AgentState


class SkillContext(BaseModel):
    user_input: str = ""
    task_id: str = ""
    thread_id: str = ""
    retrieved_docs: list[dict] = Field(default_factory=list)


def skill_context_from_state(state: "AgentState", task_id: str = "") -> SkillContext:
    docs = state.get("retrieved_context") or []
    return SkillContext(
        user_input=state.get("user_input", ""),
        task_id=task_id,
        thread_id=state.get("thread_id", ""),
        retrieved_docs=[
            d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in docs
        ],
    )


class BaseSkill(ABC):
    skill_id: str
    name: str
    description: str
    triggers: list[str]
    output_sections: list[str]

    def matches(self, text: str) -> bool:
        return any(re.search(p, text, re.I) for p in self.triggers)

    def score(self, text: str) -> int:
        return sum(1 for p in self.triggers if re.search(p, text, re.I))

    def run(self, text: str, ctx: SkillContext) -> SkillOutput:
        try:
            output = self._execute(text, ctx)
            if not output.markdown:
                output.markdown = format_skill_markdown(output, self.output_sections)
            return output
        except Exception as exc:
            return SkillOutput(
                skill_id=self.skill_id,
                skill_name=self.name,
                summary=f"执行失败: {exc}",
                sections={"错误信息": str(exc)},
                markdown=f"## [{self.skill_id}] 执行失败\n\n{exc}",
                data_sources=[],
            )

    @abstractmethod
    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        ...
