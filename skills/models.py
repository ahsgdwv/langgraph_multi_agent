from __future__ import annotations

from pydantic import BaseModel, Field


class SkillOutput(BaseModel):
    skill_id: str
    skill_name: str
    summary: str
    sections: dict[str, str] = Field(default_factory=dict)
    markdown: str
    data_sources: list[str] = Field(default_factory=list)


class SkillDefinition(BaseModel):
    id: str
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    output_sections: list[str] = Field(
        default_factory=lambda: ["数据概览", "关键发现", "建议动作"]
    )

    def matches(self, text: str) -> bool:
        import re
        return any(re.search(p, text, re.I) for p in self.triggers)


def format_skill_markdown(output: SkillOutput, sections: list[str]) -> str:
    lines = [f"## [{output.skill_id}] {output.skill_name}", "", output.summary, ""]
    for section in sections:
        body = output.sections.get(section, "").strip()
        if body:
            lines.extend([f"### {section}", body, ""])
    if output.data_sources:
        lines.append(f"*数据来源：{', '.join(output.data_sources)}*")
    return "\n".join(lines).strip()
