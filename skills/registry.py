from __future__ import annotations

from typing import Any

from skills.base import BaseSkill, SkillContext
from skills.models import SkillOutput

_REGISTRY: dict[str, BaseSkill] = {}


def register_skill_instance(skill: BaseSkill) -> None:
    _REGISTRY[skill.skill_id] = skill


def list_skills() -> list[dict[str, Any]]:
    return [
        {
            "id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "triggers": s.triggers,
            "output_sections": s.output_sections,
        }
        for s in _REGISTRY.values()
    ]


def match_and_execute_skill(text: str, ctx: SkillContext | None = None) -> SkillOutput | None:
    ctx = ctx or SkillContext()
    winner: tuple[int, int, BaseSkill] | None = None

    for skill in _REGISTRY.values():
        if not skill.matches(text):
            continue
        hits = skill.score(text)
        rank = (hits, skill.priority)
        if winner is None or rank > (winner[0], winner[1]):
            winner = (hits, skill.priority, skill)

    if winner is None:
        return None
    return winner[2].run(text, ctx)
