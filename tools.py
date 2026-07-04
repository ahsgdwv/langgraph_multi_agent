"""任务拆分与结果汇总工具。"""
from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from state import ExecutionFlags, SubTask, TaskResult


class SplitTasksInput(BaseModel):
    user_input: str = Field(..., description="用户原始需求")
    max_tasks: int = Field(default=10, ge=1, le=10)


class AggregateResultsInput(BaseModel):
    task_list_json: str
    task_results_json: str
    reflection_summary: Optional[str] = None


_PREAMBLE_TRIGGERS = (
    "以下工作：",
    "工作内容：",
    "具体任务：",
    "全部子任务：",
    "子任务：",
    "需要完成：",
    "任务：",
)


def _strip_preamble(text: str) -> str:
    """去掉需求前缀，保留子任务列表正文"""
    for trigger in _PREAMBLE_TRIGGERS:
        if trigger in text:
            return text.split(trigger, 1)[1].strip()
    match = re.search(r"[：:]\s*(?=\d+[\.、]\s*)", text)
    if match:
        return text[match.end() :].strip()
    return text.strip()


def _split_numbered_items(text: str) -> list[str]:
    """解析「1. xxx；2. xxx」或「1、xxx；2、xxx」编号列表"""
    if len(re.findall(r"\d+[\.、]\s*", text)) < 2:
        return []

    normalized = re.sub(r"[；;]\s*(?=\d+[\.、]\s*)", "\n", text)
    parts = re.split(r"(?<=\n)|(?=^\d+[\.、]\s*)", normalized)
    if len(parts) <= 1:
        parts = re.split(r"(?=\d+[\.、]\s*)", normalized)

    chunks: list[str] = []
    for part in parts:
        item = re.sub(r"^\d+[\.、]\s*", "", part.strip())
        item = item.strip().rstrip("；;，,")
        if len(item) >= 2 and not re.fullmatch(r"[\d\.、；;，,\s]+", item):
            chunks.append(item)
    return chunks


def _split_delimited_items(text: str) -> list[str]:
    """解析顿号/逗号/分号分隔的任务列表"""
    chunks = re.split(r"[、，,;；\n]+", text)
    return [c.strip() for c in chunks if len(c.strip()) >= 2]


def count_expected_tasks(user_input: str, *, max_tasks: int = 10) -> int:
    """估算用户需求中的子任务条目数，供拆分与反思校验"""
    text = _strip_preamble(user_input.strip())
    if not text:
        return 0
    numbered = _split_numbered_items(text)
    if numbered:
        return min(len(numbered), max_tasks)
    delimited = _split_delimited_items(text)
    if len(delimited) > 1:
        return min(len(delimited), max_tasks)
    return 1


def infer_max_tasks(user_input: str, *, ceiling: int = 10) -> int:
    """根据输入推断应拆分的子任务上限"""
    expected = count_expected_tasks(user_input, max_tasks=ceiling)
    return max(expected, 1) if expected else ceiling


def _parse_user_input_to_tasks(user_input: str, max_tasks: int = 10) -> list[SubTask]:
    """将自然语言需求拆分为子任务列表"""
    raw = user_input.strip()
    if not raw:
        return []

    text = _strip_preamble(raw)
    chunks = _split_numbered_items(text)
    if not chunks:
        chunks = _split_delimited_items(text)
    if not chunks:
        chunks = [raw]

    limit = max_tasks if max_tasks > 0 else infer_max_tasks(raw)
    chunks = chunks[:limit]

    tasks: list[SubTask] = []
    for i, chunk in enumerate(chunks, start=1):
        task_id = f"T{i:03d}"
        priority: str = "high" if re.search(r"紧急|重要|优先|high", chunk, re.I) else "medium"
        title = chunk if len(chunk) <= 60 else chunk[:57] + "..."
        tasks.append(
            SubTask(
                task_id=task_id,
                title=title,
                description=chunk,
                priority=priority,  # type: ignore[arg-type]
                assignee="executor",
                status="pending",
            )
        )
    return tasks


def _build_summary(
    task_list: list[SubTask],
    task_results: dict[str, TaskResult],
    reflection_summary: Optional[str] = None,
) -> str:
    """组装 Markdown 汇总报告"""
    total = len(task_list)
    success_count = sum(
        1
        for t in task_list
        if task_results.get(t.task_id, TaskResult(task_id=t.task_id, success=False)).success
    )

    lines: list[str] = [
        "# 快消业务分析周报",
        "",
        f"- 分析项：{total} 项",
        f"- 完成：{success_count} 项 | 未完成：{total - success_count} 项",
        "",
        "## 分项结果",
    ]

    for task in task_list:
        result = task_results.get(task.task_id)
        if result is None:
            status_icon = "[待执行]"
            detail = "（尚无执行结果）"
        elif result.success:
            status_icon = "[成功]"
            detail = result.output or "（无输出）"
        else:
            status_icon = "[失败]"
            detail = result.error or result.output or "（未知错误）"

        lines.append(
            f"### [{task.task_id}] {task.title} {status_icon}\n"
            f"{detail}"
        )

    if reflection_summary:
        lines.extend(["", "## 质量复核", reflection_summary])

    lines.extend(["", "---", f"*报告 ID: RPT-{uuid.uuid4().hex[:8]}*"])
    return "\n".join(lines)


def apply_split_to_state(user_input: str, max_tasks: Optional[int] = None) -> dict:
    """拆分任务并返回 State 局部更新；max_tasks 默认按输入条目数推断"""
    limit = max_tasks if max_tasks is not None else infer_max_tasks(user_input)
    tasks = _parse_user_input_to_tasks(user_input, limit)
    return {
        "task_list": tasks,
        "execution_flags": ExecutionFlags(planning_done=True, current_task_index=0),
    }


def apply_aggregate_to_state(
    task_list: list[SubTask],
    task_results: dict[str, TaskResult],
    reflection_summary: Optional[str] = None,
    current_flags: Optional[ExecutionFlags] = None,
) -> dict:
    """汇总结果并返回 State 局部更新，保留 retry_count"""
    summary = _build_summary(task_list, task_results, reflection_summary)
    base = current_flags or ExecutionFlags()
    return {
        "final_output": summary,
        "execution_flags": ExecutionFlags(
            planning_done=True,
            execution_done=True,
            reflection_done=True,
            is_finished=True,
            max_retry=base.max_retry,
            retry_count=base.retry_count,
            need_retry=False,
        ),
    }


@tool(args_schema=SplitTasksInput)
def split_tasks(user_input: str, max_tasks: int = 10) -> str:
    """将用户需求拆分为结构化子任务列表"""
    tasks = _parse_user_input_to_tasks(user_input, max_tasks)
    payload = {
        "task_list": [t.model_dump() for t in tasks],
        "execution_flags": {"planning_done": True, "current_task_index": 0},
        "message": f"已拆分 {len(tasks)} 个子任务",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(args_schema=AggregateResultsInput)
def aggregate_results(
    task_list_json: str,
    task_results_json: str,
    reflection_summary: Optional[str] = None,
) -> str:
    """合并子任务与执行结果为最终报告"""
    try:
        raw_tasks = json.loads(task_list_json)
        task_list = [SubTask.model_validate(t) for t in raw_tasks]
    except (json.JSONDecodeError, ValidationError) as e:
        return json.dumps({"error": f"task_list_json 解析失败: {e}"}, ensure_ascii=False)

    try:
        raw_results = json.loads(task_results_json)
        task_results = {k: TaskResult.model_validate(v) for k, v in raw_results.items()}
    except (json.JSONDecodeError, ValidationError) as e:
        return json.dumps({"error": f"task_results_json 解析失败: {e}"}, ensure_ascii=False)

    summary = _build_summary(task_list, task_results, reflection_summary)
    payload = {
        "final_output": summary,
        "execution_flags": {
            "planning_done": True,
            "execution_done": True,
            "reflection_done": True,
            "is_finished": True,
        },
        "message": "汇总报告已生成",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


ALL_TOOLS = [split_tasks, aggregate_results]


if __name__ == "__main__":
    print(split_tasks.invoke({"user_input": "撰写发布说明、通知团队、整理检查清单"}))
