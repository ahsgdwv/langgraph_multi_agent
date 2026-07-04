"""LLM 封装：统一调用入口，记录耗时与 token 消耗。"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from state import LLMCallLog, RunMetrics


def get_llm_status() -> str:
    if os.getenv("DEEPSEEK_API_KEY"):
        return f"deepseek ({os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')})"
    if os.getenv("OPENAI_API_KEY"):
        return f"openai ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
    return "未配置（本地规则）"


def get_llm() -> Optional[ChatOpenAI]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            temperature=0,
        )
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=0,
        )
    return None


def tracked_invoke(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    *,
    node: str,
    metrics: Optional[RunMetrics] = None,
) -> Any:
    """调用 LLM 并写入 RunMetrics 日志。"""
    started = time.perf_counter()
    result_msg = llm.invoke(messages)
    latency_ms = (time.perf_counter() - started) * 1000

    prompt_t = completion_t = total_t = 0
    if metrics is not None:
        raw = getattr(result_msg, "response_metadata", {}) or {}
        usage = raw.get("token_usage") or raw.get("usage") or {}
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        total_t = int(usage.get("total_tokens", 0) or prompt_t + completion_t)
        metrics.llm_calls.append(
            LLMCallLog(
                node=node,
                model=getattr(llm, "model_name", str(llm.model)),
                latency_ms=round(latency_ms, 2),
                prompt_tokens=prompt_t,
                completion_tokens=completion_t,
                total_tokens=total_t,
            )
        )
    return result_msg
