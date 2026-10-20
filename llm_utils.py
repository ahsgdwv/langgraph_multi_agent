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
    dk = os.getenv("DEEPSEEK_API_KEY")
    if dk:
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=dk,
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            temperature=0,
        )
    ok = os.getenv("OPENAI_API_KEY")
    if ok:
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=ok, temperature=0)
    return None


def tracked_invoke(llm, messages: list[BaseMessage], *, node: str, metrics: Optional[RunMetrics] = None) -> Any:
    t0 = time.perf_counter()
    print(f"[llm] {node} start")
    try:
        reply = llm.invoke(messages)
    except Exception as e:
        print(f"[llm] {node} fail: {e}")
        raise
    ms = (time.perf_counter() - t0) * 1000
    print(f"[llm] {node} done {ms:.0f}ms")

    if metrics is not None:
        meta = getattr(reply, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or pt + ct)
        mname = getattr(llm, "model_name", None) or str(getattr(llm, "model", "?"))
        metrics.llm_calls.append(
            LLMCallLog(node=node, model=mname, latency_ms=round(ms, 2),
                       prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
        )
    return reply
