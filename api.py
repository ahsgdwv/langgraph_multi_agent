from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

from main import get_checkpointer, resume_run, run_until_pause_or_finish
from paths import DOCUMENTS_DIR
from prompt_optimizer import (
    compare_test,
    format_length_stats,
    iterate_prompt,
    length_stats,
    list_templates,
    optimize_prompt,
    save_optimized_prompt,
    test_prompt,
)
from rag_store import ingest_documents, list_document_files

app = FastAPI(title="fmcg-agent-api", version="3.0")


class RunRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    max_retry: int = Field(default=2, ge=0, le=5)
    thread_id: Optional[str] = None
    auto_approve: bool = True


class RunResponse(BaseModel):
    thread_id: str
    finished: bool
    final_output: str
    metrics: dict[str, Any]
    task_count: int
    success_count: int
    interrupted: bool = False
    interrupt: Optional[dict[str, Any]] = None


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True
    comment: str = ""


class PromptOptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="system", pattern="^(system|user)$")
    template_id: Optional[str] = None


class PromptIterateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    requirements: str = Field(..., min_length=1)
    template_id: Optional[str] = None


class PromptTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    test_input: str = Field(..., min_length=1)
    mode: str = Field(default="system", pattern="^(system|user|iterate)$")


class PromptResult(BaseModel):
    optimized: str
    stats: dict[str, Any] = Field(default_factory=dict)
    stats_text: str = ""


class PromptTestResult(BaseModel):
    output: str


class PromptCompareRequest(BaseModel):
    original: str = Field(..., min_length=1)
    optimized: str = Field(..., min_length=1)
    test_input: str = Field(..., min_length=1)
    mode: str = Field(default="system", pattern="^(system|user|iterate)$")


class PromptCompareResult(BaseModel):
    original_output: str
    optimized_output: str
    stats: dict[str, Any] = Field(default_factory=dict)
    stats_text: str = ""


class PromptSaveRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    mode: str = Field(default="system")


def _run_to_finish(user_text: str, thread_id=None, max_retry=2, auto_approve=True):
    if not user_text or not user_text.strip():
        raise HTTPException(400, "user_input 不能为空")
    os.environ["AUTO_APPROVE_HUMAN"] = "true" if auto_approve else "false"

    print(f"[api] run start tid={thread_id or 'new'} retry={max_retry}")
    with get_checkpointer() as cp:
        state, done, interrupt_payload = run_until_pause_or_finish(
            user_text.strip(),
            thread_id=thread_id,
            max_retry=max_retry,
            checkpointer=cp,
            verbose=False,
        )
        tid = state.get("thread_id") or ""
        while auto_approve and not done:
            state, done, interrupt_payload = resume_run(
                tid,
                {"approved": True, "comment": "api续跑"},
                checkpointer=cp,
                verbose=False,
            )

    return state, tid, interrupt_payload


def _build_response(
    state, tid: str, finished: bool, interrupt_payload: Optional[dict[str, Any]]
) -> RunResponse:
    tasks = state.get("task_list") or []
    outcomes = state.get("task_results") or {}
    ok_n = 0
    for t in tasks:
        tr = outcomes.get(t.task_id)
        if tr and getattr(tr, "success", False):
            ok_n += 1
        elif isinstance(tr, dict) and tr.get("success"):
            ok_n += 1

    m = state.get("run_metrics")
    metrics_dict = m.model_dump() if hasattr(m, "model_dump") else dict(m or {})

    print(f"[api] run end tid={tid} tasks={len(tasks)} ok={ok_n} finished={finished}")
    return RunResponse(
        thread_id=tid,
        finished=finished,
        final_output=state.get("final_output") or "",
        metrics=metrics_dict,
        task_count=len(tasks),
        success_count=ok_n,
        interrupted=not finished,
        interrupt=interrupt_payload,
    )


@app.get("/health")
def health():
    from llm_utils import get_llm_status
    from branding import get_brand_name, get_display_title
    return {
        "status": "ok",
        "llm": get_llm_status(),
        "supervisor": os.getenv("USE_SUPERVISOR", "true"),
        "brand_name": get_brand_name() or None,
        "display_title": get_display_title(),
        "documents": [p.name for p in list_document_files()],
    }


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest):
    state, tid, interrupt_payload = _run_to_finish(
        req.user_input, req.thread_id, req.max_retry, req.auto_approve
    )
    flags = state.get("execution_flags")
    finished = getattr(flags, "is_finished", False) if flags else False
    return _build_response(state, tid, finished, interrupt_payload)


@app.post("/resume", response_model=RunResponse)
def resume_task(req: ResumeRequest):
    with get_checkpointer() as cp:
        state, done, interrupt_payload = resume_run(
            req.thread_id,
            {"approved": req.approved, "comment": req.comment},
            checkpointer=cp,
            verbose=False,
        )
    tid = state.get("thread_id") or req.thread_id
    flags = state.get("execution_flags")
    finished = getattr(flags, "is_finished", False) if flags else done
    return _build_response(state, tid, finished, interrupt_payload)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(400, "没文件名")
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(400, "没文件名")
    ext = Path(safe_name).suffix.lower()
    if ext not in (".md", ".txt"):
        raise HTTPException(400, "只要 md/txt")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "文件不能超过 5MB")
    dest = DOCUMENTS_DIR / safe_name
    dest.write_bytes(content)
    print(f"[api] upload {safe_name}")
    ing = ingest_documents(force=True)
    return {"saved": safe_name, **ing}


@app.post("/documents/reload")
def reload_documents():
    print("[api] reload docs")
    return ingest_documents(force=True)


@app.get("/skills")
def list_agent_skills():
    from skills import list_skills
    skills = list_skills()
    return {"count": len(skills), "skills": skills}


@app.get("/documents")
def list_documents():
    files = list_document_files()
    return {"count": len(files), "files": [f.name for f in files]}


@app.get("/prompt/templates")
def prompt_templates(mode: Optional[str] = None):
    if mode is not None and mode not in ("system", "user", "iterate"):
        raise HTTPException(400, "mode 只能是 system|user|iterate")
    items = list_templates(mode)  # type: ignore[arg-type]
    return {"count": len(items), "templates": items}


@app.post("/prompt/optimize", response_model=PromptResult)
def prompt_optimize(req: PromptOptimizeRequest):
    try:
        out = optimize_prompt(req.prompt, req.mode, req.template_id)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        print(f"[api] prompt optimize fail: {e}")
        raise HTTPException(500, f"优化失败：{e}") from e
    return PromptResult(
        optimized=out,
        stats=length_stats(req.prompt, out),
        stats_text=format_length_stats(req.prompt, out),
    )


@app.post("/prompt/iterate", response_model=PromptResult)
def prompt_iterate(req: PromptIterateRequest):
    try:
        out = iterate_prompt(req.prompt, req.requirements, req.template_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        print(f"[api] prompt iterate fail: {e}")
        raise HTTPException(500, f"迭代失败：{e}") from e
    return PromptResult(
        optimized=out,
        stats=length_stats(req.prompt, out),
        stats_text=format_length_stats(req.prompt, out),
    )


@app.post("/prompt/test", response_model=PromptTestResult)
def prompt_test(req: PromptTestRequest):
    try:
        out = test_prompt(req.prompt, req.test_input, req.mode)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        print(f"[api] prompt test fail: {e}")
        raise HTTPException(500, f"试跑失败：{e}") from e
    return PromptTestResult(output=out)


@app.post("/prompt/compare", response_model=PromptCompareResult)
def prompt_compare(req: PromptCompareRequest):
    try:
        left, right = compare_test(
            req.original, req.optimized, req.test_input, req.mode  # type: ignore[arg-type]
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        print(f"[api] prompt compare fail: {e}")
        raise HTTPException(500, f"对比失败：{e}") from e
    return PromptCompareResult(
        original_output=left,
        optimized_output=right,
        stats=length_stats(req.original, req.optimized),
        stats_text=format_length_stats(req.original, req.optimized),
    )


@app.post("/prompt/save")
def prompt_save(req: PromptSaveRequest):
    try:
        rel = save_optimized_prompt(req.prompt, mode=req.mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        print(f"[api] prompt save fail: {e}")
        raise HTTPException(500, f"保存失败：{e}") from e
    return {"path": rel}
