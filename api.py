"""FastAPI 服务：封装任务调度为 HTTP 接口。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

from main import get_checkpointer, resume_run, run_until_pause_or_finish
from rag_store import DOCS_DIR, ingest_documents, list_document_files

app = FastAPI(
    title="FMCG Analytics Agent API",
    description="快消数据分析多 Agent API（Skill 资产库 + SQL/pandas + RAG + Supervisor）",
    version="3.0.0",
)


class RunRequest(BaseModel):
    user_input: str = Field(..., min_length=2)
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


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool = True
    comment: str = ""


@app.get("/health")
def health() -> dict:
    from llm_utils import get_llm_status

    return {
        "status": "ok",
        "llm": get_llm_status(),
        "supervisor": os.getenv("USE_SUPERVISOR", "true"),
        "documents": [p.name for p in list_document_files()],
    }


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest) -> RunResponse:
    if req.auto_approve:
        os.environ["AUTO_APPROVE_HUMAN"] = "true"

    with get_checkpointer() as cp:
        state, finished, _ = run_until_pause_or_finish(
            req.user_input,
            thread_id=req.thread_id,
            max_retry=req.max_retry,
            checkpointer=cp,
            verbose=False,
        )
        tid = state.get("thread_id", "")
        while not finished:
            state, finished, _ = resume_run(
                tid,
                {"approved": True, "comment": "API续跑"},
                checkpointer=cp,
                verbose=False,
            )

    metrics = state.get("run_metrics")
    metrics_dict = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics or {})
    tasks = state.get("task_list") or []
    results = state.get("task_results") or {}
    success = sum(1 for t in tasks if results.get(t.task_id) and results[t.task_id].success)

    return RunResponse(
        thread_id=state.get("thread_id", ""),
        finished=state["execution_flags"].is_finished,
        final_output=state.get("final_output") or "",
        metrics=metrics_dict,
        task_count=len(tasks),
        success_count=success,
    )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".md", ".txt"):
        raise HTTPException(400, "仅支持 .md / .txt")
    target = DOCS_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)
    result = ingest_documents(force=True)
    return {"saved": file.filename, **result}


@app.post("/documents/reload")
def reload_documents() -> dict:
    return ingest_documents(force=True)


@app.get("/skills")
def list_agent_skills() -> dict:
    from skills import list_skills

    return {"count": len(list_skills()), "skills": list_skills()}


@app.get("/documents")
def list_documents() -> dict:
    files = list_document_files()
    return {"count": len(files), "files": [f.name for f in files]}
