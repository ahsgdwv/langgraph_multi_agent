from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from config import CFG
from paths import DOCUMENTS_DIR
from state import RetrievedDoc

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(CFG.rag.chroma_dir)
_vectorstore = None


def _get_embeddings():
    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=openai_key)
    from langchain_community.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_get_embeddings(),
        collection_name=CFG.rag.collection_name,
    )
    return _vectorstore


def reset_vectorstore() -> None:
    global _vectorstore
    _vectorstore = None


def list_document_files() -> list[Path]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for pattern in ("*.md", "*.txt"):
        files.extend(DOCUMENTS_DIR.glob(pattern))
    return sorted(files)


def _valid_doc_names() -> set[str]:
    return {p.name for p in list_document_files()}


def _indexed_sources(store) -> set[str]:
    try:
        data = store._collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        return {str(m.get("source")) for m in metas if m and m.get("source")}
    except Exception as exc:
        logger.warning("读取向量库索引失败: %s", exc)
        return set()


def _needs_reindex(store, files: list[Path]) -> bool:
    return {p.name for p in files} != _indexed_sources(store)


def ingest_documents(*, force: bool = False) -> dict:
    global _vectorstore
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files = list_document_files()
    if not files:
        return {"loaded": 0, "chunks": 0, "message": "无文档可加载", "ok": False}

    try:
        store = get_vectorstore()
        existing = store._collection.count()
    except Exception as exc:
        logger.warning("向量库不可用，尝试重建: %s", exc)
        existing = 0
        force = True

    if existing > 0 and not force and not _needs_reindex(store, files):
        return {
            "loaded": len(files),
            "chunks": existing,
            "message": f"向量库已有 {existing} 条",
            "ok": True,
        }

    if force or (existing > 0 and _needs_reindex(store, files)):
        reset_vectorstore()
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        store = get_vectorstore()

    docs: list[Document] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("读取文档失败 %s: %s", path.name, exc)
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name, "path": str(path)}))

    if not docs:
        return {"loaded": 0, "chunks": 0, "message": "文档均无法读取", "ok": False}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.rag.chunk_size,
        chunk_overlap=CFG.rag.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    if chunks:
        store.add_documents(chunks)

    return {
        "loaded": len(docs),
        "chunks": len(chunks),
        "files": [p.name for p in files if p.exists()],
        "message": f"已入库 {len(docs)} 个文件，{len(chunks)} 个片段",
        "ok": True,
    }


def retrieve_context(query: str, *, k: Optional[int] = None) -> list[RetrievedDoc]:
    k = k or CFG.rag.default_k
    valid = _valid_doc_names()
    try:
        store = get_vectorstore()
        count = store._collection.count()
    except Exception as exc:
        logger.warning("检索前向量库异常: %s", exc)
        return []

    if count == 0:
        result = ingest_documents()
        if not result.get("ok"):
            return []
        try:
            count = get_vectorstore()._collection.count()
        except Exception:
            return []

    try:
        results = store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.warning("检索失败: %s", exc)
        return []

    docs: list[RetrievedDoc] = []
    for doc, score in results:
        source = str(doc.metadata.get("source", "unknown"))
        if source not in valid:
            continue
        docs.append(
            RetrievedDoc(
                source=source,
                content=doc.page_content,
                score=round(float(score), 4),
            )
        )
    return docs


def format_retrieved_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "（未检索到相关业务文档）"
    return "\n\n".join(f"[{i}] 来源:{d.source}\n{d.content}" for i, d in enumerate(docs, 1))
