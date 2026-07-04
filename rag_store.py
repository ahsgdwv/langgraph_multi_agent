"""RAG 向量库：Chroma 持久化 + 文档检索。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from state import RetrievedDoc

PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

_vectorstore = None


def _get_embeddings():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=openai_key)
    from langchain_community.embeddings import FakeEmbeddings

    return FakeEmbeddings(size=384)


def get_vectorstore():
    """懒加载 Chroma 向量库。"""
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
        collection_name="business_docs",
    )
    return _vectorstore


def list_document_files() -> list[Path]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for pattern in ("*.md", "*.txt"):
        files.extend(DOCS_DIR.glob(pattern))
    return sorted(files)


def ingest_documents(*, force: bool = False) -> dict:
    """加载 data/documents 下文档并写入 Chroma。"""
    global _vectorstore
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files = list_document_files()
    if not files:
        return {"loaded": 0, "chunks": 0, "message": "无文档可加载"}

    store = get_vectorstore()
    try:
        existing = store._collection.count()
    except Exception:
        existing = 0

    if existing > 0 and not force:
        return {
            "loaded": len(files),
            "chunks": existing,
            "message": f"向量库已有 {existing} 条，跳过重复入库",
        }

    if force:
        _vectorstore = None
        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        store = get_vectorstore()

    docs: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            )
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    if chunks:
        store.add_documents(chunks)

    return {
        "loaded": len(files),
        "chunks": len(chunks),
        "files": [p.name for p in files],
        "message": f"已入库 {len(files)} 个文件，{len(chunks)} 个片段",
    }


def retrieve_context(query: str, *, k: int = 4) -> list[RetrievedDoc]:
    """检索与 query 相关的业务文档片段。"""
    store = get_vectorstore()
    try:
        count = store._collection.count()
    except Exception:
        count = 0
    if count == 0:
        ingest_documents()
        try:
            count = store._collection.count()
        except Exception:
            count = 0

    if count == 0:
        return []

    results = store.similarity_search_with_score(query, k=k)
    docs: list[RetrievedDoc] = []
    for doc, score in results:
        docs.append(
            RetrievedDoc(
                source=str(doc.metadata.get("source", "unknown")),
                content=doc.page_content,
                score=round(float(score), 4),
            )
        )
    return docs


def format_retrieved_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "（未检索到相关业务文档）"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源:{d.source}\n{d.content}")
    return "\n\n".join(parts)
