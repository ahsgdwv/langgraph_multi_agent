from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
from pathlib import Path

from langchain_core.embeddings import Embeddings

from config import CFG
from infra.settings import get_settings
from paths import DOCUMENTS_DIR
from state import RetrievedDoc

log = logging.getLogger(__name__)
chroma_path = Path(CFG.rag.chroma_dir)
_chroma = None


class HashingEmbeddings(Embeddings):
    """无外部依赖的确定性特征哈希 embedding，避免离线时退回随机向量。"""

    _token_re = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

    def __init__(self, size: int = 384):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        normalized = (text or "").lower()
        tokens = self._token_re.findall(normalized)
        chars = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
        grams = tokens + ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        vector = [0.0] * self.size
        for token in grams:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _embed_fn():
    settings = get_settings()
    if settings.embedding_provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=settings.embedding_model,
        )
    if settings.embedding_provider == "openai":
        log.warning("EMBEDDING_PROVIDER=openai 但缺少 OPENAI_API_KEY，回退到哈希向量")
    return HashingEmbeddings(size=384)


def get_vectorstore():
    global _chroma
    if _chroma is not None:
        return _chroma
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    chroma_path.mkdir(parents=True, exist_ok=True)
    _chroma = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=_embed_fn(),
        collection_name=CFG.rag.collection_name,
    )
    return _chroma


def reset_vectorstore():
    global _chroma
    _chroma = None


def list_document_files():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    found = []
    for pat in ("*.md", "*.txt"):
        found.extend(DOCUMENTS_DIR.glob(pat))
    return sorted(found)


def _disk_names():
    return {p.name for p in list_document_files()}


def _index_names(store):
    try:
        raw = store._collection.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        return {str(m.get("source")) for m in metas if m and m.get("source")}
    except Exception as e:
        log.warning("读索引名失败: %s", e)
        return set()


def ingest_documents(*, force=False):
    global _chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files = list_document_files()
    if not files:
        return {"loaded": 0, "chunks": 0, "message": "没文档", "ok": False}

    try:
        vs = get_vectorstore()
        n = vs._collection.count()
    except Exception as e:
        log.warning("向量库坏了，重建: %s", e)
        n = 0
        force = True

    if n > 0 and not force and _disk_names() == _index_names(vs):
        return {"loaded": len(files), "chunks": n, "message": f"已有 {n} 条", "ok": True}

    if force or (n > 0 and _disk_names() != _index_names(vs)):
        try:
            # 先删除 collection，避免 Windows 文件锁导致 rmtree 失败后旧数据残留
            vs.delete_collection()
        except Exception as e:
            log.warning("清空旧 collection 失败: %s", e)
        reset_vectorstore()
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)
        chroma_path.mkdir(parents=True, exist_ok=True)
        vs = get_vectorstore()

    docs = []
    for fp in files:
        try:
            txt = fp.read_text(encoding="utf-8")
        except OSError as e:
            log.error("读 %s 失败: %s", fp.name, e)
            continue
        if not txt.strip():
            continue
        docs.append(Document(page_content=txt, metadata={"source": fp.name, "path": str(fp)}))

    if not docs:
        return {"loaded": 0, "chunks": 0, "message": "全读失败", "ok": False}

    splitter = RecursiveCharacterTextSplitter(chunk_size=CFG.rag.chunk_size, chunk_overlap=CFG.rag.chunk_overlap)
    chunks = splitter.split_documents(docs)
    if chunks:
        vs.add_documents(chunks)
    print(f"[rag] ingest {len(docs)} files -> {len(chunks)} chunks")

    return {
        "loaded": len(docs),
        "chunks": len(chunks),
        "files": [f.name for f in files],
        "message": f"入库 {len(docs)} 文件",
        "ok": True,
    }


def retrieve_context(query: str, *, k=None) -> list[RetrievedDoc]:
    topk = k or CFG.rag.default_k
    if not query or not query.strip():
        return []
    allowed = _disk_names()

    try:
        vs = get_vectorstore()
        cnt = vs._collection.count()
    except Exception as e:
        log.warning("检索前库异常: %s", e)
        return []

    if cnt == 0:
        ing = ingest_documents()
        if not ing.get("ok"):
            return []
        try:
            cnt = get_vectorstore()._collection.count()
        except Exception:
            return []

    try:
        hits = vs.similarity_search_with_score(query, k=max(topk * 3, topk))
    except Exception as e:
        log.warning("检索挂了: %s", e)
        return []

    out: list[RetrievedDoc] = []
    seen: set[tuple[str, str]] = set()
    for page, score in hits:
        src = str(page.metadata.get("source", "unknown"))
        if allowed and src not in allowed:
            continue
        key = (src, page.page_content.strip()[:240])
        if key in seen:
            continue
        seen.add(key)
        out.append(RetrievedDoc(source=src, content=page.page_content, score=round(float(score), 4)))
        if len(out) >= topk:
            break
    return out


def format_retrieved_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "（未检索到相关业务文档）"
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] 来源:{d.source}\n{d.content}")
    return "\n\n".join(lines)
