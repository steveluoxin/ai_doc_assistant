import io
import json
import os
from typing import Optional

import numpy as np
from docx import Document
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag import (
    analyze_document_structure,
    answer_with_rag,
    build_trunks_and_embeddings,
    embed_query,
    retrieve_for_summary,
    retrieve_top_k_chunks_with_score,
    rewrite_query,
)

# ========== Upstash KV state helpers (serverless-safe) ==========

# Detect whether Upstash env vars are present. On Vercel these are
# injected automatically when a KV/Redis store is connected.
_UPSTASH_URL = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL")
_UPSTASH_TOKEN = os.environ.get("KV_REST_API_TOKEN") or os.environ.get("UPSTASH_REDIS_REST_TOKEN")

_kv_client = None


def _get_kv_client():
    global _kv_client
    if _kv_client is None and _UPSTASH_URL and _UPSTASH_TOKEN:
        from upstash_redis import Redis
        _kv_client = Redis(url=_UPSTASH_URL, token=_UPSTASH_TOKEN)
    return _kv_client


_KV_KEYS = {
    "document": "ai_doc_assistant:document",
    "structure": "ai_doc_assistant:structure",
    "chunks": "ai_doc_assistant:chunks",
    "embeddings": "ai_doc_assistant:embeddings",
}


def _load_kv_state():
    """Load persisted state from KV."""
    kv = _get_kv_client()
    if kv is None:
        return None

    values = kv.mget(
        _KV_KEYS["document"],
        _KV_KEYS["structure"],
        _KV_KEYS["chunks"],
        _KV_KEYS["embeddings"],
    )
    if not values or values[0] is None:
        return None

    try:
        document = values[0]
        structure = json.loads(values[1]) if values[1] else {}
        chunks = json.loads(values[2]) if values[2] else []
        embeddings = np.array(json.loads(values[3])) if values[3] else np.zeros((0,))
        return {
            "document": document,
            "structure": structure,
            "chunks": chunks,
            "embeddings": embeddings,
        }
    except Exception as exc:
        print(f"Failed to deserialize KV state: {exc}")
        return None


def _save_kv_state(document: str, structure: dict, chunks: list, embeddings):
    """Persist state to KV."""
    kv = _get_kv_client()
    if kv is None:
        return

    try:
        kv.mset(
            {
                _KV_KEYS["document"]: document,
                _KV_KEYS["structure"]: json.dumps(structure),
                _KV_KEYS["chunks"]: json.dumps(chunks),
                _KV_KEYS["embeddings"]: json.dumps(embeddings.tolist()),
            }
        )
    except Exception as exc:
        print(f"Failed to save KV state: {exc}")


def _clear_kv_state():
    kv = _get_kv_client()
    if kv is None:
        return
    kv.delete(
        _KV_KEYS["document"],
        _KV_KEYS["structure"],
        _KV_KEYS["chunks"],
        _KV_KEYS["embeddings"],
    )


def _load_default_document(path: str = "doc_example.docx") -> str:
    """
    支持加载 .md / .txt / .docx 作为默认文档
    """
    if path.endswith(".docx"):
        doc = Document(path)
        full_text = []
        for para in doc.paragraphs:
            style = para.style.name
            text = para.text.strip()
            if not text:
                continue
            if style.startswith("Heading 1"):
                full_text.append(f"# {text}")
            elif style.startswith("Heading 2"):
                full_text.append(f"## {text}")
            elif style.startswith("Heading 3"):
                full_text.append(f"### {text}")
            else:
                full_text.append(text)
        return "\n".join(full_text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


# ========== Lazy state initialization ==========

def _ensure_state():
    """
    Return (document, structure, chunks, embeddings).

    First tries KV. If nothing is in KV, build from the default document,
    analyze structure, build chunks/embeddings, and persist to KV.
    """
    kv_state = _load_kv_state()
    if kv_state is not None:
        return (
            kv_state["document"],
            kv_state["structure"],
            kv_state["chunks"],
            kv_state["embeddings"],
        )

    document = _load_default_document("doc_example.docx")
    structure = analyze_document_structure(document)
    chunks, embeddings = build_trunks_and_embeddings(document)
    _save_kv_state(document, structure, chunks, embeddings)
    return document, structure, chunks, embeddings


# ========== FastAPI ==========

app = FastAPI(title="AI Product Document Assistant")
templates = Jinja2Templates(directory="templates")


class AskRequest(BaseModel):
    question: str
    task: str  # summary / risk / advice


@app.post("/ask")
def ask(req: AskRequest):
    document, structure, chunks, embeddings = _ensure_state()

    # 所有任务类型都改写问题
    rewritten_query = rewrite_query(req.question, structure, task=req.task)

    if req.task == "summary":
        results = retrieve_for_summary(
            question=rewritten_query,
            structure=structure,
            chunks=chunks,
            embeddings=embeddings,
            per_module_k=3,
        )
    else:
        query_emb = embed_query(rewritten_query)
        results = retrieve_top_k_chunks_with_score(
            chunks,
            embeddings,
            query_embedding=query_emb,
            top_k=6,
        )

    # 统一 results 结构为 (chunk, score, module)
    normalized_results = []
    for item in results:
        if len(item) == 3:
            normalized_results.append(item)
        elif len(item) == 2:
            c, s = item
            normalized_results.append((c, s, "unknown"))

    answer = answer_with_rag(
        question=req.question,
        retrieved_chunks=normalized_results,
        task=req.task,
    )

    chunks_for_frontend = [
        {"content": c, "score": s, "module": m}
        for c, s, m in normalized_results
    ]

    return {
        "rewritten_query": rewritten_query,
        "answer": answer,
        "chunks": chunks_for_frontend,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 文档解析函数

def parse_md(file_bytes: bytes) -> str:
    """
    解析 Markdown 文件，返回文本（保留标题 # / ## / ###）
    """
    return file_bytes.decode("utf-8")


def parse_docx(file_bytes: bytes) -> str:
    """
    解析 DOCX 文件，返回 Markdown 风格文本（# / ## / ###）
    """
    doc = Document(io.BytesIO(file_bytes))
    full_text = []
    for para in doc.paragraphs:
        style = para.style.name
        text = para.text.strip()
        if not text:
            continue
        if style.startswith("Heading 1"):
            full_text.append(f"# {text}")
        elif style.startswith("Heading 2"):
            full_text.append(f"## {text}")
        elif style.startswith("Heading 3"):
            full_text.append(f"### {text}")
        else:
            full_text.append(text)
    return "\n".join(full_text)


# 上传接口

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    上传文档（支持 .md 和 .docx），并生成 RAG chunk + embedding
    """
    file_bytes = await file.read()

    if file.filename.endswith(".md"):
        document = parse_md(file_bytes)
    elif file.filename.endswith(".docx"):
        document = parse_docx(file_bytes)
    else:
        return {"error": "Only MD and DOCX files are supported."}

    # 重新分析结构
    structure = analyze_document_structure(document)
    # 生成结构化 chunk + embeddings
    chunks, embeddings = build_trunks_and_embeddings(document)
    _save_kv_state(document, structure, chunks, embeddings)

    return {"message": f"Document '{file.filename}' uploaded and indexed successfully."}


# /result 路由用于处理表单提交并渲染 result.html

@app.post("/result", response_class=HTMLResponse)
async def result(
    request: Request,
    rewritten_query: Optional[str] = Form(None),
    answer: Optional[str] = Form(None),
    chunks: Optional[str] = Form(None),
):
    try:
        chunks_obj = json.loads(chunks) if chunks else []
    except json.JSONDecodeError:
        chunks_obj = []

    try:
        answer_obj = json.loads(answer) if answer else ""
    except json.JSONDecodeError:
        answer_obj = answer

    # Load the current document from KV for side-by-side display
    document, _, _, _ = _ensure_state()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "rewritten_query": rewritten_query or "",
            "answer": answer_obj,
            "chunks": chunks_obj,
            "document_content": document,
        },
    )
