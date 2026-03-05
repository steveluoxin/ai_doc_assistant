from fastapi import FastAPI
from pydantic import BaseModel
from rag import (
    build_trunks_and_embeddings,
    analyze_document_structure,
    retrieve_for_summary,
    retrieve_top_k_chunks_with_score,
    embed_query,
    rewrite_query,
    answer_with_rag
)

document = ""
structure = {}
chunks = []
embeddings = []

# ========== 启动时加载文档（避免每次请求重算） ==========

def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

document = load_document("prd.txt")
structure = analyze_document_structure(document)
chunks, embeddings = build_trunks_and_embeddings(document)

# ========== FastAPI ==========

app = FastAPI(title="AI Product Document Assistant")

class AskRequest(BaseModel):
    question: str
    task: str  # summary / risk / advice

@app.post("/ask")
def ask(req: AskRequest):
    rewritten_query = rewrite_query(req.question, structure)

    if req.task == "summary":
        results = retrieve_for_summary(
            question=req.question,
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
            top_k=6
        )

    answer = answer_with_rag(
        question=req.question,
        retrieved_chunks=results,
        task=req.task
    )

    chunks_for_frontend = [
        {"content": c, "score": s, "module": m} for c, s, m in results
    ]

    return {
        "rewritten_query": rewritten_query,
        "answer": answer,
        "chunks": chunks_for_frontend
    }

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

from fastapi import UploadFile, File
from docx import Document
import io

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
        # 将 Heading 转成 Markdown
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
    global document, structure, chunks, embeddings

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

    return {"message": f"Document '{file.filename}' uploaded and indexed successfully."}