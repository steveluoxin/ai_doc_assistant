from typing import List, Tuple
import numpy as np

from embedding import embed
from utils import split_text
from llm import call_llm
from prompts import build_prompt


def build_trunks_and_embeddings(
    document: str, max_chars: int = 1000
) -> Tuple[List[str], np.ndarray]:
    lines = [line.strip() for line in document.splitlines() if line.strip()]

    chunks: List[str] = []
    h1_context = ""
    h2_context = ""
    buffer: List[str] = []

    def flush_buffer():
        if not buffer:
            return
        content = "\n".join(buffer)
        context_lines = []
        if h1_context:
            context_lines.append(f"【一级标题】{h1_context}")
        if h2_context:
            context_lines.append(f"【二级标题】{h2_context}")
        chunk_text = "\n".join(context_lines + ["【内容】", content])
        chunks.append(chunk_text)
        buffer.clear()

    for line in lines:
        if line.startswith("# "):
            flush_buffer()
            h1_context = line[2:].strip()
            h2_context = ""
        elif line.startswith("## "):
            flush_buffer()
            h2_context = line[3:].strip()
        elif line.startswith("### "):
            buffer.append(line)
        else:
            buffer.append(line)
            current_len = sum(len(l) for l in buffer)
            if current_len >= max_chars:
                flush_buffer()

    flush_buffer()

    # embeddings = np.array([embed(chunk) for chunk in chunks])
    embeddings = np.vstack([embed(chunk) for chunk in chunks])
    return chunks, embeddings


def embed_query(question: str) -> np.ndarray:
    return embed(question)


def retrieve_top_k_chunks(
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> List[str]:
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_k_indices]


def retrieve_top_k_chunks_with_score(
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.25,
    relative_threshold: float = 0.75,
    task: str = "summary",
) -> List[Tuple[str, float, str]]:
    similarities = np.dot(chunk_embeddings, query_embedding)
    sorted_indices = np.argsort(similarities)[::-1]
    results: List[Tuple[str, float, str]] = []

    if len(sorted_indices) == 0:
        return results

    top_score = similarities[sorted_indices[0]]

    # 针对 risk/advice 设置不同的过滤阈值
    if task == "risk":
        min_score = 0
        relative_threshold = 0
    elif task == "advice":
        min_score = 0.1
        relative_threshold = 0.35

    for idx in sorted_indices:
        score = similarities[idx]
        if score < min_score:
            break
        if score < top_score * relative_threshold:
            break

        chunk_text = chunks[idx]
        module = "unknown"
        for line in chunk_text.splitlines():
            if "【一级标题】" in line:
                module = line.replace("【一级标题】", "").strip()
                break
            if "【二级标题】" in line:
                module = line.replace("【二级标题】", "").strip()
        results.append((chunk_text, float(score), module))

        if len(results) >= top_k:
            break

    return results


def answer_with_rag(
    question: str, retrieved_chunks: List[Tuple[str, float, str]], task: str
) -> dict:
    if not retrieved_chunks:
        if task == "summary":
            return {"points": []}
        elif task == "risk":
            return {"risks": []}
        elif task == "advice":
            return {"advices": []}
        else:
            return {}

    context = "\n\n".join(
        f"【文档片段 {i+1}】 模块: {module} | 相似度: {score:.2f}\n{chunk}"
        for i, (chunk, score, module) in enumerate(retrieved_chunks)
    )

    prompt = build_prompt(task=task, document=context, question=question)
    raw_output = call_llm(prompt, task)
    return raw_output


def rewrite_query(question: str, structure: dict, task: str) -> str:
    primary = structure.get("primary_modules", [])
    if task == "summary":
        instruction = "针对每个一级模块，分别询问其核心功能。"
    elif task == "risk":
        instruction = "针对每个一级模块，分别询问其可能存在的风险点。"
    elif task == "advice":
        instruction = "针对每个一级模块，分别询问其可以改进的建议。"
    else:
        instruction = "针对每个一级模块，生成有助于检索相关内容的具体查询。"

    prompt = f"""
你是查询改写助手。
该产品包含以下一级模块：
{primary}

用户问题：
{question}

请严格按照要求改写：
- {instruction}
- 遍历所有一级模块
- 生成具体长句
- 不允许与原问题相同
- 只输出改写后的问题
"""

    rewritten = call_llm(prompt, task="rewrite")
    return rewritten.strip()


def analyze_document_structure(document: str) -> dict:
    prompt = f"""
你是一名产品架构分析专家。
下面是一份产品需求文档，请识别：
1. 一级模块 / 子系统
2. 二级功能模块
要求：
- 只输出 JSON
- 不解释
- 不遗漏明显模块
文档内容：
----------------
{document}
----------------
严格返回格式：
{{
"primary_modules": ["..."],
"secondary_modules": ["..."]
}}
"""

    result = call_llm(prompt, task="advice")
    if isinstance(result, dict):
        return result
    return {"primary_modules": [], "secondary_modules": []}


def retrieve_for_summary(
    question: str, structure: dict, chunks, embeddings, per_module_k=3, task="summary"
):
    primary_modules = structure.get("primary_modules", [])
    all_results = []

    for module in primary_modules:
        if task == "summary":
            sub_query = f"{module} 的核心功能是什么？"
        elif task == "risk":
            sub_query = f"{module} 的可能风险点有哪些？"
        elif task == "advice":
            sub_query = f"{module} 的改进建议有哪些？"
        else:
            sub_query = f"{module} 的相关信息有哪些？"

        query_emb = embed_query(sub_query)

        results = retrieve_top_k_chunks_with_score(
            chunks,
            embeddings,
            query_embedding=query_emb,
            top_k=per_module_k,
            min_score=0.2,
            relative_threshold=0.7,
            task=task
        )

        for chunk_text, score, _ in results:
            all_results.append((chunk_text, score, module))

    unique = {}
    for chunk_text, score, module in all_results:
        if chunk_text not in unique or score > unique[chunk_text]["score"]:
            unique[chunk_text] = {"score": score, "module": module}

    final_results = [
        (chunk_text, info["score"], info["module"])
        for chunk_text, info in unique.items()
    ]

    return final_results
