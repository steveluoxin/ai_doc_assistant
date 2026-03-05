from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import split_text
from llm import call_llm
from prompts import build_prompt

# 加载 embedding 模型
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_trunks_and_embeddings(
    document: str, max_chars: int = 1000
) -> Tuple[List[str], np.ndarray]:
    """
    结构感知的文本分块（chunk）
    - 利用 Markdown 标题 (# / ## / ###) 做分块边界
    - 每个 chunk 都携带最近的一级/二级标题上下文
    - 如果 chunk 太长，会按 max_chars 再拆分
    """

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
            # 一级标题
            flush_buffer()
            h1_context = line[2:].strip()
            h2_context = ""
        elif line.startswith("## "):
            # 二级标题
            flush_buffer()
            h2_context = line[3:].strip()
        elif line.startswith("### "):
            # 三级标题也算正文
            buffer.append(line)
        else:
            buffer.append(line)
            # 超长拆分
            current_len = sum(len(l) for l in buffer)
            if current_len >= max_chars:
                flush_buffer()

    flush_buffer()

    embeddings = _embedding_model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=False
    )

    return chunks, embeddings


def embed_query(question: str) -> np.ndarray:
    """
    输入: question 文本
    输出: question 的向量表示

    注意：
    - 返回 shape: (embedding_dim,)
    """
    embedding = _embedding_model.encode(
        question, convert_to_numpy=True, show_progress_bar=False
    )
    return embedding


def retrieve_top_k_chunks(
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> List[str]:
    """
    从所有文本 chunk 中，检索与 query 最相关的 top_k 个 chunk
    参数:
      - chunks: List[str]                文本分块
      - chunk_embeddings: np.ndarray
        shape: (N, embedding_dim)
      - query_embedding: np.ndarray
        shape: (embedding_dim,)
      - top_k: int                       返回最相关的 top_k 个文本块

    返回:
      - top_k_chunks: List[str]         最相关的 top_k 个文本块
    """
    # 计算相似度（点积）
    similarities = np.dot(chunk_embeddings, query_embedding)

    # 获取 top_k 索引
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # 返回对应的文本块
    top_k_chunks = [chunks[i] for i in top_k_indices]
    return top_k_chunks


def retrieve_top_k_chunks_with_score(
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 5,
    min_score: float = 0.25,
    relative_threshold: float = 0.75,
) -> List[Tuple[str, float]]:
    """
    从所有文本 chunk 中，检索与 query 最相关的 top_k 个 chunk（带相似度分数）

    参数:
      - chunks: List[str]
      - chunk_embeddings: np.ndarray
        shape: (N, embedding_dim)
      - query_embedding: np.ndarray
        shape: (embedding_dim,)
      - top_k: int
        最多返回多少个 chunk
      - min_score: float
        相似度阈值，低于该值的 chunk 会被过滤

    返回:
      - List[Tuple[str, float]]
        [(chunk_text, similarity_score), ...]
        按 score 从高到低排序
    """
    # 1. 计算相似度（点积）
    # shape: (N,)
    similarities = np.dot(chunk_embeddings, query_embedding)

    # 2. 按相似度从高到底排序索引
    sorted_indices = np.argsort(similarities)[::-1]
    results: List[Tuple[str, float]] = []

    if len(sorted_indices) == 0:
        return results

    top_score = similarities[sorted_indices[0]]

    # 3. 依次取 top_k 个，并做阈值过滤
    for idx in sorted_indices:
        score = similarities[idx]
        if score < min_score:
            break  # 后面的只会更小，直接终止
        if score < top_score * relative_threshold:
            break  # 相对阈值过滤

        results.append((chunks[idx], float(score)))

        if len(results) >= top_k:
            break

    return results


def answer_with_rag(
    question: str, retrieved_chunks: List[Tuple[str, float]], task: str
) -> dict:
    """
    基于检索到的文本块，调用 LLM 生成最终答案

    参数:
      - question: str
        用户问题
      - retrieved_chunks: List[Tuple[str, float]]
        [(chunk_text, similarity_score), ...]
      - task: str
        任务类型，如 "summary", "risk", "advice"

    返回:
      - result: Dict
        LLM 返回的结果，结构依据具体任务而定
    """

    # 1️⃣ 如果没有任何相关 chunk，直接兜底
    if not retrieved_chunks:
        if task == "summary":
            return {"points": []}
        elif task == "risk":
            return {"risks": []}
        elif task == "advice":
            return {"advices": []}
        else:
            return {}

    # 2️⃣ 拼接 context（只取文本，不暴露 score 给模型）
    context = "\n\n".join(
        f"【文档片段 {i+1}】 模块: {module} | 相似度: {score:.2f}\n{chunk}"
        for i, (chunk, score, module) in enumerate(retrieved_chunks)
    )

    # 3️⃣ 构造 prompt（严格约束）
    prompt = build_prompt(task=task, document=context, question=question)

    # 4️⃣ 调用 LLM
    raw_output = call_llm(prompt, task)

    return raw_output

def rewrite_query(question: str, structure: dict) -> str:
    primary = structure.get("primary_modules", [])
    # 构造 prompt 强制展开（之前强化过）
    prompt = f"""
    你是一个查询改写助手。
    该产品包含以下一级模块：
    {primary}

    用户问题：
    {question}

    请严格按照要求改写：
    - 列出所有一级模块
    - 将抽象问题“核心功能”拆解为“各模块分别提供的核心功能”
    - 生成更长的句子
    - 不允许与原问题相同
    - 只输出改写后的问题
    """

    # 直接拿返回文本
    rewritten = call_llm(prompt, task="rewrite")
    return rewritten.strip()

def analyze_document_structure(document: str) -> dict:
    """
    使用 LLM 分析单个产品文档的整体结构。
    只在系统初始化时调用一次。

    返回：
    {
        "primary_modules": [...],
        "secondary_modules": [...]
    }
    """

    prompt = f"""
    你是一名产品架构分析专家。

    下面是一份产品需求文档，请你识别该产品的：

    1. 一级模块 / 子系统（例如：管理端、考试端、监考端）
    2. 二级功能模块（例如：考试管理、成绩管理、权限管理）

    要求：
    - 只输出 JSON
    - 不要解释
    - 不要遗漏明显的模块
    - 不要生成文档中不存在的结构

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
    question: str,
    structure: dict,
    chunks,
    embeddings,
    per_module_k=3
):
    primary_modules = structure.get("primary_modules", [])
    all_results = []

    for module in primary_modules:
        sub_query = f"{module} 的核心功能是什么？"
        query_emb = embed_query(sub_query)

        results = retrieve_top_k_chunks_with_score(
            chunks,
            embeddings,
            query_embedding=query_emb,
            top_k=per_module_k,
            min_score=0.2,
            relative_threshold=0.7
        )

        # 这里把 module 信息加进去
        for chunk_text, score in results:
            all_results.append((chunk_text, score, module))

    # 去重（防止重复 chunk）
    unique = {}
    for chunk_text, score, module in all_results:
        if chunk_text not in unique or score > unique[chunk_text]['score']:
            unique[chunk_text] = {"score": score, "module": module}

    # 转回 list
    final_results = [
        (chunk_text, info["score"], info["module"]) for chunk_text, info in unique.items()
    ]

    return final_results
