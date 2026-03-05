import os
import numpy as np

# 判断是否使用 API embedding
USE_API_EMBEDDING = os.getenv("USE_API_EMBEDDING", "false").lower() == "true"

if USE_API_EMBEDDING:
    # 使用 DeepSeek API embedding
    from llm import get_embedding

    def embed(text: str) -> np.ndarray:
        """调用 API 获取 embedding 并返回 numpy array"""
        return np.array(get_embedding(text))

else:
    # 本地 embedding
    from sentence_transformers import SentenceTransformer

    # 初始化本地模型
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> np.ndarray:
        """本地生成 embedding"""
        return model.encode(text)