import os
import numpy as np

# 判断是否使用 API embedding
USE_API_EMBEDDING = os.getenv("USE_API_EMBEDDING", "false").lower() == "true"

if USE_API_EMBEDDING:
    try:
        from llm import get_embedding
    except ImportError:
        raise ImportError("USE_API_EMBEDDING=True 但是 llm.get_embedding 未找到，请确保 llm.py 配置正确")

    def embed(text: str) -> np.ndarray:
        """
        使用 API 获取 embedding 并返回 numpy array。
        """
        try:
            return np.array(get_embedding(text))
        except Exception as e:
            raise RuntimeError(f"API embedding 调用失败: {e}")

else:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("USE_API_EMBEDDING=False 但是未安装 sentence-transformers，请执行 'pip install sentence-transformers torch'")

    # 初始化本地模型，只初始化一次
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> np.ndarray:
        """
        使用本地 sentence-transformers 生成 embedding。
        """
        try:
            return model.encode(text)
        except Exception as e:
            raise RuntimeError(f"本地 embedding 生成失败: {e}")