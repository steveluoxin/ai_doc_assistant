import os
import numpy as np
import requests

# By default we now prefer API embeddings. This keeps the Vercel serverless
# bundle small (no torch / sentence-transformers). For local development you
# can still opt into local embeddings by setting USE_API_EMBEDDING=false.
USE_API_EMBEDDING = os.getenv("USE_API_EMBEDDING", "true").lower() == "true"

# SiliconFlow is used for embeddings because DeepSeek does not expose a
# public embedding endpoint. SiliconFlow offers an OpenAI-compatible API and
# a generous free tier (e.g. BAAI/bge-m3).
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")


def _siliconflow_embedding(text: str) -> list[float]:
    """Call SiliconFlow's OpenAI-compatible embedding endpoint."""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("SILICONFLOW_API_KEY is not set")

    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": SILICONFLOW_EMBEDDING_MODEL, "input": text},
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


if USE_API_EMBEDDING:

    def embed(text: str) -> np.ndarray:
        """
        Use SiliconFlow's embedding API and return a numpy array.
        """
        return np.array(_siliconflow_embedding(text), dtype=np.float32)

else:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "USE_API_EMBEDDING=false but sentence-transformers is not installed. "
            "Run: pip install sentence-transformers torch"
        )

    _model = None

    def embed(text: str) -> np.ndarray:
        """
        Use a local sentence-transformers model to generate embeddings.
        """
        global _model
        if _model is None:
            print("Loading local embedding model...")
            _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
            print("Local embedding model loaded.")
        return _model.encode(text)
