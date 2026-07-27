import os
import numpy as np

# By default we now prefer API embeddings. This keeps the Vercel serverless
# bundle small (no torch / sentence-transformers). For local development you
# can still opt into local embeddings by setting USE_API_EMBEDDING=false.
USE_API_EMBEDDING = os.getenv("USE_API_EMBEDDING", "true").lower() == "true"

if USE_API_EMBEDDING:
    try:
        from llm import get_embedding
    except ImportError:
        raise ImportError(
            "USE_API_EMBEDDING=true but llm.get_embedding is not available. "
            "Ensure llm.py is present and DEEPSEEK_API_KEY is configured."
        )

    def embed(text: str) -> np.ndarray:
        """
        Use the DeepSeek embedding API and return a numpy array.
        """
        return np.array(get_embedding(text), dtype=np.float32)

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
