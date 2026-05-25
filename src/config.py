# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VECTOR_DIR = os.getenv("VCTOR_DIR", str(ROOT / "vectordb"))
    INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
    META_PATH = os.path.join(VECTOR_DIR, "metadata.json")
    DATA_DIR = os.getenv("DATA_DIR", str(ROOT / "data"))
    TOP_K = int(os.getenv("TOP_K", 4))
    # Fallback local sentence-transformers model
    FALLBACK_EMBED_MODEL = os.getenv("FALLBACK_EMBED_MODEL", "all-MiniLM-L6-v2")
