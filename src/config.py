# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    # Intercepts standard cloud calls and transparently proxies them to Groq's high-speed LPUs
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    
    # Swapped defaults to high-performance open weights and offline models to bypass cloud fees
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    VECTOR_DIR = os.getenv("VCTOR_DIR", str(ROOT / "vectordb"))
    INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
    META_PATH = os.path.join(VECTOR_DIR, "metadata.json")
    DATA_DIR = os.getenv("DATA_DIR", str(ROOT / "data"))
    TOP_K = int(os.getenv("TOP_K", 4))
    # Fallback local sentence-transformers model
    FALLBACK_EMBED_MODEL = os.getenv("FALLBACK_EMBED_MODEL", "all-MiniLM-L6-v2")

    # -----------------------------------------------------------------
    # New Face Recognition Core Paths (Preserved Exactly)
    # -----------------------------------------------------------------
    # Points to data/faculty_images
    FACULTY_IMAGES_DIR = os.path.join(DATA_DIR, "faculty_images")
    # Points to data/faculty_images/faculty_details.json as seen in your explorer
    FACULTY_JSON_PATH = os.path.join(FACULTY_IMAGES_DIR, "faculty_details.json")

# Failsafe directory generation to prevent startup crashes
os.makedirs(Config.FACULTY_IMAGES_DIR, exist_ok=True)