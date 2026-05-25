# src/ingest.py
"""
Ingest documents from ./data -> chunk -> compute embeddings -> save FAISS index + metadata.
Run: python src/ingest.py
"""
import os
import json
from pathlib import Path
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()

from config import Config
from utils import load_document, chunk_text

# Embedding helpers
def compute_embeddings(texts, use_openai=False, openai_model=None, fallback_model=None):
    if use_openai and openai_model:
        try:
            import openai
            openai.api_key = Config.OPENAI_API_KEY
            resp = openai.Embedding.create(model=openai_model, input=texts)
            embs = [d["embedding"] for d in resp["data"]]
            return np.array(embs, dtype="float32")
        except Exception as e:
            print(f"[ingest] OpenAI embedding failed, falling back: {e}")

    # fallback: sentence-transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(fallback_model)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")

def ingest(data_dir: str = Config.DATA_DIR, vector_dir: str = Config.VECTOR_DIR, chunk_size=800, overlap=150):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data dir not found: {p}")

    docs = []
    for f in p.iterdir():
        if f.is_file():
            print(f"[ingest] loading {f.name}")
            raw = load_document(str(f))
            chunks = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
            for i, c in enumerate(chunks):
                docs.append({
                    "source": f.name,
                    "chunk_id": i,
                    "text": c
                })

    texts = [d["text"] for d in docs]
    if not texts:
        print("[ingest] no text found in data dir")
        return

    print(f"[ingest] computing embeddings for {len(texts)} chunks...")
    vectors = compute_embeddings(
        texts,
        use_openai=bool(Config.OPENAI_API_KEY),
        openai_model=Config.EMBEDDING_MODEL,
        fallback_model=Config.FALLBACK_EMBED_MODEL
    )

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs(vector_dir, exist_ok=True)
    index_path = Path(vector_dir) / "faiss.index"
    meta_path = Path(vector_dir) / "metadata.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)

    print(f"[ingest] saved index -> {index_path}")
    print(f"[ingest] saved metadata -> {meta_path}")

if __name__ == "__main__":
    ingest()
