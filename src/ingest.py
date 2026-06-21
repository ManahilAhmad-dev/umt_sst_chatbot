# src/ingest.py
"""
Ingest documents from ./data -> chunk -> compute local embeddings -> save FAISS index + metadata.
Run: python src/ingest.py
"""
import os
import json
from pathlib import Path
import faiss
from config import Config
from utils import load_document, chunk_text

# ----------------------------------------------------
# Compute embeddings using strictly offline local CPU models
# ----------------------------------------------------
def compute_embeddings(texts, model_name=Config.EMBEDDING_MODEL):
    """
    Exclusively utilizes local SentenceTransformers for zero-cost vectorization.
    Keeps database math perfectly aligned with rag.py logic.
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")

# ----------------------------------------------------
# Main Ingestion Protocol
# ----------------------------------------------------
def ingest(data_dir: str = Config.DATA_DIR, vector_dir: str = Config.VECTOR_DIR, chunk_size=800, overlap=150):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data dir not found: {p}")

    docs = []
    for f in p.iterdir():
        # Safely targets documents and skips the faculty_images subfolder directory
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

    print(f"[ingest] computing local vector embeddings for {len(texts)} chunks...")
    vectors = compute_embeddings(texts)

    # Calculate vector space dimensions dynamically from the local transformer model
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