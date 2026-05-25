# src/rag.py
import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORDB_DIR = os.path.join(BASE_DIR, "vectordb")

INDEX_PATH = os.path.join(VECTORDB_DIR, "faiss.index")
META_PATH = os.path.join(VECTORDB_DIR, "metadata.json")

TOP_K = int(os.getenv("TOP_K", 2))

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
FALLBACK_MODEL = "all-MiniLM-L6-v2"

# Compute embedding for a query
def compute_query_embedding(query: str):
    # --- Try OPENAI first ---
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and len(api_key) > 20:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )

            emb = resp.data[0].embedding
            return np.array(emb, dtype="float32")

        else:
            print("⚠️ OPENAI_API_KEY is missing or incomplete → using fallback.")

    except Exception as e:
        print("⚠️ OpenAI embedding error — using fallback:", e)

    # --- FALLBACK EMBEDDING (SentenceTransformer) ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(FALLBACK_MODEL)
    emb = model.encode([query], convert_to_numpy=True)
    return emb[0].astype("float32")

# ----------------------------------------------------
# Retrieve relevant chunks from FAISS index
# ----------------------------------------------------
def retrieve(query: str, top_k: int = TOP_K):

        # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata JSON
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Compute query embedding
    q_emb = compute_query_embedding(query)
    q_emb = q_emb.reshape(1, -1)

    # Perform FAISS search
    D, I = index.search(q_emb, top_k)

    # Collect results
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])

    return results

# ----------------------------------------------------
# Build readable context string for LLM
# ----------------------------------------------------
def format_context(chunks):
    out = []
    for c in chunks:
        out.append(
            f"Source: {c.get('source', 'unknown')} (chunk {c.get('chunk_id', 'N/A')})\n"
            f"{c.get('text', '')}\n---"
        )
    return "\n".join(out)

# ----------------------------------------------------
# Ask LLM or fallback answer builder
# ----------------------------------------------------
def ask_llm(question: str, context: str):
    system_prompt = (
        "You are UMT-SST assistant. Use the provided context to answer "
        "student questions. Cite sources in square brackets like [source:filename]. "
        "If you don't find the answer, say you don't know and recommend contacting the department."
    )

    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=600,
            temperature=0.1
        )

        return response.choices[0].message.content

    else:
        answer = "Based on the following sources:\n\n"
        for i, c in enumerate(context.split("---")[:4]):
            answer += f"{i+1}. {c.strip()}\n\n"

        answer += "\n(Install an API key to generate better LLM answers.)"
        return answer



