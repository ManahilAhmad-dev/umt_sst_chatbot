# src/rag.py
import json
import faiss
from config import Config

# ----------------------------------------------------
# Faculty Metadata Pre-Retrieval Scanner
# ----------------------------------------------------
COUNSELLING_KEYWORDS = {
    "counselling", "counseling", "consulting", "office hours",
    "availability", "when is", "consultation", "meet the teacher",
    "teacher available", "professor available"
}

def search_faculty_registry(query: str):
    """
    Scans the local JSON registry for:
    - Name-specific queries: full profile for any faculty whose name appears in the query.
    - Counselling-specific queries: all faculty who have known counselling hours set.
    """
    try:
        with open(Config.FACULTY_JSON_PATH, 'r', encoding='utf-8') as f:
            faculty_db = json.load(f)

        query_lower = query.lower()
        is_counselling_query = any(kw in query_lower for kw in COUNSELLING_KEYWORDS)
        matches = []

        for key, details in faculty_db.items():
            name = details.get("name", "").lower()
            name_parts = [p for p in name.replace("dr.", "").replace("mr.", "").replace("ms.", "").split() if len(p) > 2]
            name_in_query = name and (name in query_lower or any(p in query_lower for p in name_parts))

            if name_in_query:
                profile = f"Faculty Name: {details.get('name')}\n"
                profile += f"Department: {details.get('department', 'N/A')}\n"
                profile += f"Email: {details.get('email', 'N/A')}\n"
                profile += f"Phone: {details.get('phone', 'N/A')}\n"
                profile += f"Office: {details.get('office_location', 'N/A')}\n"
                profile += f"Counselling Hours: {details.get('office_timings', 'N/A')}\n"
                matches.append(profile)
            elif is_counselling_query:
                timings = details.get("office_timings", "N/A")
                if timings and timings != "N/A":
                    profile = f"Faculty Name: {details.get('name')}\n"
                    profile += f"Counselling Hours: {timings}\n"
                    profile += f"Office: {details.get('office_location', 'N/A')}\n"
                    matches.append(profile)

        if matches:
            return "--- UMT-SST FACULTY REGISTRY DATA ---\n" + "\n\n".join(matches)
        return ""

    except Exception as e:
        print(f"Faculty lookup bypassed: {e}")
        return ""

# ----------------------------------------------------
# Compute embedding using strictly offline, local CPU models
# ----------------------------------------------------
def compute_query_embedding(query: str):
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    emb = model.encode([query], convert_to_numpy=True)
    return emb[0].astype("float32")

# ----------------------------------------------------
# Retrieve relevant chunks from FAISS index & JSON Registry
# ----------------------------------------------------
def retrieve(query: str, top_k: int = Config.TOP_K):
    # Load FAISS index and PDF metadata
    index = faiss.read_index(Config.INDEX_PATH)
    
    with open(Config.META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Compute query embedding mathematically using the local CPU
    q_emb = compute_query_embedding(query)
    q_emb = q_emb.reshape(1, -1)

    # Perform FAISS vector similarity search
    # Renamed ambiguous variables to comply with PEP 8 standards
    _, indices = index.search(q_emb, top_k)

    # Collect PDF results
    results = []
    for idx in indices[0]:
        if idx < len(meta):
            results.append(meta[idx])

    # Check the JSON database for Teacher Info
    faculty_context = search_faculty_registry(query)
    
    if faculty_context:
        results.insert(0, {
            "source": "faculty_details.json",
            "chunk_id": "Biometric Registry",
            "text": faculty_context
        })

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
# Ask LLM via Groq LPU Routing
# ----------------------------------------------------
def ask_llm(question: str, context: str):
    system_prompt = (
        "You are the UMT-SST assistant. Use the provided context to answer "
        "student questions. Cite sources in square brackets like [source:filename]. "
        "If you don't find the answer, say you don't know and recommend contacting the department."
    )

    from openai import OpenAI
    
    client = OpenAI(
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL
    )

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
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