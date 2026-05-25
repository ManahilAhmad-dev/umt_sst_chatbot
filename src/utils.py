# src/utils.py
import os
import re
from typing import List
import pdfplumber
from docx import Document
from tqdm import tqdm

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)

def load_docx(path: str) -> str:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def load_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in [".docx", ".doc"]:
        return load_docx(path)
    else:
        return load_text_file(path)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Naive chunking: split by sentences until chunk_size reached."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
