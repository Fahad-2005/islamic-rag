import os
import re
from pathlib import Path

# Fix for Ubuntu SQLite version requirement in ChromaDB
try:
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Islamic RAG Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BACKEND_DIR / "chroma_store")

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Load multilingual sentence transformer model
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Fetch existing collections
fatwa_collection = client.get_collection(name="binoria_toy_fatawa")
sunnah_collection = client.get_collection(name="sunnah_book")

class QueryRequest(BaseModel):
    query: str

def is_urdu_query(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def extract_clean_answer(raw_text: str) -> str:
    """
    Cleans raw fatwa snippet for primary display by stripping repetitive 
    headers ('سوال:' / 'جواب:') and keeping the main direct ruling.
    """
    cleaned = raw_text.strip()
    
    # 1. Strip repetitive question block if embedded in text
    if "جواب:" in cleaned:
        cleaned = cleaned.split("جواب:", 1)[-1].strip()
    elif "سوال:" in cleaned and "جواب" in cleaned:
        cleaned = cleaned.split("جواب", 1)[-1].strip()

    # Remove any leading colon or spaces leftover from 'جواب:'
    cleaned = re.sub(r'^[:\s]+', '', cleaned)

    # 2. Extract first 2 paragraphs for a clean direct summary
    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    
    summary_paragraphs = []
    for p in paragraphs:
        if any(marker in p for marker in ["فتاویٰ", "کتاب الوقف", "رد المحتار", "البحر الرائق"]):
            break
        summary_paragraphs.append(p)
        
    if summary_paragraphs:
        return "\n\n".join(summary_paragraphs[:2])
    
    return paragraphs[0] if paragraphs else raw_text

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Islamic RAG Retrieval API",
        "fatwa_collection_count": fatwa_collection.count(),
        "sunnah_collection_count": sunnah_collection.count()
    }

# ----------------------------------------------------
# 1. Fatwa Retrieval Endpoint
# ----------------------------------------------------
@app.post("/ask/fatwa")
@app.post("/ask")  # Fallback for backward compatibility
async def ask_fatwa(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        query_embeddings = embed_fn([user_query])
        search_results = fatwa_collection.query(
            query_embeddings=query_embeddings,
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fatwa Search Error: {str(e)}")

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    is_urdu = is_urdu_query(user_query)
    DIST_THRESHOLD = 0.65 if is_urdu else 0.82

    fallback_msg = (
        "مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔"
        if is_urdu
        else "The ruling is not available in the verified records."
    )

    valid_sources = []
    if docs and dists and metas:
        for doc, meta, dist in zip(docs, metas, dists):
            if dist <= DIST_THRESHOLD:
                reference_data = {
                    "fatwa_number": meta.get("fatwa_number") or meta.get("id"),
                    "title": meta.get("title") or meta.get("question"),
                    "category": meta.get("category"),
                    "sub_category": meta.get("sub_category"),
                    "url": meta.get("url"),
                    "distance": round(float(dist), 4),
                    "snippet": doc
                }
                cleaned_reference = {k: v for k, v in reference_data.items() if v is not None}
                valid_sources.append(cleaned_reference)

    if not valid_sources:
        return {
            "answer": fallback_msg,
            "sources": []
        }

    primary_clean_answer = extract_clean_answer(valid_sources[0]["snippet"])

    return {
        "answer": primary_clean_answer,
        "sources": valid_sources
    }

# ----------------------------------------------------
# 2. Sunnah Retrieval Endpoint
# ----------------------------------------------------
@app.post("/ask/sunnah")
async def ask_sunnah(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        query_embeddings = embed_fn([user_query])
        search_results = sunnah_collection.query(
            query_embeddings=query_embeddings,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunnah Search Error: {str(e)}")

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    is_urdu = is_urdu_query(user_query)
    DIST_THRESHOLD = 0.75 if is_urdu else 0.85

    fallback_msg = (
        "مطلوبہ سنت یا دعاؤں کا مضمون اس کتاب میں نہیں ملا۔"
        if is_urdu
        else "The queried Sunnah or Dua was not found in the record."
    )

    valid_sources = []
    if docs and dists and metas:
        for doc, meta, dist in zip(docs, metas, dists):
            if dist <= DIST_THRESHOLD:
                reference_data = {
                    "section": meta.get("section", "عام"),
                    "book_title": meta.get("book_title", "پیارے نبی ﷺ کی پیاری سنتیں"),
                    "source": meta.get("source"),
                    "distance": round(float(dist), 4),
                    "snippet": doc
                }
                valid_sources.append(reference_data)

    if not valid_sources:
        return {
            "answer": fallback_msg,
            "sources": []
        }

    return {
        "answer": valid_sources[0]["snippet"],
        "sources": valid_sources
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)