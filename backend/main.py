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
fatwa_collection = client.get_or_create_collection(
    name="binoria_toy_fatawa", 
    embedding_function=embed_fn
)
sunnah_collection = client.get_or_create_collection(
    name="sunnah_book", 
    embedding_function=embed_fn
)

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
# 2. Sunnah Retrieval Endpoint (Clean Single Source Mode)
# ----------------------------------------------------
@app.post("/ask/sunnah")
async def ask_sunnah(request: QueryRequest):
    user_query = request.query.strip().lower()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Keyword to Exact Section Mapping
    target_section = None
    target_section = None
    if any(k in user_query for k in ["پانی", "پینے", "پینے کا"]):
        target_section = "پانی پینے کی سنتیں"
    elif any(k in user_query for k in ["کھانا", "کھانے", "دسترخوان"]):
        target_section = "کھانے کی چند سنتیں"
    elif any(k in user_query for k in ["غسل", "نہانا"]):
        target_section = "غسل کرنے کا مسنون طریقہ"
    elif any(k in user_query for k in ["وضو", "وضو کی"]):
        target_section = "وضو کی سنتیں"
    elif any(k in user_query for k in ["سونا", "سونے", "بستر"]):
        target_section = "سونے کی سنتیں"
    elif any(k in user_query for k in ["رکوع", "رکوع میں"]):
        target_section = "رکوع کی آٹھ سنتیں"
    elif any(k in user_query for k in ["سجدہ", "سجدے", "سجدہ کی"]):
        target_section = "سجدہ کی بارہ سنتیں"
    elif any(k in user_query for k in ["قیام", "قیام میں"]):
        target_section = "قیام میں گیارہ (۱۱) سنتیں"
    elif any(k in user_query for k in ["قراءت", "قرأت"]):
        target_section = "قراءت کی سات سنتیں"
    elif any(k in user_query for k in ["قعدہ", "بیٹھنے"]):
        target_section = "قعدہ کی تیرہ سنتیں"
    elif any(k in user_query for k in ["اذان", "اقامت"]):
        target_section = "اذان و اقامت کی سنتیں"
    elif any(k in user_query for k in ["نماز کی سنتیں", "نماز کی"]):
        target_section = "نماز کی (۵۱) سنتیں"
    elif any(k in user_query for k in ["گھر میں داخل", "گھر سے نکل", "گھر"]):
        target_section = "گھر میں داخل ہونے کی دعا"
    elif any(k in user_query for k in ["مسجد", "مسجد میں"]):
        target_section = "مسجد میں داخل ہونے کی سنتیں"
    elif any(k in user_query for k in ["سفر", "مسافر"]):
        target_section = "سفر کی سنتیں"
    elif any(k in user_query for k in ["جمعہ", "جمعہ کے دن"]):
        target_section = "جمعہ کی سنتیں"
    elif any(k in user_query for k in ["لباس", "پہننے", "کپڑے"]):
        target_section = "لباس کی سنتیں"
    elif any(k in user_query for k in ["مسواک", "دانت"]):
        target_section = "مسواک کی سنتیں"
    elif any(k in user_query for k in ["استخارہ"]):
        target_section = "صلوٰۃ استخارہ"
    elif any(k in user_query for k in ["حاجت"]):
        target_section = "صلوٰۃ حاجت"

    primary_snippet = ""
    valid_sources = []

    # 1. Fetch exact section match via keyword intent
    if target_section:
        try:
            exact_res = sunnah_collection.get(
                where={"section": target_section},
                include=["documents", "metadatas"]
            )
            if exact_res and exact_res.get("documents"):
                primary_snippet = exact_res["documents"][0]
                valid_sources.append({
                    "section": target_section,
                    "book_title": "پیارے نبی ﷺ کی پیاری سنتیں",
                    "source": None,
                    "distance": 0.0000,
                    "snippet": primary_snippet
                })
        except Exception:
            pass

    # 2. Fallback to vector search if no keyword match was caught
    if not valid_sources:
        try:
            query_embeddings = embed_fn([user_query])
            search_results = sunnah_collection.query(
                query_embeddings=query_embeddings,
                n_results=1,  # Only fetch the top 1 exact match
                include=["documents", "metadatas", "distances"]
            )
            docs = search_results["documents"][0] if search_results.get("documents") else []
            metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
            dists = search_results["distances"][0] if search_results.get("distances") else []

            if docs:
                primary_snippet = docs[0]
                valid_sources.append({
                    "section": metas[0].get("section", "عام"),
                    "book_title": metas[0].get("book_title", "پیارے نبی ﷺ کی پیاری سنتیں"),
                    "source": metas[0].get("source"),
                    "distance": round(float(dists[0]), 4),
                    "snippet": docs[0]
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sunnah Search Error: {str(e)}")

    if not valid_sources:
        return {
            "answer": "مطلوبہ سنت یا دعاؤں کا مضمون اس کتاب میں نہیں ملا۔",
            "sources": []
        }

    return {
        "answer": primary_snippet,
        "sources": valid_sources  # Will contain strictly 1 exact source card
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)