import os
import re
from pathlib import Path

# Fix for Ubuntu SQLite version requirement in ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Islamic Fatwa Semantic Retrieval API")

# Allow requests from your teammate's frontend
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

# Best model for Urdu/Arabic/English cross-lingual semantic search
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_collection(
    name="binoria_toy_fatawa",
    embedding_function=embed_fn
)

class QueryRequest(BaseModel):
    query: str

def is_urdu_query(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Islamic Fatwa Semantic Retrieval API",
        "collection_count": collection.count()
    }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        search_results = collection.query(
            query_texts=[user_query],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

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
            # Only include sources that satisfy similarity distance threshold
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

    return {
        "answer": valid_sources[0]["snippet"],
        "sources": valid_sources
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)