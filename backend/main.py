import os
import re
from pathlib import Path
import requests
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Islamic Fatwa Semantic Retrieval API")

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
collection = client.get_collection(name="binoria_toy_fatawa")

# Remote HF API endpoint — 0 MB local RAM usage
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_query_embedding(text: str):
    headers = {}
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": [text], "options": {"wait_for_model": True}},
        timeout=10
    )
    
    if response.status_code != 200:
        raise Exception(f"HF API Error {response.status_code}: {response.text}")
    
    res = response.json()
    # Format ensure 2D list for ChromaDB
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], float):
        return [res]
    return res

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
        query_embeddings = get_query_embedding(user_query)

        search_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    DIST_THRESHOLD = 0.45
    fallback_msg = (
        "مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔"
        if is_urdu_query(user_query)
        else "The ruling is not available in the verified records."
    )

    if not docs or (dists and dists[0] > DIST_THRESHOLD):
        return {
            "answer": fallback_msg,
            "sources": []
        }

    valid_sources = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= DIST_THRESHOLD:
            valid_sources.append({
                "fatwa_number": meta.get("fatwa_number"),
                "category": meta.get("category"),
                "sub_category": meta.get("sub_category"),
                "url": meta.get("url"),
                "distance": round(float(dist), 4),
                "snippet": doc
            })

    return {
        "answer": docs[0],
        "sources": valid_sources
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)