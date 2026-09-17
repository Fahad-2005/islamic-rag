import os
import re
import requests
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Islamic RAG MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChromaDB without loading any heavy local models
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name="binoria_toy_fatawa")

# Initialize Groq
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_query_embedding(text: str) -> list:
    """Embed single query string via Hugging Face API (~50KB memory overhead)."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, 
            detail=f"Hugging Face API error ({response.status_code}): {response.text}"
        )
    return response.json()

class QueryRequest(BaseModel):
    query: str

def is_urdu_query(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Islamic RAG API"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    is_urdu = is_urdu_query(user_query)

    # 1. Fetch remote embedding
    query_vector = get_query_embedding(user_query)

    # 2. Query Chroma vector store using raw embeddings
    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    # 3. Filter by distance threshold
    DISTANCE_THRESHOLD = 0.72
    valid_sources = []
    filtered_snippets = []

    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= DISTANCE_THRESHOLD:
            filtered_snippets.append(doc)
            valid_sources.append({
                "fatwa_number": meta.get("fatwa_number"),
                "category": meta.get("category"),
                "sub_category": meta.get("sub_category"),
                "url": meta.get("url"),
                "snippet": doc[:350] + "..." if len(doc) > 350 else doc
            })

    fallback_msg = (
        "مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔"
        if is_urdu
        else "The ruling is not available in the verified records."
    )

    if not filtered_snippets:
        return {
            "answer": fallback_msg,
            "sources": []
        }

    # 4. Prompt construction
    context_text = "\n\n---\n\n".join(filtered_snippets)

    if is_urdu:
        system_prompt = (
            "آپ صرف جامعہ بنوریہ کے مستند فتاویٰ کے لیے ایک سخت اور غیر جانبدار معاون ہیں۔\n"
            "اصول:\n"
            "1. صرف اور صرف نیچے دیے گئے 'سیاق و سباق' (Context) کی بنیاد پر جواب دیں۔\n"
            "2. اپنی ذاتی معلومات سے کوئی بات شامل نہ کریں۔\n"
            "3. جواب لازماً اردو میں دیں۔\n"
            "4. اگر سیاق و سباق میں سوال کا واضح جواب موجود نہ ہو، تو صرف یہی لکھیں:\n"
            "'مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔'"
        )
    else:
        system_prompt = (
            "You are a strict Islamic QA assistant grounded EXCLUSIVELY in verified Jamia Binoria fatawa.\n"
            "RULES:\n"
            "1. Answer strictly and solely using the provided Context below. Do NOT extrapolate or use external knowledge.\n"
            "2. LANGUAGE REQUIREMENT: The user asked in English. You MUST respond completely in English.\n"
            "3. If the provided Context does not contain the specific legal ruling for the query, output EXACTLY:\n"
            "'The ruling is not available in the verified records.' and nothing else."
        )

    user_prompt = f"Context:\n{context_text}\n\nUser Question:\n{user_query}\n\nAnswer:"

    try:
        if not groq_client:
            return {
                "answer": "[API Key Missing] " + fallback_msg,
                "sources": valid_sources
            }

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        generated_answer = response.choices[0].message.content.strip()

        negative_indicators = [
            "ruling is not available",
            "not available in the verified records",
            "مطلوبہ مسئلہ فراہم کردہ فتاویٰ",
            "دستیاب نہیں ہے",
            "معلومات دستیاب نہیں"
        ]

        if not generated_answer or any(indicator in generated_answer.lower() for indicator in negative_indicators):
            return {
                "answer": fallback_msg,
                "sources": []
            }

        return {
            "answer": generated_answer,
            "sources": valid_sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)