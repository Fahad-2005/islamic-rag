import os
import re
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Islamic RAG MVP API")

# Enable CORS for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to local Chroma store
CHROMA_PATH = "./chroma_store"
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_or_create_collection(
    name="binoria_toy_fatawa", 
    embedding_function=embedding_func
)

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

class QueryRequest(BaseModel):
    query: str

def is_urdu_query(text: str) -> bool:
    """Detect if the query contains Urdu/Arabic script characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    is_urdu = is_urdu_query(user_query)

    # 1. Retrieve top records WITH distance scores
    # Cosine distance ranges from 0.0 (identical) to 2.0 (opposite).
    search_results = collection.query(
        query_texts=[user_query],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    # 2. Filter out irrelevant records using similarity cutoff
    # Cosine distance <= 0.72 is generally relevant for multilingual-MiniLM
    # 2. Filter out irrelevant records using similarity cutoff
    # 0.62 is optimal for multilingual-MiniLM on Urdu-to-Urdu legal texts
    DISTANCE_THRESHOLD = 0.62
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

    # Fallback message
    fallback_msg = (
        "مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔"
        if is_urdu
        else "The ruling is not available in the verified records."
    )

    # If no records meet the strict cutoff, reject immediately
    if not filtered_snippets:
        return {
            "answer": fallback_msg,
            "sources": []
        }

    # 3. Formulate strict grounding context and prompts
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
            temperature=0.0  # Set to 0.0 for deterministic, strictly grounded output
        )
        generated_answer = response.choices[0].message.content.strip()

        # Guard against blank LLM output or negative statements
        negative_indicators = [
            "ruling is not available",
            "not available in the verified records",
            "مطلوبہ مسئلہ فراہم کردہ فتاویٰ",
            "دستیاب نہیں ہے",
            "معلومات دستیاب نہیں"
        ]

        # If empty OR flagged as unavailable, clean out the sources and show fallback
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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)