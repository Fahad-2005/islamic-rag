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
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from translation import (
    DEFAULT_GROQ_MODEL,
    GROQ_CHAT_COMPLETIONS_URL,
    TranslationConfigurationError,
    TranslationError,
    translate_to_english,
)

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

# Load BAAI/bge-m3 embedding model to match ingestion
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3"
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


def safe_translate_to_english(text: str) -> tuple[str | None, str | None]:
    """Translates Urdu text to English using translation.py without breaking the API."""
    if not text or not text.strip():
        return None, None
    try:
        return translate_to_english(text), None
    except (TranslationConfigurationError, TranslationError) as exc:
        return None, str(exc)


def translate_english_query_to_urdu(query: str) -> str:
    """Translates an English query into Urdu to improve retrieval against Urdu collections."""
    if is_urdu_query(query):
        return query

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query

    try:
        response = requests.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's Islamic or fiqh question into clear Urdu "
                            "for semantic search in an Urdu Fatawa database. "
                            "Output only the Urdu translation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )
        response.raise_for_status()
        translated = response.json()["choices"][0]["message"]["content"]
        if isinstance(translated, str) and translated.strip():
            return translated.strip()
    except Exception:
        pass

    return query


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

    is_urdu = is_urdu_query(user_query)
    retrieval_query = user_query if is_urdu else translate_english_query_to_urdu(user_query)

    # If the English query was translated to Urdu, evaluate distance as Urdu query
    effective_urdu_search = is_urdu or is_urdu_query(retrieval_query)
    DIST_THRESHOLD = 0.45 if effective_urdu_search else 0.70

    try:
        query_embeddings = embed_fn([retrieval_query])
        search_results = fatwa_collection.query(
            query_embeddings=query_embeddings,
            n_results=1,  # STRICTLY 1 to eliminate the wrong second reference
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fatwa Search Error: {str(e)}")

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
    dists = search_results["distances"][0] if search_results.get("distances") else []

    fallback_msg = (
        "مطلوبہ مسئلہ فراہم کردہ فتاویٰ کے ریکارڈ میں دستیاب نہیں ہے۔"
        if is_urdu
        else "The ruling is not available in the verified records."
    )

    valid_sources = []
    if docs and dists and metas:
        doc = docs[0]
        meta = metas[0]
        dist = dists[0]

        if dist <= DIST_THRESHOLD:
            retrieved_question = meta.get("question", "")

            reference_data = {
                "fatwa_number": meta.get("fatwa_number") or meta.get("id"),
                "title": retrieved_question,
                "category": meta.get("category"),
                "sub_category": meta.get("sub_category"),
                "url": meta.get("url"),
                "distance": round(float(dist), 4),
                "snippet": doc
            }
            cleaned_reference = {k: v for k, v in reference_data.items() if v is not None}
            valid_sources.append(cleaned_reference)

    if not valid_sources:
        english_fallback, warning = safe_translate_to_english(fallback_msg)
        payload = {
            "answer": fallback_msg,
            "english_translation": english_fallback or fallback_msg,
            "sources": []
        }
        if warning:
            payload["translation_warning"] = warning
        return payload

    primary_clean_answer = extract_clean_answer(valid_sources[0]["snippet"])
    english_answer, warning = safe_translate_to_english(primary_clean_answer)

    payload = {
        "answer": primary_clean_answer,
        "english_translation": english_answer,
        "sources": valid_sources
    }
    if warning:
        payload["translation_warning"] = warning

    return payload


# ----------------------------------------------------
# 2. Sunnah Retrieval Endpoint
# ----------------------------------------------------
@app.post("/ask/sunnah")
async def ask_sunnah(request: QueryRequest):
    user_query = request.query.strip().lower()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    target_section = None
    if any(k in user_query for k in ["پانی", "پینے", "پینے کا"]):
        target_section = "پانی پینے کی سنتیں"
    elif any(k in user_query for k in ["کھانا", "کھانے", "دسترخوان"]):
        target_section = "کھانے کی چند سنتیں"
    elif any(k in user_query for k in ["غسل", "نہانا"]):
        target_section = "غسل کرنے کا مسنون طریقہ"
    elif any(k in user_query for k in ["وضو", "وضو کی"]):
        target_section = "وضو کی سنتیں"
    elif any(k in user_query for k in ["سونا", "سونے", "بستر", "سونے کی"]):
        target_section = "سونے کی سنتیں"
    elif any(k in user_query for k in ["فرائض نماز", "نماز کے فرائض", "نماز کا طریقہ", "نماز کیسے پڑھیں", "نماز"]):
        target_section = "فرائض نماز"
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
    elif any(k in user_query for k in ["بیماری", "علاج", "عیادت", "بیمار"]):
        target_section = "بیماری، علاج اور عیادت کی سنتیں"
    elif any(k in user_query for k in ["نکاح", "شادی", "منگنی"]):
        target_section = "نکاح کی سنتیں"
    elif any(k in user_query for k in ["ولیمہ", "ولیمے"]):
        target_section = "ولیمہ"
    elif any(k in user_query for k in ["بچہ", "ولادت", "عقیقہ", "نام رکھنا"]):
        target_section = "بچہ ہونے کے وقت کی سنتیں"
    elif any(k in user_query for k in ["موت", "ج جنازہ", "دفن", "میت"]):
        target_section = "موت اور اس کے بعد کی سنتیں"
    elif any(k in user_query for k in ["معاشرت", "سلام", "آپس میں"]):
        target_section = "معاشرت کی چند سنتیں"
    elif any(k in user_query for k in ["وساوس", "وسوسہ", "دل میں خیال"]):
        target_section = "وساوس کے وقت کی سنت"
    elif any(k in user_query for k in ["استخارہ"]):
        target_section = "صلوٰۃ استخارہ"
    elif any(k in user_query for k in ["حاجت"]):
        target_section = "صلوٰۃ حاجت"

    # --- English Keyword / Intent Mapping ---
    elif any(k in user_query for k in ["water", "drink", "drinking"]):
        target_section = "پانی پینے کی سنتیں"
    elif any(k in user_query for k in ["eat", "food", "dining", "meal"]):
        target_section = "کھانے کی چند سنتیں"
    elif any(k in user_query for k in ["bath", "shower", "ghusl"]):
        target_section = "غسل کرنے کا مسنون طریقہ"
    elif any(k in user_query for k in ["wudu", "ablution"]):
        target_section = "وضو کی سنتیں"
    elif any(k in user_query for k in ["sleep", "bed", "sleeping", "night"]):
        target_section = "سونے کی سنتیں"
    elif any(k in user_query for k in ["namaz", "prayer", "salah", "faraiz", "obligatory"]):
        target_section = "فرائض نماز"
    elif any(k in user_query for k in ["ruku", "bowing"]):
        target_section = "رکوع کی آٹھ سنتیں"
    elif any(k in user_query for k in ["sujood", "sajdah", "prostration"]):
        target_section = "سجدہ کی بارہ سنتیں"
    elif any(k in user_query for k in ["miswak", "teeth", "tooth"]):
        target_section = "مسواک کی سنتیں"
    elif any(k in user_query for k in ["marriage", "nikah", "wedding"]):
        target_section = "نکاح کی سنتیں"
    elif any(k in user_query for k in ["valimah", "reception"]):
        target_section = "ولیمہ"
    elif any(k in user_query for k in ["sick", "illness", "disease", "hospital", "visiting"]):
        target_section = "بیماری، علاج اور عیادت کی سنتیں"
    elif any(k in user_query for k in ["travel", "journey", "trip"]):
        target_section = "سفر کی سنتیں"
    elif any(k in user_query for k in ["friday", "jummah"]):
        target_section = "جمعہ کی سنتیں"
    elif any(k in user_query for k in ["clothes", "dress", "clothing", "wear"]):
        target_section = "لباس کی سنتیں"

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
                n_results=1,
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
        not_found_msg = "مطلوبہ سنت یا دعاؤں کا مضمون اس کتاب میں نہیں ملا۔"
        english_not_found, warning = safe_translate_to_english(not_found_msg)
        payload = {
            "answer": not_found_msg,
            "english_translation": english_not_found,
            "sources": []
        }
        if warning:
            payload["translation_warning"] = warning
        return payload

    english_answer, warning = safe_translate_to_english(primary_snippet)
    payload = {
        "answer": primary_snippet,
        "english_translation": english_answer,
        "sources": valid_sources
    }
    if warning:
        payload["translation_warning"] = warning

    return payload


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)