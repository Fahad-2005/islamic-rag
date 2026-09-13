import os
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

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Retrieve top 2 most relevant records
    search_results = collection.query(
        query_texts=[user_query],
        n_results=2
    )

    retrieved_docs = search_results["documents"][0]
    retrieved_meta = search_results["metadatas"][0]

    if not retrieved_docs:
        return {
            "answer": "معذرت، فراہم کردہ ڈیٹا بیس میں اس سوال کا کوئی مصدقہ حوالہ نہیں ملا۔",
            "sources": []
        }

    # Format context for the LLM
    context_text = "\n\n---\n\n".join(retrieved_docs)

    system_prompt = (
        "You are an assistant for verified Islamic rulings. "
        "Answer the question strictly based on the provided context in the language of the question (Urdu or English). "
        "Do not extrapolate, assume, or invent rulings. "
        "If the context does not contain enough information to answer the question, "
        "reply strictly that the ruling is not available in the verified records."
    )

    user_prompt = f"""Context:
{context_text}

User Question:
{user_query}

Provide a concise, grounded answer:"""

    try:
        if not groq_client:
            # Fallback if no API key is provided
            generated_answer = (
                "[LLM API Key Missing] Displaying retrieved context directly:\n\n"
                + context_text
            )
        else:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            generated_answer = response.choices[0].message.content

        sources_list = []
        for meta, doc in zip(retrieved_meta, retrieved_docs):
            sources_list.append({
                "fatwa_number": meta.get("fatwa_number"),
                "category": meta.get("category"),
                "sub_category": meta.get("sub_category"),
                "url": meta.get("url"),
                "snippet": doc
            })

        return {
            "answer": generated_answer,
            "sources": sources_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)