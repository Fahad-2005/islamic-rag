import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import hashlib
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "binoria_toy_fatawa"
BATCH_SIZE = 1  # Ultimate safe batch size for EC2

# ... (keep your load_all_csvs function the same) ...

def run_ingestion():
    df = load_all_csvs()

    print("Initializing BAAI/bge-m3 embedding function...")
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_data = collection.get(include=["metadatas"])
    existing_questions = set()
    if existing_data and "metadatas" in existing_data and existing_data["metadatas"]:
        for meta in existing_data["metadatas"]:
            if meta and "question" in meta:
                existing_questions.add(meta["question"].strip())

    print(f"Already successfully ingested unique questions in DB: {len(existing_questions)}")

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        question = str(row["question"]).strip()
        
        if question[:200] in existing_questions:
            continue

        fatwa_id = str(row.get("fatwa_number", index)).strip()
        answer = str(row["answer"]).strip()
        category = str(row.get("category", "عام")).strip()
        sub_category = str(row.get("sub_category", "متفرق")).strip()
        url = str(row.get("url", "")).strip()
        source_origin = str(row.get("source_origin", "مستند فتاویٰ")).strip()

        unique_string = f"{fatwa_id}_{question}"
        hash_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
        doc_id = f"fatwa_{hash_id}"

        doc_text = f"زمرہ: {category}\nسوال: {question}\nجواب: {answer}"

        documents.append(doc_text)
        metadatas.append({
            "fatwa_number": fatwa_id,
            "category": category,
            "sub_category": sub_category,
            "question": question[:200],
            "url": url,
            "source_origin": source_origin,
        })
        ids.append(doc_id)

    print(f"Remaining records left to ingest: {len(documents)}")

    if len(documents) == 0:
        print("All records are already fully ingested!")
        return

    total_items = len(documents)
    for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
        collection.add(
            documents=[doc],
            metadatas=[meta],
            ids=[doc_id],
        )
        
        gc.collect()

        if (i + 1) % 50 == 0 or (i + 1) == total_items:
            print(f"Progress: {i + 1} / {total_items} remaining records processed")

    print(f"Ingestion complete! Total records in collection now: {collection.count()}")

if __name__ == "__main__":
    run_ingestion()