import os
import json
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "sunnah_book"
BATCH_SIZE = 64

def run_sunnah_ingestion():
    json_file = DATA_DIR / "sunnah_data.json"
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found at {json_file}. Please ensure sunnah_data.json is in data/")

    print(f"Loading structured JSON data: {json_file.name}")
    with open(json_file, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for idx, chap in enumerate(chapters, 1):
        title = chap.get("section")
        content = chap.get("content")
        
        # Store strictly the clean content text without any prefixes
        documents.append(content)
        
        metadatas.append({
            "section": title,
            "book_title": "Pyarey Nabi ki Pyari Sunnatein"
        })
        ids.append(f"sunnah_chunk_{idx}")

    print(f"\n✅ Total distinct sections prepared: {len(documents)}")

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Cleared old collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[start:start + BATCH_SIZE],
            metadatas=metadatas[start:start + BATCH_SIZE],
            ids=ids[start:start + BATCH_SIZE],
        )

    print(f"\n✅ Success! Database updated with {collection.count()} full chunks.\n")

if __name__ == "__main__":
    run_sunnah_ingestion()