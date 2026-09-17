from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "banuri_scraped.csv"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "binoria_toy_fatawa"
BATCH_SIZE = 64


def run_ingestion():
    print(f"Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig").fillna("")
    df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")]
    print(f"Total valid records to index: {len(df)}")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []
    for index, row in df.iterrows():
        fatwa_id = str(row.get("fatwa_number", index)).strip()
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        documents.append(f"سوال: {question}\nجواب: {answer}")
        metadatas.append({
            "fatwa_number": fatwa_id,
            "category": str(row.get("category", "عام")),
            "sub_category": str(row.get("sub_category", "متفرق")),
            "question": question[:200],
            "url": str(row.get("url", "")),
        })
        ids.append(f"fatwa_{fatwa_id}_{index}")

    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[start:start + BATCH_SIZE],
            metadatas=metadatas[start:start + BATCH_SIZE],
            ids=ids[start:start + BATCH_SIZE],
        )
        print(f"Indexed batch {start // BATCH_SIZE + 1} / {total_batches}")

    print(f"Ingestion complete! Indexed {collection.count()} documents into {CHROMA_PATH}")


if __name__ == "__main__":
    run_ingestion()