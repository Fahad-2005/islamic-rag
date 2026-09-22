import os
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "binoria_toy_fatawa"
BATCH_SIZE = 64

def load_all_csvs():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    dfs = []
    for file_path in csv_files:
        print(f"Loading dataset: {file_path.name}")
        df = pd.read_csv(file_path, encoding="utf-8-sig").fillna("")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter empty questions or answers
    combined_df = combined_df[
        (combined_df["question"].astype(str).str.strip() != "") & 
        (combined_df["answer"].astype(str).str.strip() != "")
    ]

    # Remove duplicates if same question exists across files
    combined_df = combined_df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    
    print(f"Total combined unique records to index: {len(combined_df)}")
    return combined_df

def run_ingestion():
    df = load_all_csvs()

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        print(f"Resetting collection: {COLLECTION_NAME}")
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

    print(f"Ingestion complete! Successfully indexed {collection.count()} documents into {CHROMA_PATH}")

if __name__ == "__main__":
    run_ingestion()