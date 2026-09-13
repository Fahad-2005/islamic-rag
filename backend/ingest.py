import os
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

CSV_PATH = os.path.join("..", "data", "toyData.csv")
CHROMA_PATH = "./chroma_store"

def run_ingestion():
    print("Reading CSV data...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    
    # Slice first 70 rows to ensure exact toy size
    subset = df.head(70).fillna("")

    # Initialize Chroma persistent storage
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Lightweight multilingual model (supports Urdu, Arabic, and English)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="binoria_toy_fatawa",
        embedding_function=embedding_func
    )

    documents = []
    metadatas = []
    ids = []

    for idx, row in subset.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        category = str(row.get("category", "")).strip()
        sub_category = str(row.get("sub_category", "")).strip()
        fatwa_no = str(row.get("fatwa_number", "")).strip()
        url = str(row.get("url", "")).strip()

        # Combine question and answer so search matches both inquiry style and ruling text
        text_content = f"سوال: {question}\nجواب: {answer}"

        documents.append(text_content)
        metadatas.append({
            "fatwa_number": fatwa_no,
            "category": category,
            "sub_category": sub_category,
            "url": url,
            "question": question
        })
        ids.append(f"fatwa_{fatwa_no if fatwa_no else idx}")

    print(f"Embedding and storing {len(documents)} records into Chroma DB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Ingestion complete! Data is indexed in ./chroma_store")

if __name__ == "__main__":
    run_ingestion()