import os
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "binoria_300.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Ensure no empty questions or answers exist
df = df.dropna(subset=["question", "answer"])
print(f"Total valid records to index: {len(df)}")

# 2. Local SentenceTransformer Embedding Setup
# Produces the exact same mathematical vectors as the Hugging Face API,
# but runs locally on your PC without network/DNS connection failures.
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Initialize Persistent Vector Client
client = chromadb.PersistentClient(path=CHROMA_PATH)

COLLECTION_NAME = "binoria_toy_fatawa"

# Clear existing collection to avoid duplicate vectors
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Resetting existing collection: {COLLECTION_NAME}")
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

# 4. Prepare Batch Ingestion
documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    fatwa_id = str(row.get("fatwa_number", idx))
    q_text = str(row.get("question", "")).strip()
    a_text = str(row.get("answer", "")).strip()

    # Formatted document chunk for semantic retrieval
    doc_chunk = f"سوال: {q_text}\nجواب: {a_text}"

    documents.append(doc_chunk)
    metadatas.append({
        "fatwa_number": fatwa_id,
        "category": str(row.get("category", "عام")),
        "sub_category": str(row.get("sub_category", "متفرق")),
        "question": q_text[:200],  # preview
        "url": str(row.get("url", ""))
    })
    ids.append(f"fatwa_{fatwa_id}_{idx}")

# 5. Insert in Chunks
BATCH_SIZE = 64
total_batches = (len(documents) - 1) // BATCH_SIZE + 1

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]

    collection.add(
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )
    current_batch = (i // BATCH_SIZE) + 1
    print(f"Indexed batch {current_batch} / {total_batches}")

print(f"\nIngestion complete! Successfully indexed {collection.count()} documents into {CHROMA_PATH}")