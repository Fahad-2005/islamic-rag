import os
from pathlib import Path
import docx
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
CHROMA_PATH = BASE_DIR / "chroma_store"
COLLECTION_NAME = "sunnah_book"
DOCX_FILE = DATA_DIR / "Piyare_Nabi_Ki_Piyari_Sunnatein_Clean_RTL.docx"
BATCH_SIZE = 64

def run_sunnah_ingestion():
    if not DOCX_FILE.exists():
        raise FileNotFoundError(f"Word file not found at: {DOCX_FILE}")

    print(f"Loading document: {DOCX_FILE.name}")
    doc = docx.Document(DOCX_FILE)

    # Use the exact same embedding model as your fatwa collection
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Safely create or recreate the sunnah_book collection
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

    current_section = "General"
    count = 0

    print("Parsing paragraphs and headers...")
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect section headings
        if len(text) < 60 and not text.startswith("1") and not text.startswith("|") and not text.startswith("("):
            current_section = text
            continue

        count += 1
        documents.append(text)
        metadatas.append({
            "section": current_section,
            "book_title": "Pyarey Nabi ki Pyari Sunnatein",
            "source": "Khanqah Imdadia Ashrafia"
        })
        ids.append(f"sunnah_{count}")

    print(f"Total extracted Sunnah records to index: {len(documents)}")

    # Batch indexing into ChromaDB
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    for start in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[start:start + BATCH_SIZE],
            metadatas=metadatas[start:start + BATCH_SIZE],
            ids=ids[start:start + BATCH_SIZE],
        )
        print(f"Indexed batch {start // BATCH_SIZE + 1} / {total_batches}")

    print(f" Success! Ingested {collection.count()} items into '{COLLECTION_NAME}' inside {CHROMA_PATH}")

if __name__ == "__main__":
    run_sunnah_ingestion()