import os
from pathlib import Path
import docx
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

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

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Clear old fragmented data
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Cleared old fragmented collection.")
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
    
    current_section = "مقدمہ (Introduction)"
    section_content = []
    count = 0

    # Words to ignore so we don't index introductory pages/author titles
    ignore_list = ["خلیفہ", "دامت برکاتہم", "قارئین", "ہدایت و ارشاد"]

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text or any(ignored in text for ignored in ignore_list):
            continue

        # Detect headers (short lines without full stops or brackets)
        is_heading = len(text) < 65 and not text.endswith("۔") and not text.startswith("(")

        if is_heading:
            if section_content:
                count += 1
                # Join paragraphs with double line breaks for readability
                documents.append("\n\n".join(section_content))
                metadatas.append({
                    "section": current_section,
                    "book_title": "Pyarey Nabi ki Pyari Sunnatein"
                })
                ids.append(f"sunnah_grouped_{count}")
            
            current_section = text
            section_content = []
        else:
            section_content.append(text)

    # Save final section
    if section_content:
        count += 1
        documents.append("\n\n".join(section_content))
        metadatas.append({
            "section": current_section,
            "book_title": "Pyarey Nabi ki Pyari Sunnatein"
        })
        ids.append(f"sunnah_grouped_{count}")

    print(f"Grouped {len(documents)} complete sections.")

    # Batch insert into ChromaDB
    for start in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[start:start + BATCH_SIZE],
            metadatas=metadatas[start:start + BATCH_SIZE],
            ids=ids[start:start + BATCH_SIZE],
        )

    print(f"✅ Success! Ingested into '{COLLECTION_NAME}'")

if __name__ == "__main__":
    run_sunnah_ingestion()