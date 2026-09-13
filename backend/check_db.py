import chromadb

# Point to your persistent chroma store folder
client = chromadb.PersistentClient(path="./chroma_store")

# Get the collection
try:
    collection = client.get_collection(name="binoria_toy_fatawa")
    count = collection.count()
    print(f"Total documents stored: {count}")

    if count > 0:
        # Fetch the first 3 documents and their metadata
        sample = collection.peek(limit=3)
        print("\n--- SAMPLE STORED DATA ---")
        for i in range(len(sample["ids"])):
            print(f"\nID: {sample['ids'][i]}")
            print(f"Metadata: {sample['metadatas'][i]}")
            print(f"Document Text:\n{sample['documents'][i][:150]}...")
    else:
        print("Collection is empty. Ingest script needs to be re-run.")
except Exception as e:
    print(f"Error accessing collection: {e}")