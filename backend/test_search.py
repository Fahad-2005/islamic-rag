import chromadb

# 1. Connect to your persistent ChromaDB store
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_collection("binoria_toy_fatawa")

# 2. Check total record count
print(f"Total records ready in ChromaDB: {collection.count()}\n")

# 3. Test a sample query to evaluate bge-m3 retrieval accuracy
query_text = "نماز میں بھول جانے پر سجدہ سہو کا کیا طریقہ ہے؟"
print(f"Running test query: '{query_text}'\n")

# Query the collection (fetching top 1 result)
results = collection.query(
    query_texts=[query_text],
    n_results=1
)

# 4. Print out the retrieved results
if results and results['documents'] and len(results['documents'][0]) > 0:
    print("--- Top Match Found ---")
    print(f"Source Origin: {results['metadatas'][0][0].get('source_origin', 'Unknown')}")
    print(f"Content snippet:\n{results['documents'][0][0][:400]}...\n")
    print(f"Distance Score: {results['distances'][0][0]:.4f}")
else:
    print("No matching records found.")