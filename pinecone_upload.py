import json
import time
from tqdm import tqdm
from pinecone import Pinecone
import config
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32  # Increased for efficiency (local model handles larger batches well)
INDEX_NAME = config.PINECONE_INDEX_NAME
LOCAL_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim embeddings; alternative: "all-mpnet-base-v2" for 768-dim, better quality

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)
model = SentenceTransformer(LOCAL_MODEL)  # Load local model (downloads on first run)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts, model=model):
    """Generate embeddings locally using SentenceTransformers"""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    print(f"Embedding length: {embeddings.shape[1]}")  # Debug
    return embeddings.tolist()  # Convert to list for Pinecone

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    if not items:
        print("No items to upload!")
        return

    # Get first batch embeddings to determine dimension
    first_texts = [item[1] for item in items[:BATCH_SIZE]]
    first_embeddings = get_embeddings(first_texts)
    embedding_dim = len(first_embeddings[0])
    print(f"Detected embedding dimension: {embedding_dim}")
    if embedding_dim != config.PINECONE_VECTOR_DIM:
        raise ValueError(f"Detected dim {embedding_dim} != config {config.PINECONE_VECTOR_DIM}. Update config.py.")

    # -----------------------------
    # Create or replace index if needed
    # -----------------------------
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME in existing_indexes:
        idx_info = pc.describe_index(INDEX_NAME)
        idx_dim = idx_info["dimension"]
        if idx_dim != embedding_dim:
            print(f"Existing index dimension ({idx_dim}) != detected ({embedding_dim}). Recreating index...")
            pc.delete_index(INDEX_NAME)
            time.sleep(10)  # Wait for deletion
            existing_indexes.remove(INDEX_NAME)

    if INDEX_NAME not in existing_indexes:
        print(f"Creating index {INDEX_NAME} with dimension {embedding_dim}")
        from pinecone import ServerlessSpec
        PINECONE_CLOUD = "aws"
        PINECONE_REGION = config.PINECONE_ENV  # Use corrected region from config
        pc.create_index(
            name=INDEX_NAME,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        time.sleep(30)  # Wait for index to initialize

    # Connect to the index
    index = pc.Index(INDEX_NAME)

    # -----------------------------
    # Upload in batches
    # -----------------------------
    print(f"Preparing to upsert {len(items)} items to Pinecone...")
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        # Safety check
        for emb in embeddings:
            if len(emb) != embedding_dim:
                raise ValueError(f"Embedding dimension {len(emb)} does not match index dimension {embedding_dim}")

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.5)  # Minor delay for Pinecone stability (no rate limits now)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()