"""
ingest.py — Load documents into Endee vector database.

Steps:
1. Read documents from data/documents.json
2. Generate embeddings using sentence-transformers
3. Upsert into Endee index
"""

import json
import sys
import time
from pathlib import Path

from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
ENDEE_URL   = "http://localhost:8080/api/v1"
INDEX_NAME  = "smartsearch"
MODEL_NAME  = "all-MiniLM-L6-v2"   # 384-dim, fast, no API key needed
DIMENSION   = 384
BATCH_SIZE  = 16
DATA_FILE   = Path(__file__).parent / "data" / "documents.json"


def wait_for_endee(client: Endee, retries: int = 15, delay: int = 2) -> None:
    """Poll until Endee is ready."""
    print("⏳ Waiting for Endee to be ready...")
    for i in range(retries):
        try:
            client.list_indexes()
            print("✅ Endee is ready!\n")
            return
        except Exception:
            print(f"   Attempt {i+1}/{retries} — retrying in {delay}s...")
            time.sleep(delay)
    print("❌ Could not connect to Endee. Is the server running?")
    sys.exit(1)


def main():
    # 1. Connect to Endee
    print("=" * 60)
    print("  SmartSearch — Endee Ingestion Pipeline")
    print("=" * 60)

    client = Endee()
    client.set_base_url(ENDEE_URL)
    wait_for_endee(client)

    # 2. Create index (idempotent)
    existing = [idx["name"] for idx in client.list_indexes().get("indexes", [])]
    if INDEX_NAME in existing:
        print(f"ℹ️  Index '{INDEX_NAME}' already exists — skipping creation.\n")
    else:
        print(f"📦 Creating index '{INDEX_NAME}' (dim={DIMENSION}, cosine, INT8)...")
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8,
        )
        print("✅ Index created.\n")

    index = client.get_index(name=INDEX_NAME)

    # 3. Load documents
    print(f"📂 Loading documents from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"   Loaded {len(docs)} documents.\n")

    # 4. Generate embeddings
    print(f"🤖 Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print("   Model loaded.\n")

    texts = [f"{doc['title']}. {doc['content']}" for doc in docs]

    print(f"🔢 Generating {len(texts)} embeddings (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    print()

    # 5. Upsert into Endee
    print("📤 Upserting vectors into Endee...")
    vectors = []
    for doc, emb in zip(docs, embeddings):
        vectors.append({
            "id":     doc["id"],
            "vector": emb.tolist(),
            "meta": {
                "title":    doc["title"],
                "content":  doc["content"],
                "category": doc["category"],
            },
        })

    # Upsert in batches
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="   Upserting"):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(batch)

    print()
    print("=" * 60)
    print(f"✅ Ingestion complete! {len(vectors)} documents indexed.")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Endee Dashboard: http://localhost:8080")
    print(f"   App: python app.py  →  http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
