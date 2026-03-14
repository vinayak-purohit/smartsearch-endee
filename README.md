# 🔍 SmartSearch — AI Semantic Document Search & RAG Q&A

> **Built for Endee.io SDE/ML Engineer Trainee Evaluation**
> An end-to-end AI system using [Endee Vector Database](https://github.com/endee-io/endee) for semantic search and Retrieval-Augmented Generation (RAG).

[![Endee](https://img.shields.io/badge/Vector_DB-Endee-00e5ff?style=flat-square)](https://github.com/endee-io/endee)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Problem Statement

Traditional keyword-based search fails when users phrase queries differently from how documents are written. A user asking *"how do I store vectors?"* gets no results when the document says *"indexing high-dimensional embeddings"* — even though they mean the same thing.

**SmartSearch** solves this with semantic understanding: it embeds both documents and queries into a shared vector space, then uses Endee's high-performance ANN search to find the most *meaning-similar* results — regardless of exact word choice.

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        SmartSearch System                        │
├──────────────────────────┬──────────────────────────────────────┤
│     INGESTION PIPELINE   │          QUERY PIPELINE              │
│                          │                                      │
│  documents.json          │  User Query (text)                   │
│       │                  │       │                              │
│       ▼                  │       ▼                              │
│  SentenceTransformer     │  SentenceTransformer                 │
│  (all-MiniLM-L6-v2)      │  (all-MiniLM-L6-v2)                  │
│       │                  │       │                              │
│  384-dim embeddings      │  384-dim query vector                │
│       │                  │       │                              │
│       ▼                  │       ▼                              │
│  ┌─────────────────┐     │  ┌─────────────────┐                │
│  │  Endee Vector   │     │  │  Endee ANN      │                │
│  │  Database       │◄────┼──│  Search (HNSW)  │                │
│  │  (HNSW + INT8)  │     │  └─────────────────┘                │
│  └─────────────────┘     │       │                              │
│                          │       ▼                              │
│                          │  Top-K Results + Scores              │
│                          │       │                              │
│                          │  [RAG mode] Synthesise Answer        │
│                          │       │                              │
│                          │  Flask API → Web UI                  │
└──────────────────────────┴──────────────────────────────────────┘
```

### How Endee is Used

| Step | Endee API | Purpose |
|------|-----------|---------|
| Index creation | `client.create_index(dimension=384, space_type="cosine", precision=INT8)` | Create a cosine-similarity index with INT8 quantisation |
| Document storage | `index.upsert([{id, vector, meta}])` | Store 384-dim embeddings + metadata |
| Semantic search | `index.query(vector=query_emb, top_k=5)` | ANN search returns top-K similar docs with similarity scores |
| Index inspection | `client.list_indexes()` | Display index stats in the UI |

### Why Endee?

- **Speed**: HNSW indexing delivers sub-millisecond ANN search at scale
- **INT8 precision**: Reduces memory footprint by 4× vs float32 with negligible accuracy loss
- **Simple API**: Python SDK makes integration effortless
- **Self-hosted**: Data never leaves your infrastructure — ideal for private document search

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose (for Endee)
- Python 3.10+

### Step 1 — Fork & Star the Endee Repository

> ⚠️ **Mandatory before proceeding:**

1. Go to [https://github.com/endee-io/endee](https://github.com/endee-io/endee)
2. Click **⭐ Star** the repository
3. Click **Fork** → Fork to your personal GitHub account
4. Clone **your fork**:

```bash
git clone https://github.com/<YOUR_USERNAME>/endee.git
```

### Step 2 — Clone This Project

```bash
git clone https://github.com/<YOUR_USERNAME>/smartsearch-endee.git
cd smartsearch-endee
```

### Step 3 — Start Endee Vector Database

```bash
docker compose up -d
```

This pulls the official `endeeio/endee-server:latest` image and starts Endee on port `8080`.

Verify it's running:

```bash
curl http://localhost:8080/api/v1/index/list
# → {"indexes":[]}
```

You can also visit the **Endee Dashboard** at [http://localhost:8080](http://localhost:8080).

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) automatically.

### Step 5 — Ingest Documents into Endee

```bash
python ingest.py
```

This will:
1. Connect to Endee and create the `smartsearch` index
2. Load 30 AI/tech documents from `data/documents.json`
3. Generate 384-dimensional embeddings using SentenceTransformers
4. Upsert all vectors + metadata into Endee

Expected output:
```
✅ Endee is ready!
📦 Creating index 'smartsearch' (dim=384, cosine, INT8)...
📂 Loading 30 documents...
🤖 Loading embedding model...
🔢 Generating embeddings...
📤 Upserting vectors into Endee...
✅ Ingestion complete! 30 documents indexed.
```

### Step 6 — Launch the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🎯 Features

### 1. Semantic Search
Search documents by meaning, not just keywords. The query is embedded and compared against all document vectors in Endee using cosine similarity.

**Example queries:**
- `"how does neural network learn?"` → finds ML, deep learning, and transformer docs
- `"storing data efficiently"` → finds vector DB and indexing docs
- `"building AI applications"` → finds RAG, agentic AI, and LLM docs

### 2. RAG Q&A Mode
Ask questions in natural language. The system:
1. Retrieves the top-3 most relevant documents from Endee
2. Synthesises a grounded answer from the retrieved context
3. Shows the source documents used

**Example:**
> Q: *What is the difference between semantic search and keyword search?*
>
> A: Based on the most relevant document — **Semantic Search vs Keyword Search**: Keyword search matches exact words or phrases...

### 3. Live Index Stats
The UI displays real-time stats from Endee: document count, dimensions, precision mode, and distance metric.

---

## 📁 Project Structure

```
smartsearch-endee/
├── docker-compose.yml      # Endee vector database setup
├── requirements.txt        # Python dependencies
├── ingest.py               # Data ingestion pipeline → Endee
├── search.py               # Core semantic search + RAG logic
├── app.py                  # Flask web server & REST API
├── data/
│   └── documents.json      # 30 AI/tech documents (sample corpus)
├── templates/
│   └── index.html          # Web UI (dark terminal aesthetic)
└── README.md
```

---

## 🔌 REST API Reference

### `POST /api/search`
Semantic vector search via Endee.

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do transformers work?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "how do transformers work?",
  "count": 5,
  "results": [
    {
      "id": "doc_005",
      "similarity": 0.8921,
      "title": "Transformer Architecture",
      "content": "...",
      "category": "AI/ML"
    }
  ]
}
```

### `POST /api/rag`
Retrieval-Augmented Generation — retrieves context from Endee and synthesises an answer.

```bash
curl -X POST http://localhost:5000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG and why is it useful?"}'
```

**Response:**
```json
{
  "query": "What is RAG and why is it useful?",
  "answer": "Based on the most relevant document — Retrieval-Augmented Generation (RAG): ...",
  "sources": [ ... ]
}
```

### `GET /api/stats`
Returns index metadata from Endee.

```bash
curl http://localhost:5000/api/stats
```

---

## ⚙️ Configuration

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `ENDEE_URL` | `search.py` / `ingest.py` | `http://localhost:8080/api/v1` | Endee server URL |
| `INDEX_NAME` | `search.py` / `ingest.py` | `smartsearch` | Endee index name |
| `MODEL_NAME` | `search.py` / `ingest.py` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `DIMENSION` | `ingest.py` | `384` | Embedding dimensions |
| `TOP_K` | `search.py` | `5` | Default number of results |

---

## 🧪 Adding Your Own Documents

Edit `data/documents.json` following this schema:

```json
{
  "id": "doc_unique_id",
  "title": "Document Title",
  "content": "Full document text content...",
  "category": "Category Name"
}
```

Then re-run `python ingest.py` to index the new documents.

---

## 🔄 Extending to Full LLM Generation

To upgrade the RAG pipeline to use a real LLM (e.g. OpenAI GPT-4o-mini), replace the `_synthesise` method in `search.py`:

```python
import openai

def _synthesise(self, query, docs):
    context = "\n\n".join(d["content"] for d in docs)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely using only the provided context."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
    )
    return response.choices[0].message.content
```

---

## 🛑 Stopping the Services

```bash
# Stop the Flask app: Ctrl+C

# Stop Endee
docker compose down

# Stop Endee and remove all indexed data
docker compose down -v
```

---

## 🙏 Built With

| Tool | Role |
|------|------|
| [Endee Vector DB](https://github.com/endee-io/endee) | High-performance ANN search & vector storage |
| [Sentence Transformers](https://sbert.net) | Text → embedding model (`all-MiniLM-L6-v2`) |
| [Flask](https://flask.palletsprojects.com) | REST API & web server |
| [Docker](https://docker.com) | Containerised Endee deployment |

---

## 📄 License

MIT © 2025
