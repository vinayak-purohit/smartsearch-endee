"""
app.py — Flask web server for SmartSearch.

Routes:
    GET  /              → Web UI
    POST /api/search    → Semantic search
    POST /api/rag       → RAG Q&A
    GET  /api/stats     → Index statistics
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from search import SmartSearchEngine

app = Flask(__name__)
CORS(app)

# Initialise the engine once on startup
engine = SmartSearchEngine()


# ── Web UI ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API endpoints ────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search():
    """Semantic search endpoint."""
    data    = request.get_json(force=True)
    query   = data.get("query", "").strip()
    top_k   = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = engine.search(query, top_k=top_k)
    return jsonify({
        "query":   query,
        "results": results,
        "count":   len(results),
    })


@app.route("/api/rag", methods=["POST"])
def api_rag():
    """Retrieval-Augmented Generation endpoint."""
    data  = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    result = engine.rag_answer(query)
    return jsonify(result)


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Index statistics from Endee."""
    stats = engine.index_stats()
    return jsonify(stats)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  🚀  SmartSearch is running!")
    print("  📖  UI  →  http://localhost:5000")
    print("  🔍  API →  http://localhost:5000/api/search")
    print("  🤖  RAG →  http://localhost:5000/api/rag")
    print("=" * 55)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
