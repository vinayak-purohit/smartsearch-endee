"""
search.py — Core semantic search and RAG logic using Endee.
"""

from typing import Any

from endee import Endee
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
ENDEE_URL  = "http://localhost:8080/api/v1"
INDEX_NAME = "smartsearch"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 5


class SmartSearchEngine:
    """
    Semantic search engine backed by Endee vector database.

    Workflow:
        query (text) ──► embed ──► Endee ANN search ──► ranked results
    """

    def __init__(self):
        print("🔧 Initialising SmartSearch engine...")
        self.model = SentenceTransformer(MODEL_NAME)

        self.client = Endee()
        self.client.set_base_url(ENDEE_URL)
        self.index = self.client.get_index(name=INDEX_NAME)
        print("✅ Engine ready.")

    # ── Public API ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """
        Run semantic search against Endee and return ranked results.

        Args:
            query:  Natural-language search query.
            top_k:  Number of results to return.

        Returns:
            List of result dicts with keys: id, similarity, title, content, category.
        """
        if not query.strip():
            return []

        # 1. Embed the query
        query_vector = self.model.encode(query).tolist()

        # 2. Query Endee vector database
        raw_results = self.index.query(vector=query_vector, top_k=top_k)

        # 3. Format results
        results = []
        for item in raw_results:
            meta = item.get("meta", {})
            results.append({
                "id":         item.get("id"),
                "similarity": round(float(item.get("similarity", 0)), 4),
                "title":      meta.get("title", "Untitled"),
                "content":    meta.get("content", ""),
                "category":   meta.get("category", "General"),
            })

        return results

    def rag_answer(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """
        RAG pipeline: retrieve relevant documents then synthesise an answer.

        In this implementation we use a template-based summariser so no
        external API key is required.  Swap `_synthesise` for an LLM call
        (e.g. OpenAI / Groq) to upgrade to full generation.

        Returns:
            {answer, sources, query}
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return {
                "answer":  "I could not find any relevant documents for your query.",
                "sources": [],
                "query":   query,
            }

        answer = self._synthesise(query, results)

        return {
            "answer":  answer,
            "sources": results,
            "query":   query,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _synthesise(self, query: str, docs: list[dict]) -> str:
        """
        Template-based answer synthesis from retrieved documents.

        Replace this method with an LLM call for richer generation:

            import openai
            ctx = "\\n\\n".join(d["content"] for d in docs)
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer using only the context."},
                    {"role": "user",   "content": f"Context:\\n{ctx}\\n\\nQuestion: {query}"},
                ]
            )
            return resp.choices[0].message.content
        """
        top = docs[0]
        others = docs[1:]

        lines = [
            f"Based on the most relevant document — **{top['title']}**:",
            "",
            top["content"],
        ]

        if others:
            lines += [
                "",
                "**Related information:**",
            ]
            for d in others:
                # One sentence excerpt from each supporting document
                snippet = d["content"].split(".")[0] + "."
                lines.append(f"- *{d['title']}*: {snippet}")

        return "\n".join(lines)

    def index_stats(self) -> dict:
        """Return index metadata from Endee."""
        try:
            indexes = self.client.list_indexes().get("indexes", [])
            for idx in indexes:
                if idx.get("name") == INDEX_NAME:
                    return idx
        except Exception:
            pass
        return {}
