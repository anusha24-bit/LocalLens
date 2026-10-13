"""MCP tool server for LocalLens.

Exposes the LocalLens retrieval stack to any MCP client (Claude Desktop, Claude
Code, an agent framework) as three tools, so an agent can call the same hybrid
search (BM25 + MiniLM, reciprocal rank fusion) that the Streamlit app uses.

Tools
-----
search_evidence   Hybrid retrieval only. Returns ranked, cited passages. Does NOT
                  call the local LLM, so it is fast and deterministic.
ask_locallens     Full grounded pipeline (intent parsing, retrieval, answer
                  composition with citations, abstention when evidence is missing).
index_stats       Corpus size, locations, topics, and which backends are available.

Run (stdio transport):
    PYTHONPATH=src python -m locallens.mcp_server

Claude Desktop / Claude Code config:
    {"mcpServers": {"locallens": {
        "command": "python", "args": ["-m", "locallens.mcp_server"],
        "env": {"PYTHONPATH": "/absolute/path/to/LocalLens/src"}}}}
"""
from __future__ import annotations

from typing import Any

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

MAX_TOP_K = 20
SNIPPET_CHARS = 600

mcp = _Server(
    "locallens",
    instructions=(
        "Grounded local-knowledge search over a curated corpus of city and park guides. "
        "Use search_evidence to fetch cited passages, ask_locallens for a full grounded "
        "answer, and index_stats to see which locations and topics are covered. "
        "If the corpus has no evidence for a question, say so instead of guessing."
    ),
)

_service: Any = None


def get_service() -> Any:
    """Build the LocalLens service once; loading the index is the slow part."""
    global _service
    if _service is None:
        from locallens.service import LocalLensService

        _service = LocalLensService()
    return _service


def _clip(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@mcp.tool()
def search_evidence(
    query: str,
    top_k: int = 5,
    location: str = "",
    topic: str = "",
) -> dict[str, Any]:
    """Hybrid (BM25 + dense, RRF) search over the LocalLens corpus.

    Args:
        query: Natural-language question or keywords.
        top_k: Number of passages to return (1-20).
        location: Optional exact location filter, e.g. "Austin".
        topic: Optional exact topic filter, e.g. "food". See index_stats for values.
    """
    query = query.strip()
    if not query:
        return {"error": "query must not be empty", "results": []}
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    service = get_service()
    if service.retriever is None:
        return {"error": "index not built; run scripts/build_index.py", "results": []}

    filters = {k: v for k, v in (("location", location), ("topic", topic)) if v}
    hits = service.retriever.search(query, top_k=top_k, filters=filters or None)
    return {
        "query": query,
        "filters": filters,
        "count": len(hits),
        "results": [
            {
                "rank": rank,
                "chunk_id": hit.chunk.chunk_id,
                "title": hit.chunk.title,
                "location": hit.chunk.location,
                "topic": hit.chunk.topic,
                "source_type": hit.chunk.source_type,
                "source_url": hit.chunk.source_url,
                "score": round(hit.final_score, 4),
                "passage": _clip(hit.chunk.passage_text),
            }
            for rank, hit in enumerate(hits, start=1)
        ],
    }


@mcp.tool()
def ask_locallens(question: str, location: str = "", topic: str = "") -> dict[str, Any]:
    """Answer a local-knowledge question with citations, or abstain if the corpus
    lacks evidence. Uses the local Ollama model when it is running.
    """
    question = question.strip()
    if not question:
        return {"error": "question must not be empty"}
    payload = get_service().answer(question, location=location, topic=topic)
    data = payload.to_dict()
    # Keep the tool result compact and text-only for agent consumption.
    data.pop("gallery_images", None)
    data["citations"] = [
        {
            "title": c.get("title"),
            "source_url": c.get("source_url"),
            "location": c.get("location"),
            "passage_text": _clip(c.get("passage_text", ""), 300),
        }
        for c in data.get("citations", [])
    ]
    return data


@mcp.tool()
def index_stats() -> dict[str, Any]:
    """Corpus size, covered locations and topics, and available backends."""
    return get_service().stats()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
