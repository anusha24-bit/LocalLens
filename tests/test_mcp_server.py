"""Protocol-level tests for the LocalLens MCP server.

The LocalLens service is replaced with a stub so these run without the index,
embedding model, or Ollama. They check tool registration, argument handling,
filters, clamping, and error paths through the real MCP tool-call layer.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from locallens import mcp_server


@dataclass
class Chunk:
    chunk_id: str
    title: str = "Guide"
    location: str = "Austin"
    topic: str = "food"
    source_type: str = "wikivoyage"
    source_url: str = "https://example.com"
    passage_text: str = "Tacos " * 300


@dataclass
class Hit:
    chunk: Chunk
    final_score: float = 0.0123456


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=None, candidate_k=None, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [Hit(Chunk(f"c{i}")) for i in range(top_k)]


@dataclass
class Payload:
    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": "Try the tacos.",
            "citations": [{"title": "Guide", "source_url": "u", "location": "Austin", "passage_text": "x" * 900}],
            "gallery_images": [{"url": "img"}],
            "used_local_llm": False,
        }


@dataclass
class StubService:
    retriever: Any = field(default_factory=StubRetriever)

    def answer(self, question, *, location="", topic=""):
        self.last = (question, location, topic)
        return Payload()

    def stats(self):
        return {"chunks": 3, "locations": ["Austin"], "topics": ["food"]}


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    svc = StubService()
    monkeypatch.setattr(mcp_server, "_service", svc)
    return svc


def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(mcp_server.mcp.call_tool(name, args))
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc)
    if isinstance(result, tuple):
        return result[1].get("result", result[1]) if isinstance(result[1], dict) else result[1]
    return json.loads(result[0].text if isinstance(result, list) else result.content[0].text)


def test_three_tools_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {"search_evidence", "ask_locallens", "index_stats"}


def test_search_returns_ranked_clipped_cited_results(stub):
    out = call("search_evidence", {"query": "best tacos", "top_k": 3})
    assert out["count"] == 3
    assert [r["rank"] for r in out["results"]] == [1, 2, 3]
    assert all(len(r["passage"]) <= mcp_server.SNIPPET_CHARS for r in out["results"])
    assert all(r["source_url"] for r in out["results"])


def test_search_passes_filters_and_clamps_top_k(stub):
    call("search_evidence", {"query": "q", "top_k": 999, "location": "Austin", "topic": "food"})
    sent = stub.retriever.calls[-1]
    assert sent["top_k"] == mcp_server.MAX_TOP_K
    assert sent["filters"] == {"location": "Austin", "topic": "food"}


def test_search_without_filters_sends_none(stub):
    call("search_evidence", {"query": "q"})
    assert stub.retriever.calls[-1]["filters"] is None


def test_empty_query_is_an_error_not_a_crash():
    out = call("search_evidence", {"query": "   "})
    assert out["results"] == [] and "error" in out


def test_unbuilt_index_reports_error(stub):
    stub.retriever = None
    out = call("search_evidence", {"query": "q"})
    assert "index not built" in out["error"]


def test_ask_compacts_payload(stub):
    out = call("ask_locallens", {"question": "where to eat?", "location": "Austin"})
    assert "gallery_images" not in out
    assert len(out["citations"][0]["passage_text"]) <= 300
    assert stub.last == ("where to eat?", "Austin", "")


def test_index_stats():
    assert call("index_stats", {})["chunks"] == 3
