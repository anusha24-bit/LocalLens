"""Profile the LocalLens query pipeline and the MiniLM encoder inside it.

Two complementary views, because they answer different questions:

1. Stage breakdown (wall clock) over the 10 benchmark queries: intent parsing,
   place search, BM25, MiniLM query encoding, dense scan, reranking, compose.
   This tells you which component to optimize.
2. torch.profiler over the MiniLM encoder forward pass. This tells you where the
   time inside the model goes once you know the model is worth optimizing.

Run:   PYTHONPATH=src python3 scripts/profile_query_pipeline.py
Saves: artifacts/profile_minilm_trace.json  (chrome://tracing / perfetto)
       artifacts/profile_query_pipeline.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_presentation_eval import QUERIES  # noqa: E402

from locallens.config import get_settings  # noqa: E402
from locallens.service import LocalLensService  # noqa: E402

TRACE_PATH = ROOT / "artifacts" / "profile_minilm_trace.json"
OUT_PATH = ROOT / "artifacts" / "profile_query_pipeline.json"
REPEATS = 3


def stage_breakdown(service: LocalLensService) -> dict[str, Any]:
    """Wrap each pipeline stage to attribute wall time without altering behaviour."""
    dense = service.retriever.dense
    tally = {
        "encode_calls": 0,
        "encode_ms": 0.0,
        "bm25_calls": 0,
        "bm25_ms": 0.0,
        "dense_scan_ms": 0.0,
        "rerank_ms": 0.0,
        "search_calls": 0,
    }

    original = {
        "encode": dense.query_backend.encode,
        "scores": dense._dense_scores,
        "bm25": service.retriever.bm25.search,
        "rerank": service.retriever.reranker.rerank,
        "search": service.retriever.search,
    }

    def timed(key_ms: str, func, key_calls: str | None = None):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            tally[key_ms] += (time.perf_counter() - start) * 1000
            if key_calls:
                tally[key_calls] += 1
            return result

        return wrapper

    dense.query_backend.encode = timed("encode_ms", original["encode"], "encode_calls")
    dense._dense_scores = timed("dense_scan_ms", original["scores"])
    service.retriever.bm25.search = timed("bm25_ms", original["bm25"], "bm25_calls")
    service.retriever.reranker.rerank = timed("rerank_ms", original["rerank"])

    def counted_search(*args, **kwargs):
        tally["search_calls"] += 1
        return original["search"](*args, **kwargs)

    service.retriever.search = counted_search

    for query in QUERIES:  # warmup: first call pays lazy-init costs
        service.answer(query)
    for key in tally:
        tally[key] = 0 if key.endswith("_calls") else 0.0

    start = time.perf_counter()
    for _ in range(REPEATS):
        for query in QUERIES:
            service.answer(query)
    total_ms = (time.perf_counter() - start) * 1000

    # Restore so the service is reusable by later phases.
    dense.query_backend.encode = original["encode"]
    dense._dense_scores = original["scores"]
    service.retriever.bm25.search = original["bm25"]
    service.retriever.reranker.rerank = original["rerank"]
    service.retriever.search = original["search"]

    runs = REPEATS * len(QUERIES)
    accounted = tally["encode_ms"] + tally["bm25_ms"] + tally["dense_scan_ms"] + tally["rerank_ms"]
    return {
        "queries_executed": runs,
        "total_ms": round(total_ms, 1),
        "ms_per_query": round(total_ms / runs, 2),
        "stages": {
            "bm25": {
                "ms_per_query": round(tally["bm25_ms"] / runs, 2),
                "pct": round(100 * tally["bm25_ms"] / total_ms, 1),
                "calls_per_query": round(tally["bm25_calls"] / runs, 2),
            },
            "minilm_encode": {
                "ms_per_query": round(tally["encode_ms"] / runs, 2),
                "pct": round(100 * tally["encode_ms"] / total_ms, 1),
                "calls_per_query": round(tally["encode_calls"] / runs, 2),
            },
            "dense_scan": {
                "ms_per_query": round(tally["dense_scan_ms"] / runs, 2),
                "pct": round(100 * tally["dense_scan_ms"] / total_ms, 1),
            },
            "rerank": {
                "ms_per_query": round(tally["rerank_ms"] / runs, 2),
                "pct": round(100 * tally["rerank_ms"] / total_ms, 1),
            },
            "other_intent_places_compose": {
                "ms_per_query": round((total_ms - accounted) / runs, 2),
                "pct": round(100 * (total_ms - accounted) / total_ms, 1),
            },
        },
        "retriever_search_calls_per_query": round(tally["search_calls"] / runs, 2),
    }


def profile_encoder(service: LocalLensService, top_n: int = 15) -> dict[str, Any]:
    """torch.profiler over the MiniLM forward pass as the pipeline actually calls it."""
    encode = service.retriever.dense.query_backend.encode
    for query in QUERIES:
        encode([query])

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for query in QUERIES:
            with record_function("minilm_encode_query"):
                encode([query])

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(TRACE_PATH))

    table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=top_n, max_name_column_width=45
    )
    rows = []
    for event in sorted(
        prof.key_averages(), key=lambda item: item.self_cpu_time_total, reverse=True
    )[:top_n]:
        rows.append(
            {
                "op": event.key,
                "self_cpu_ms": round(event.self_cpu_time_total / 1000, 3),
                "total_cpu_ms": round(event.cpu_time_total / 1000, 3),
                "calls": event.count,
            }
        )
    return {"table": table, "top_ops": rows, "threads": torch.get_num_threads()}


def main() -> None:
    settings = get_settings()
    service = LocalLensService(settings)

    breakdown = stage_breakdown(service)
    print("=" * 78)
    print(f"PIPELINE STAGE BREAKDOWN ({breakdown['queries_executed']} query executions)")
    print(f"total {breakdown['total_ms']} ms -> {breakdown['ms_per_query']} ms/query")
    print("-" * 78)
    for name, data in sorted(
        breakdown["stages"].items(), key=lambda kv: kv[1]["pct"], reverse=True
    ):
        calls = data.get("calls_per_query")
        suffix = f"  ({calls} calls/query)" if calls is not None else ""
        print(f"  {name:<30} {data['ms_per_query']:>8.2f} ms/query  {data['pct']:>5.1f}%{suffix}")

    encoder = profile_encoder(service)
    print("=" * 78)
    print(f"TORCH.PROFILER - MiniLM encoder (torch threads={encoder['threads']})")
    print(encoder["table"])

    OUT_PATH.write_text(
        json.dumps(
            {"stage_breakdown": breakdown, "minilm_top_ops": encoder["top_ops"]}, indent=2
        )
    )
    print(f"saved: {OUT_PATH.relative_to(ROOT)} and {TRACE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
