"""FP32 vs INT8 dense-index comparison over the existing 10-query evaluation set.

Runs the same queries as scripts/run_presentation_eval.py twice end to end:

  fp32  -> LOCALLENS_DENSE_INT8 off (default)
  int8  -> LOCALLENS_DENSE_INT8 on  (compressed index from Settings.dense_int8)

For every query it compares the final answer text, the retrieved citations, and the
place cards, so any behaviour change caused by the compressed index is visible rather
than hidden behind an aggregate recall number.

Index-scan latency is measured separately, after both indexes are warm and with the two
modes interleaved. Single-shot timings taken during the answer loop are dominated by
cold-cache and model-warmup effects and are not reported as a speed result.

Run:  PYTHONPATH=src python3 scripts/benchmark_fp32_vs_int8.py
Saves: artifacts/fp32_vs_int8_eval.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_presentation_eval import QUERIES  # noqa: E402  the existing 10-query benchmark

OUT_PATH = ROOT / "artifacts" / "fp32_vs_int8_eval.json"
LATENCY_ROUNDS = 5
LATENCY_TRIALS_PER_ROUND = 6


def run_mode(int8_enabled: bool) -> tuple[dict[str, Any], Any]:
    """Build a service with the requested precision and answer every benchmark query."""
    os.environ["LOCALLENS_DENSE_INT8"] = "1" if int8_enabled else "0"

    # Imported lazily and re-read per mode so Settings picks up the env flag above.
    from locallens.config import get_settings
    from locallens.service import LocalLensService

    settings = get_settings()
    assert settings.dense_int8 is int8_enabled, "dense_int8 setting did not take effect"

    load_start = time.perf_counter()
    service = LocalLensService(settings)
    load_seconds = time.perf_counter() - load_start

    dense = service.retriever.dense
    records: list[dict[str, Any]] = []
    for query in QUERIES:
        query_start = time.perf_counter()
        payload = service.answer(query)
        elapsed_ms = (time.perf_counter() - query_start) * 1000
        dense_hits = dense.search(query, top_k=settings.candidate_k)

        records.append(
            {
                "query": query,
                "answer": payload.answer,
                "why": payload.why_this_recommendation,
                "key_tips": payload.key_tips,
                "confidence_note": payload.confidence_note,
                "source_summary": payload.source_summary,
                "used_local_llm": payload.used_local_llm,
                "citation_ids": [citation["chunk_id"] for citation in payload.citations],
                "citation_titles": [citation["title"] for citation in payload.citations],
                "place_cards": [card["name"] for card in payload.place_cards],
                "dense_top10": [hit.chunk.chunk_id for hit in dense_hits[:10]],
                "answer_ms_single_shot": round(elapsed_ms, 1),
            }
        )

    mode = {
        "backend": dense.backend_name,
        "index_bytes": dense.index_nbytes,
        "index_mb": round(dense.index_nbytes / 1e6, 2),
        "service_load_seconds": round(load_seconds, 2),
        "records": records,
    }
    return mode, service


def measure_dense_latency(fp32_dense: Any, int8_dense: Any) -> dict[str, Any]:
    """Interleaved, warmed-up timing of the index scan alone.

    Queries are encoded once up front so the MiniLM forward pass is excluded, and the
    two modes alternate in short rounds so cache state and thermal drift hit both.
    """
    vectors = [fp32_dense.query_backend.encode([query])[0] for query in QUERIES]

    def round_of(dense: Any) -> list[float]:
        samples: list[float] = []
        for _ in range(LATENCY_TRIALS_PER_ROUND):
            for vector in vectors:
                start = time.perf_counter()
                dense._dense_scores(vector)
                samples.append((time.perf_counter() - start) * 1000)
        return samples

    for vector in vectors:  # warmup for both indexes
        fp32_dense._dense_scores(vector)
        int8_dense._dense_scores(vector)

    fp32_samples: list[float] = []
    int8_samples: list[float] = []
    for _ in range(LATENCY_ROUNDS):
        fp32_samples += round_of(fp32_dense)
        int8_samples += round_of(int8_dense)

    def describe(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        return {
            "median_ms": round(statistics.median(samples), 3),
            "mean_ms": round(statistics.mean(samples), 3),
            "p90_ms": round(ordered[int(0.9 * len(ordered))], 3),
            "samples": len(samples),
        }

    fp32_stats = describe(fp32_samples)
    int8_stats = describe(int8_samples)
    return {
        "fp32": fp32_stats,
        "int8": int8_stats,
        "int8_speedup_x": round(fp32_stats["median_ms"] / int8_stats["median_ms"], 2),
    }


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def recall_at_k(reference: list[str], candidate: list[str]) -> float:
    if not reference:
        return 1.0
    return len(set(reference) & set(candidate)) / len(reference)


def compare(fp32: dict[str, Any], int8: dict[str, Any]) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for base, test in zip(fp32["records"], int8["records"]):
        per_query.append(
            {
                "query": base["query"],
                "answer_identical": base["answer"] == test["answer"],
                "why_identical": base["why"] == test["why"],
                "key_tips_identical": base["key_tips"] == test["key_tips"],
                "citations_identical": base["citation_ids"] == test["citation_ids"],
                "citation_jaccard": round(jaccard(base["citation_ids"], test["citation_ids"]), 4),
                "place_cards_identical": base["place_cards"] == test["place_cards"],
                "dense_recall_at_10": round(recall_at_k(base["dense_top10"], test["dense_top10"]), 4),
                "dense_order_identical": base["dense_top10"] == test["dense_top10"],
            }
        )

    count = len(per_query) or 1
    return {
        "queries": len(per_query),
        "answers_identical": sum(item["answer_identical"] for item in per_query),
        "citations_identical": sum(item["citations_identical"] for item in per_query),
        "place_cards_identical": sum(item["place_cards_identical"] for item in per_query),
        "dense_order_identical": sum(item["dense_order_identical"] for item in per_query),
        "mean_citation_jaccard": round(sum(i["citation_jaccard"] for i in per_query) / count, 4),
        "mean_dense_recall_at_10": round(sum(i["dense_recall_at_10"] for i in per_query) / count, 4),
        "index_mb_fp32": fp32["index_mb"],
        "index_mb_int8": int8["index_mb"],
        "index_size_reduction_x": round(fp32["index_bytes"] / max(int8["index_bytes"], 1), 2),
        "per_query": per_query,
    }


def main() -> None:
    fp32, fp32_service = run_mode(int8_enabled=False)
    int8, int8_service = run_mode(int8_enabled=True)
    summary = compare(fp32, int8)
    latency = measure_dense_latency(fp32_service.retriever.dense, int8_service.retriever.dense)

    print("=" * 78)
    print(f"fp32 backend : {fp32['backend']}  ({fp32['index_mb']} MB)")
    print(f"int8 backend : {int8['backend']}  ({int8['index_mb']} MB)")
    print(f"index size reduction: {summary['index_size_reduction_x']}x")
    print(f"local llm used: {fp32['records'][0]['used_local_llm']}")
    print("=" * 78)
    for item in summary["per_query"]:
        flag = "SAME" if item["answer_identical"] else "DIFF"
        print(f"[{flag}] {item['query']}")
        print(
            f"        citations identical={item['citations_identical']} "
            f"jaccard={item['citation_jaccard']} "
            f"dense recall@10={item['dense_recall_at_10']} "
            f"place cards identical={item['place_cards_identical']} "
            f"dense order identical={item['dense_order_identical']}"
        )
    print("=" * 78)
    print(
        f"identical answers      : {summary['answers_identical']}/{summary['queries']}\n"
        f"identical citations    : {summary['citations_identical']}/{summary['queries']}\n"
        f"identical place cards  : {summary['place_cards_identical']}/{summary['queries']}\n"
        f"identical dense order  : {summary['dense_order_identical']}/{summary['queries']}\n"
        f"mean citation jaccard  : {summary['mean_citation_jaccard']}\n"
        f"mean dense recall@10   : {summary['mean_dense_recall_at_10']}"
    )
    print("-" * 78)
    print(
        f"index scan latency (warm, interleaved, n={latency['fp32']['samples']} each)\n"
        f"  fp32 median {latency['fp32']['median_ms']} ms | p90 {latency['fp32']['p90_ms']} ms\n"
        f"  int8 median {latency['int8']['median_ms']} ms | p90 {latency['int8']['p90_ms']} ms\n"
        f"  int8 speedup: {latency['int8_speedup_x']}x "
        f"({'faster' if latency['int8_speedup_x'] > 1 else 'slower'})"
    )

    OUT_PATH.write_text(
        json.dumps(
            {
                "notes": {
                    "ollama_available": fp32["records"][0]["used_local_llm"],
                    "answer_ms_single_shot": (
                        "single untimed-warmup measurement, kept for reference only; "
                        "use dense_latency for the precision comparison"
                    ),
                },
                "fp32": fp32,
                "int8": int8,
                "comparison": summary,
                "dense_latency": latency,
            },
            indent=2,
        )
    )
    print(f"\nsaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
