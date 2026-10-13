"""Embedding-quantization benchmark on LocalLens's real dense index.

Loads artifacts/chunk_embeddings.npy (all-MiniLM-L6-v2 vectors for every chunk) and
measures how much retrieval quality is lost when the vectors are stored at lower precision:

  fp32 (baseline) | fp16 | int8 (per-vector symmetric scalar quantization) | 1-bit (sign)

Method (leave-one-out): sample N chunk vectors as queries, rank all *other* chunks by cosine
similarity, and compare each lower-precision top-10 against the fp32 top-10 (recall@10).
Queries stay fp32; only the stored index is quantized.

Run:  python scripts/benchmark_embedding_quantization.py
Saves: artifacts/embedding_quantization_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "artifacts" / "chunk_embeddings.npy"
OUT_PATH = ROOT / "artifacts" / "embedding_quantization_results.json"
N_QUERIES = 500
TOP_K = 10
SEED = 0


def top_k_excluding_self(scores: np.ndarray, query_ids: np.ndarray, k: int) -> np.ndarray:
    scores = scores.copy()
    scores[np.arange(len(query_ids)), query_ids] = -np.inf  # a chunk is not its own neighbour
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def recall_at_k(ref: np.ndarray, test: np.ndarray) -> float:
    hits = [len(set(r) & set(t)) / len(r) for r, t in zip(ref, test)]
    return float(np.mean(hits))


def main() -> None:
    matrix = np.load(MATRIX_PATH).astype(np.float32)
    n, d = matrix.shape
    rng = np.random.default_rng(SEED)
    query_ids = rng.choice(n, size=N_QUERIES, replace=False)
    queries = matrix[query_ids]

    t0 = time.perf_counter()
    fp32_scores = queries @ matrix.T
    fp32_ms_per_query = (time.perf_counter() - t0) * 1000 / N_QUERIES
    ref = top_k_excluding_self(fp32_scores, query_ids, TOP_K)

    results = {
        "index_shape": [n, d],
        "queries": N_QUERIES,
        "top_k": TOP_K,
        "fp32": {"size_mb": round(matrix.nbytes / 1e6, 2), "recall_at_10": 1.0,
                 "search_ms_per_query_numpy": round(fp32_ms_per_query, 3)},
    }

    # fp16
    m16 = matrix.astype(np.float16)
    s16 = queries @ m16.astype(np.float32).T
    results["fp16"] = {"size_mb": round(m16.nbytes / 1e6, 2),
                       "recall_at_10": round(recall_at_k(ref, top_k_excluding_self(s16, query_ids, TOP_K)), 4)}

    # int8: per-vector symmetric scale
    scale = np.abs(matrix).max(axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q8 = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)
    deq = q8.astype(np.float32) * scale
    s8 = queries @ deq.T
    results["int8"] = {
        "size_mb": round((q8.nbytes + scale.astype(np.float32).nbytes) / 1e6, 2),
        "recall_at_10": round(recall_at_k(ref, top_k_excluding_self(s8, query_ids, TOP_K)), 4),
        "mean_abs_score_error": float(np.abs(s8 - fp32_scores).mean()),
    }

    # 1-bit (sign) quantization
    bits = np.packbits(matrix > 0, axis=1)
    signs = np.where(matrix > 0, 1.0, -1.0).astype(np.float32)
    s1 = np.where(queries > 0, 1.0, -1.0).astype(np.float32) @ signs.T
    results["1bit"] = {"size_mb": round(bits.nbytes / 1e6, 2),
                       "recall_at_10": round(recall_at_k(ref, top_k_excluding_self(s1, query_ids, TOP_K)), 4)}

    for name in ("fp16", "int8", "1bit"):
        results[name]["size_reduction_x"] = round(results["fp32"]["size_mb"] / results[name]["size_mb"], 2)

    print(json.dumps(results, indent=2))
    OUT_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
