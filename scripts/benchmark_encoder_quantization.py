"""INT8 weight quantization of the MiniLM query encoder vs FP32.

This quantizes the *model weights*, which is a different axis from the int8 dense
index already supported by Settings.dense_int8 (that compresses stored document
vectors; this compresses the encoder that produces query vectors).

Method: torch dynamic quantization over nn.Linear modules (qint8 weights, activations
quantized on the fly). Dynamic quantization is the right fit for transformer inference
on CPU, and unlike static quantization it needs no calibration set.

Measured:
  - encoder weight size, fp32 vs int8
  - query-encode latency, warm and interleaved
  - embedding fidelity vs fp32 (cosine similarity)
  - end-to-end retrieval impact: answers/citations over the 10 benchmark queries,
    with the quantized encoder querying the existing fp32 index (the realistic
    deployment: you quantize the model you run at query time)

Run:   PYTHONPATH=src python3 scripts/benchmark_encoder_quantization.py
Saves: artifacts/encoder_quantization_results.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_presentation_eval import QUERIES  # noqa: E402

from locallens.config import get_settings  # noqa: E402
from locallens.service import LocalLensService  # noqa: E402

OUT_PATH = ROOT / "artifacts" / "encoder_quantization_results.json"
ROUNDS = 5
TRIALS = 6


def weight_bytes(module: nn.Module) -> int:
    """Size of the parameters actually stored, counting packed int8 Linear weights."""
    total = 0
    for sub in module.modules():
        if hasattr(sub, "_packed_params") and hasattr(sub._packed_params, "_weight_bias"):
            weight, bias = sub._packed_params._weight_bias()
            total += weight.numel() * weight.element_size()
            if bias is not None:
                total += bias.numel() * bias.element_size()
        else:
            for name, param in sub.named_parameters(recurse=False):
                total += param.numel() * param.element_size()
            for name, buf in sub.named_buffers(recurse=False):
                total += buf.numel() * buf.element_size()
    return total


def encode_all(backend, queries: list[str]) -> np.ndarray:
    return np.vstack([backend.encode([query])[0] for query in queries])


def bench_encode(backend, queries: list[str]) -> list[float]:
    samples: list[float] = []
    for _ in range(TRIALS):
        for query in queries:
            start = time.perf_counter()
            backend.encode([query])
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def describe(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "p90_ms": round(ordered[int(0.9 * len(ordered))], 3),
        "samples": len(samples),
    }


def answers_for(service: LocalLensService) -> list[dict[str, Any]]:
    rows = []
    for query in QUERIES:
        payload = service.answer(query)
        rows.append(
            {
                "query": query,
                "answer": payload.answer,
                "citations": [citation["chunk_id"] for citation in payload.citations],
                "place_cards": [card["name"] for card in payload.place_cards],
            }
        )
    return rows


def main() -> None:
    settings = get_settings()
    service = LocalLensService(settings)
    backend = service.retriever.dense.query_backend
    if not hasattr(backend, "model"):
        raise SystemExit("query backend is not a SentenceTransformer; cannot quantize")

    # This macOS arm64 wheel ships qnnpack but leaves the engine unselected, which
    # makes quantized::linear_prepack fail with NoQEngine. Select it explicitly.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"

    fp32_model = backend.model
    fp32_bytes = weight_bytes(fp32_model)

    baseline_answers = answers_for(service)
    fp32_vectors = encode_all(backend, QUERIES)

    # Dynamic quantization of every Linear layer to int8 weights.
    quantized_model = torch.ao.quantization.quantize_dynamic(
        fp32_model, {nn.Linear}, dtype=torch.qint8
    )
    int8_bytes = weight_bytes(quantized_model)

    class QuantizedBackend:
        model_name = backend.model_name

        def __init__(self, model):
            self.model = model

        def encode(self, texts: list[str]) -> np.ndarray:
            matrix = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(matrix, dtype=np.float32)

    int8_backend = QuantizedBackend(quantized_model)
    int8_vectors = encode_all(int8_backend, QUERIES)

    cosines = [
        float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for a, b in zip(fp32_vectors, int8_vectors)
    ]

    # Warm both, then interleave so cache and thermal effects hit each equally.
    bench_encode(backend, QUERIES)
    bench_encode(int8_backend, QUERIES)
    fp32_samples: list[float] = []
    int8_samples: list[float] = []
    for _ in range(ROUNDS):
        fp32_samples += bench_encode(backend, QUERIES)
        int8_samples += bench_encode(int8_backend, QUERIES)

    # End-to-end: swap the quantized encoder into the live retriever.
    service.retriever.dense.query_backend = int8_backend
    int8_answers = answers_for(service)

    identical_answers = sum(
        a["answer"] == b["answer"] for a, b in zip(baseline_answers, int8_answers)
    )
    identical_citations = sum(
        a["citations"] == b["citations"] for a, b in zip(baseline_answers, int8_answers)
    )
    identical_cards = sum(
        a["place_cards"] == b["place_cards"] for a, b in zip(baseline_answers, int8_answers)
    )

    fp32_stats = describe(fp32_samples)
    int8_stats = describe(int8_samples)
    results = {
        "encoder": backend.model_name,
        "weights": {
            "fp32_mb": round(fp32_bytes / 1e6, 2),
            "int8_mb": round(int8_bytes / 1e6, 2),
            "reduction_x": round(fp32_bytes / max(int8_bytes, 1), 2),
        },
        "encode_latency": {
            "fp32": fp32_stats,
            "int8": int8_stats,
            "int8_speedup_x": round(fp32_stats["median_ms"] / int8_stats["median_ms"], 2),
        },
        "embedding_fidelity": {
            "mean_cosine_vs_fp32": round(float(np.mean(cosines)), 6),
            "min_cosine_vs_fp32": round(float(np.min(cosines)), 6),
        },
        "retrieval_quality": {
            "queries": len(QUERIES),
            "identical_answers": identical_answers,
            "identical_citations": identical_citations,
            "identical_place_cards": identical_cards,
        },
    }

    print("=" * 78)
    print(f"MiniLM encoder weight quantization - {results['encoder']}")
    print("=" * 78)
    w = results["weights"]
    print(f"weights        : fp32 {w['fp32_mb']} MB -> int8 {w['int8_mb']} MB  ({w['reduction_x']}x smaller)")
    print(f"encode latency : fp32 median {fp32_stats['median_ms']} ms | int8 median {int8_stats['median_ms']} ms")
    print(f"                 int8 is {results['encode_latency']['int8_speedup_x']}x "
          f"{'faster' if results['encode_latency']['int8_speedup_x'] > 1 else 'slower'}")
    f = results["embedding_fidelity"]
    print(f"embeddings     : mean cosine vs fp32 {f['mean_cosine_vs_fp32']} | min {f['min_cosine_vs_fp32']}")
    r = results["retrieval_quality"]
    print(f"retrieval      : identical answers {r['identical_answers']}/{r['queries']} | "
          f"citations {r['identical_citations']}/{r['queries']} | "
          f"place cards {r['identical_place_cards']}/{r['queries']}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
