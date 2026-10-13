from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from locallens.schemas import ChunkRecord, SearchResult


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def match_filters(chunk: ChunkRecord, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    if filters.get("location") and chunk.location != filters["location"]:
        return False
    if filters.get("topic") and chunk.topic != filters["topic"]:
        return False
    return True


class BM25Retriever:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [tokenize(_chunk_text(chunk)) for chunk in chunks]
        self.model = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._length_norm: np.ndarray | None = None
        self._build_postings()

    def _build_postings(self) -> None:
        """Invert the corpus so scoring touches only documents containing a query term.

        rank_bm25's get_scores() walks all N documents for every query term via a
        Python dict lookup per document. Documents that lack the term contribute
        exactly zero, so restricting the update to the term's posting list is
        mathematically identical while scaling with matches instead of corpus size.
        The BM25 parameters are taken from the fitted BM25Okapi model so scores stay
        consistent with rank_bm25.
        """
        if self.model is None:
            return
        doc_len = np.asarray(self.model.doc_len, dtype=np.float64)
        self._length_norm = self.model.k1 * (
            1 - self.model.b + self.model.b * doc_len / self.model.avgdl
        )
        collected: dict[str, list[tuple[int, float]]] = {}
        for doc_index, frequencies in enumerate(self.model.doc_freqs):
            for term, frequency in frequencies.items():
                collected.setdefault(term, []).append((doc_index, frequency))
        self._postings = {
            term: (
                np.fromiter((item[0] for item in entries), dtype=np.int32, count=len(entries)),
                np.fromiter((item[1] for item in entries), dtype=np.float64, count=len(entries)),
            )
            for term, entries in collected.items()
        }

    def _score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        if self.model is None or self._length_norm is None:
            return scores
        for term in query_tokens:
            posting = self._postings.get(term)
            if posting is None:
                continue
            indices, frequencies = posting
            idf = self.model.idf.get(term, 0.0)
            if not idf:
                continue
            contribution = idf * (frequencies * (self.model.k1 + 1)) / (
                frequencies + self._length_norm[indices]
            )
            scores[indices] += contribution
        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if not self.model:
            return []
        scores = self._score(tokenize(query))
        # Non-positive scores are always skipped below, so rank only the positives.
        # A stable sort preserves the original index order among ties, matching the
        # previous sorted(enumerate(...), reverse=True) behaviour.
        candidates = np.flatnonzero(scores > 0)
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        results: list[SearchResult] = []
        for index in ranked:
            chunk = self.chunks[int(index)]
            if not match_filters(chunk, filters):
                continue
            score = float(scores[int(index)])
            results.append(
                SearchResult(
                    chunk=chunk,
                    bm25_score=score,
                    final_score=score,
                )
            )
            if len(results) == top_k:
                break
        return results


def _chunk_text(chunk: ChunkRecord) -> str:
    return " ".join(
        [
            chunk.title,
            chunk.location,
            chunk.topic,
            chunk.source_type.replace("_", " "),
            chunk.passage_text,
        ]
    )

