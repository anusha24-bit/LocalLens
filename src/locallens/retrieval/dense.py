from __future__ import annotations

import json

import numpy as np

from locallens.config import Settings
from locallens.retrieval.bm25 import match_filters, tokenize
from locallens.schemas import ChunkRecord, SearchResult
from locallens.utils import ensure_parent

DEFAULT_QDRANT_BATCH_SIZE = 256
QDRANT_SEARCH_MIN_LIMIT = 64
INT8_BACKEND_SUFFIX = "+int8"
INT8_MAX = 127


def quantize_matrix_int8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vector symmetric scalar quantization of a dense index to int8.

    Every row gets its own scale (its max absolute value) so the full int8 range is
    used, which keeps cosine ranking close to fp32 while storing the index at roughly
    a quarter of the size. Returns the int8 matrix plus the per-row fp32 scales needed
    to recover approximate dot products.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.int8), np.zeros((0,), dtype=np.float32)
    scales = np.abs(matrix).max(axis=1) / float(INT8_MAX)
    scales[scales == 0] = 1.0
    quantized = np.clip(np.round(matrix / scales[:, None]), -INT8_MAX, INT8_MAX).astype(np.int8)
    return quantized, scales.astype(np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbeddingBackend:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row_index, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in tokenize(text):
                index = hash(token) % self.dimensions
                sign = 1.0 if hash(f"{token}:sign") % 2 == 0 else -1.0
                counts[index] = counts.get(index, 0.0) + sign
            for index, value in counts.items():
                matrix[row_index, index] = value
        return _normalize(matrix)


class SentenceTransformerBackend:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype=np.float32)


class QdrantUnavailable(RuntimeError):
    pass


def build_dense_embeddings(
    settings: Settings,
    chunks: list[ChunkRecord],
) -> tuple[np.ndarray, list[str], str]:
    embedding_backend = _make_embedding_backend(settings)
    embedding_backend_name = getattr(embedding_backend, "model_name", "hash")
    matrix = _load_cached_matrix(settings, chunks, embedding_backend_name)
    if matrix is None:
        chunk_texts = [_document_text(chunk) for chunk in chunks]
        matrix = embedding_backend.encode(chunk_texts)

    if settings.vector_backend == "qdrant":
        try:
            _build_qdrant_index(settings, chunks, matrix)
            backend_name = f"qdrant:{embedding_backend_name}"
            _write_chunk_manifest(settings, chunks, backend_name)
            return matrix, [chunk.chunk_id for chunk in chunks], backend_name
        except Exception:
            pass

    ensure_parent(settings.embedding_matrix_path)
    np.save(settings.embedding_matrix_path, matrix)
    backend_name = f"numpy:{embedding_backend_name}"
    _write_chunk_manifest(settings, chunks, backend_name)
    return matrix, [chunk.chunk_id for chunk in chunks], backend_name


class DenseRetriever:
    def __init__(
        self,
        chunks: list[ChunkRecord],
        settings: Settings,
    ) -> None:
        self.chunks = chunks
        self.settings = settings
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_by_point_id = {index: chunk for index, chunk in enumerate(chunks)}
        self.backend_name = "numpy:hash"
        self.query_backend = None
        self.client = None
        self.using_qdrant = self.settings.vector_backend == "qdrant"
        # Off by default (Settings.dense_int8); when on, the fp32 matrix is replaced by a
        # compressed int8 index plus per-vector scales.
        self.use_int8 = bool(self.settings.dense_int8)
        self.int8_matrix: np.ndarray | None = None
        self.int8_scales: np.ndarray | None = None
        self.matrix, self.ids = self._load_or_build()

    def _load_or_build(self) -> tuple[np.ndarray, list[str]]:
        if self.using_qdrant and self._load_qdrant_or_build():
            return np.empty((0, 0), dtype=np.float32), []

        self.using_qdrant = False
        if self.settings.chunk_ids_path.exists() and (
            self.settings.embedding_matrix_path.exists() or self._int8_cache_is_usable()
        ):
            with self.settings.chunk_ids_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            ids = list(payload.get("chunk_ids", []))
            self.backend_name = str(payload.get("backend", "numpy:hash"))
            self.query_backend = self._make_query_backend(_embedding_backend_name(self.backend_name))
            if self.use_int8:
                cached = self._load_int8_cache(len(ids))
                if cached is not None:
                    # Searching straight off the compressed index: fp32 is never loaded.
                    self._activate_int8(*cached)
                    return np.empty((0, 0), dtype=np.float32), ids
            matrix = np.load(self.settings.embedding_matrix_path)
            return self._maybe_quantize(matrix), ids

        matrix, ids, backend_name = build_dense_embeddings(self.settings, self.chunks)
        self.backend_name = backend_name
        self.query_backend = self._make_query_backend(_embedding_backend_name(backend_name))
        if backend_name.startswith("qdrant:"):
            self.using_qdrant = True
            self.client = self._connect_qdrant()
            return np.empty((0, 0), dtype=np.float32), []
        return self._maybe_quantize(matrix), ids

    def _maybe_quantize(self, matrix: np.ndarray) -> np.ndarray:
        """Compress the freshly loaded fp32 index to int8 when the setting is on."""
        if not self.use_int8 or matrix.size == 0:
            return matrix
        quantized, scales = quantize_matrix_int8(matrix)
        self._save_int8_cache(quantized, scales)
        self._activate_int8(quantized, scales)
        return np.empty((0, 0), dtype=np.float32)

    def _activate_int8(self, quantized: np.ndarray, scales: np.ndarray) -> None:
        self.int8_matrix = quantized
        self.int8_scales = scales
        if not self.backend_name.endswith(INT8_BACKEND_SUFFIX):
            self.backend_name = f"{self.backend_name}{INT8_BACKEND_SUFFIX}"

    def _int8_cache_is_usable(self) -> bool:
        """True when a cached int8 index exists and is not older than the fp32 index."""
        if not self.use_int8 or not self.settings.embedding_int8_path.exists():
            return False
        if not self.settings.embedding_matrix_path.exists():
            return True
        return (
            self.settings.embedding_int8_path.stat().st_mtime
            >= self.settings.embedding_matrix_path.stat().st_mtime
        )

    def _load_int8_cache(self, expected_rows: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._int8_cache_is_usable():
            return None
        try:
            with np.load(self.settings.embedding_int8_path) as payload:
                quantized = payload["quantized"]
                scales = payload["scales"]
        except Exception:
            return None
        if quantized.shape[0] != expected_rows or scales.shape[0] != expected_rows:
            return None
        return quantized.astype(np.int8, copy=False), scales.astype(np.float32, copy=False)

    def _save_int8_cache(self, quantized: np.ndarray, scales: np.ndarray) -> None:
        try:
            ensure_parent(self.settings.embedding_int8_path)
            np.savez(self.settings.embedding_int8_path, quantized=quantized, scales=scales)
        except Exception:
            pass

    def _dense_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine scores against whichever index precision is active.

        For the int8 index, dequantization folds out of the dot product:
        (q8 * scale) . query == scale * (q8 . query), so only one scale multiply is
        needed per chunk instead of materializing an fp32 copy of the matrix.
        """
        if self.int8_matrix is not None and self.int8_scales is not None:
            raw = self.int8_matrix @ np.asarray(query_vector, dtype=np.float32)
            return np.asarray(raw, dtype=np.float32) * self.int8_scales
        return self.matrix @ query_vector

    @property
    def index_nbytes(self) -> int:
        """On-disk/in-memory footprint of the active dense index, for benchmarking."""
        if self.int8_matrix is not None and self.int8_scales is not None:
            return int(self.int8_matrix.nbytes + self.int8_scales.nbytes)
        return int(self.matrix.nbytes)

    def _load_qdrant_or_build(self) -> bool:
        client = self._connect_qdrant()
        if client is None:
            return False
        self.client = client
        if self.settings.chunk_ids_path.exists():
            with self.settings.chunk_ids_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            stored_backend = str(payload.get("backend", ""))
            if stored_backend.startswith("qdrant:") and self._collection_exists():
                self.backend_name = stored_backend
                self.query_backend = self._make_query_backend(_embedding_backend_name(stored_backend))
                return True

        _, _, backend_name = build_dense_embeddings(self.settings, self.chunks)
        if not backend_name.startswith("qdrant:"):
            return False
        self.backend_name = backend_name
        self.query_backend = self._make_query_backend(_embedding_backend_name(backend_name))
        return True

    def _connect_qdrant(self):
        try:
            from qdrant_client import QdrantClient
        except Exception:
            return None
        self.settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(self.settings.qdrant_path))

    def _collection_exists(self) -> bool:
        if self.client is None:
            return False
        try:
            return bool(self.client.collection_exists(self.settings.qdrant_collection))
        except Exception:
            return False

    def _make_query_backend(self, backend_name: str):
        if backend_name == self.settings.dense_model_name:
            try:
                return SentenceTransformerBackend(self.settings.dense_model_name)
            except Exception:
                self.backend_name = "numpy:hash"
        return HashingEmbeddingBackend()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if self.using_qdrant:
            return self._search_qdrant(query, top_k=top_k, filters=filters)
        if self.int8_matrix is None and self.matrix.size == 0:
            return []
        if self.query_backend is None:
            self.query_backend = self._make_query_backend(_embedding_backend_name(self.backend_name))
        query_vector = self.query_backend.encode([query])[0]
        scores = self._dense_scores(query_vector)
        ranked_indices = np.argsort(scores)[::-1]
        results: list[SearchResult] = []
        for index in ranked_indices:
            chunk = self.chunk_by_id.get(self.ids[int(index)])
            if chunk is None or not match_filters(chunk, filters):
                continue
            score = float(scores[int(index)])
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=score,
                    final_score=score,
                )
            )
            if len(results) == top_k:
                break
        return results

    def _search_qdrant(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if self.client is None:
            return []
        if self.query_backend is None:
            self.query_backend = self._make_query_backend(_embedding_backend_name(self.backend_name))
        query_vector = self.query_backend.encode([query])[0]
        limit = max(top_k * 8, QDRANT_SEARCH_MIN_LIMIT if filters else top_k)
        try:
            query_filter = _make_qdrant_filter(filters)
            hits = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=query_filter,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            return []

        results: list[SearchResult] = []
        for hit in hits:
            chunk = self.chunk_by_point_id.get(int(hit.id))
            if chunk is None or not match_filters(chunk, filters):
                continue
            score = float(hit.score)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=score,
                    final_score=score,
                )
            )
            if len(results) == top_k:
                break
        return results


def _document_text(chunk: ChunkRecord) -> str:
    return " ".join(
        [
            chunk.title,
            chunk.location,
            chunk.topic,
            chunk.source_type.replace("_", " "),
            chunk.passage_text,
        ]
    )


def _make_embedding_backend(settings: Settings):
    try:
        return SentenceTransformerBackend(settings.dense_model_name)
    except Exception:
        return HashingEmbeddingBackend()


def _load_cached_matrix(
    settings: Settings,
    chunks: list[ChunkRecord],
    embedding_backend_name: str,
) -> np.ndarray | None:
    if not settings.embedding_matrix_path.exists() or not settings.chunk_ids_path.exists():
        return None
    try:
        with settings.chunk_ids_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        backend_name = str(payload.get("backend", ""))
        cached_ids = list(payload.get("chunk_ids", []))
    except Exception:
        return None

    if len(cached_ids) != len(chunks):
        return None
    if cached_ids != [chunk.chunk_id for chunk in chunks]:
        return None
    if _embedding_backend_name(backend_name) != embedding_backend_name:
        return None

    try:
        return np.load(settings.embedding_matrix_path)
    except Exception:
        return None


def _write_chunk_manifest(settings: Settings, chunks: list[ChunkRecord], backend_name: str) -> None:
    ensure_parent(settings.chunk_ids_path)
    with settings.chunk_ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"chunk_ids": [chunk.chunk_id for chunk in chunks], "backend": backend_name}, handle)


def _build_qdrant_index(settings: Settings, chunks: list[ChunkRecord], matrix: np.ndarray) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception as exc:
        raise QdrantUnavailable("qdrant-client is not installed") from exc

    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.qdrant_path))
    client.recreate_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=int(matrix.shape[1]), distance=Distance.COSINE),
    )

    for start in range(0, len(chunks), DEFAULT_QDRANT_BATCH_SIZE):
        stop = min(start + DEFAULT_QDRANT_BATCH_SIZE, len(chunks))
        points = [
            PointStruct(
                id=index,
                vector=matrix[index].tolist(),
                payload={
                    "chunk_id": chunks[index].chunk_id,
                    "location": chunks[index].location,
                    "topic": chunks[index].topic,
                    "source_type": chunks[index].source_type,
                },
            )
            for index in range(start, stop)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def _embedding_backend_name(backend_name: str) -> str:
    if ":" in backend_name:
        return backend_name.split(":", 1)[1]
    return backend_name


def _make_qdrant_filter(filters: dict[str, str] | None):
    if not filters:
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except Exception:
        return None

    must = []
    for key in ("location", "topic", "source_type"):
        value = filters.get(key)
        if value:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    if not must:
        return None
    return Filter(must=must)
