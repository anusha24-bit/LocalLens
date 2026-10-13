from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY_ENV_VALUES


@dataclass(frozen=True)
class Settings:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    docs_dir: Path
    database_path: Path
    embedding_matrix_path: Path
    embedding_int8_path: Path
    chunk_ids_path: Path
    qdrant_path: Path
    qdrant_collection: str
    image_cache_dir: Path
    ollama_base_url: str
    ollama_generation_model: str
    dense_model_name: str
    rerank_model_name: str
    vector_backend: str
    dense_int8: bool
    query_encoder_device: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    google_maps_api_key: str
    nps_api_key: str
    chunk_max_words: int
    chunk_overlap_words: int
    top_k: int
    candidate_k: int


def get_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    artifacts_dir = root / "artifacts"
    return Settings(
        project_root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        docs_dir=root / "docs",
        database_path=processed_dir / "locallens.db",
        embedding_matrix_path=artifacts_dir / "chunk_embeddings.npy",
        embedding_int8_path=artifacts_dir / "chunk_embeddings_int8.npz",
        chunk_ids_path=artifacts_dir / "chunk_ids.json",
        qdrant_path=artifacts_dir / "qdrant",
        qdrant_collection=os.getenv("LOCALLENS_QDRANT_COLLECTION", "locallens_chunks"),
        image_cache_dir=artifacts_dir / "images",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_generation_model=os.getenv(
            "LOCALLENS_OLLAMA_MODEL",
            "llama3.1:8b-instruct-q4_K_M",
        ),
        dense_model_name=os.getenv(
            "LOCALLENS_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        rerank_model_name=os.getenv(
            "LOCALLENS_RERANK_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        vector_backend=os.getenv("LOCALLENS_VECTOR_BACKEND", "qdrant").strip().lower() or "qdrant",
        # Off by default: the dense index is searched at full fp32 precision unless
        # LOCALLENS_DENSE_INT8 is set, in which case the compressed int8 index is used.
        dense_int8=_env_flag("LOCALLENS_DENSE_INT8", False),
        # Query encoding is batch-of-one on short text, where accelerator transfer
        # overhead exceeds the compute saved. Set to "" to let sentence-transformers
        # auto-select a device instead.
        query_encoder_device=os.getenv("LOCALLENS_QUERY_ENCODER_DEVICE", "cpu").strip(),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "LocalLens/0.1"),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", ""),
        nps_api_key=os.getenv("NPS_API_KEY", ""),
        chunk_max_words=int(os.getenv("LOCALLENS_CHUNK_MAX_WORDS", "180")),
        chunk_overlap_words=int(os.getenv("LOCALLENS_CHUNK_OVERLAP_WORDS", "40")),
        top_k=int(os.getenv("LOCALLENS_TOP_K", "6")),
        candidate_k=int(os.getenv("LOCALLENS_CANDIDATE_K", "24")),
    )
