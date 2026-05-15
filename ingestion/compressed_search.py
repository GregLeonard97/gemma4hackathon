"""Compressed vector search using TurboQuantMSE embeddings.

This module provides:
- CompressedVectorStore: in-memory compressed retrieval store
- compress_collection: helper to build from an existing Chroma collection

Storage strategy:
- Quantized indices are bit-packed into uint8 rows.
- Per-vector L2 norms are stored as float32 for exact reconstruction support.
- Rotation matrix and Lloyd-Max codebook are persisted for reproducibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingestion.abbreviations import expand_query
from ingestion.turboquant import TurboQuantMSE

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "guidelines"


_EMBEDDING_FN_CACHE: Dict[str, SentenceTransformerEmbeddingFunction] = {}


def _get_embedding_fn(model_name: str) -> SentenceTransformerEmbeddingFunction:
    fn = _EMBEDDING_FN_CACHE.get(model_name)
    if fn is None:
        fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
        _EMBEDDING_FN_CACHE[model_name] = fn
    return fn


def _pack_indices(indices: np.ndarray, b: int) -> np.ndarray:
    """Pack (n, d) quantization indices into uint8 rows."""
    idx = np.asarray(indices, dtype=np.uint8)
    if idx.ndim != 2:
        raise ValueError(f"Expected 2D indices array, got shape {idx.shape}")
    if b <= 0 or b > 8:
        raise ValueError("Bit width b must be in [1, 8]")

    levels = 1 << b
    if np.any(idx >= levels):
        raise ValueError("Found index outside representable range for the given bit width")

    bit_slices = ((idx[..., None] >> np.arange(b, dtype=np.uint8)) & 1).astype(np.uint8)
    flat_bits = bit_slices.reshape(idx.shape[0], idx.shape[1] * b)
    return np.packbits(flat_bits, axis=1, bitorder="little")


def _unpack_indices(packed: np.ndarray, d: int, b: int) -> np.ndarray:
    """Unpack uint8 rows into (n, d) quantization indices."""
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D packed array, got shape {arr.shape}")
    if d <= 0:
        raise ValueError("d must be positive")
    if b <= 0 or b > 8:
        raise ValueError("Bit width b must be in [1, 8]")

    total_bits = d * b
    unpacked = np.unpackbits(arr, axis=1, count=total_bits, bitorder="little")
    bit_groups = unpacked.reshape((arr.shape[0], d, b))
    weights = (1 << np.arange(b, dtype=np.uint16)).reshape(1, 1, b)
    indices = np.sum(bit_groups.astype(np.uint16) * weights, axis=2).astype(np.uint8)
    return indices


def _extract_collection_embeddings(collection, batch_size: int = 512):
    """Extract embeddings + metadata + documents from a populated Chroma collection."""
    total = collection.count()
    embeddings: List[Any] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=min(batch_size, total - offset),
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )

        batch_embeddings_raw = batch.get("embeddings")
        batch_documents_raw = batch.get("documents")
        batch_metadatas_raw = batch.get("metadatas")

        batch_embeddings = list(batch_embeddings_raw) if batch_embeddings_raw is not None else []
        batch_documents = list(batch_documents_raw) if batch_documents_raw is not None else []
        batch_metadatas = list(batch_metadatas_raw) if batch_metadatas_raw is not None else []

        embeddings.extend(batch_embeddings)
        documents.extend(batch_documents)
        metadatas.extend(batch_metadatas)

    if not embeddings:
        raise ValueError("Collection contains no embeddable vectors")

    emb_array = np.asarray(embeddings, dtype=np.float32)
    if emb_array.ndim != 2:
        raise ValueError(f"Unexpected embedding shape from collection: {emb_array.shape}")

    records: List[Dict[str, Any]] = []
    for doc, meta in zip(documents, metadatas):
        md = meta or {}
        records.append(
            {
                "content": md.get("display_text", doc),
                "source": md.get("source"),
                "page": md.get("page"),
                "section": md.get("section"),
                "chunk_type": md.get("chunk_type", "text"),
            }
        )

    return emb_array, records


@dataclass
class CompressedVectorStore:
    """TurboQuant-compressed searchable vector store."""

    quantizer: TurboQuantMSE
    packed_indices: np.ndarray
    norms: np.ndarray
    metadata: List[Dict[str, Any]]
    embedding_model: str = EMBEDDING_MODEL

    def __post_init__(self) -> None:
        self.packed_indices = np.asarray(self.packed_indices, dtype=np.uint8)
        self.norms = np.asarray(self.norms, dtype=np.float32).reshape(-1)

        if self.packed_indices.ndim != 2:
            raise ValueError("packed_indices must be a 2D uint8 array")
        if self.packed_indices.shape[0] != self.norms.shape[0]:
            raise ValueError("packed_indices rows must match norms length")
        if len(self.metadata) != self.packed_indices.shape[0]:
            raise ValueError("metadata length must match number of vectors")

        self._indices_cache: Optional[np.ndarray] = None
        self._rotated_vectors_cache: Optional[np.ndarray] = None
        self._rotated_norms_cache: Optional[np.ndarray] = None
        self._embedding_fn = _get_embedding_fn(self.embedding_model)

    @property
    def count(self) -> int:
        return int(self.packed_indices.shape[0])

    @property
    def d(self) -> int:
        return int(self.quantizer.d)

    @property
    def b(self) -> int:
        return int(self.quantizer.b)

    @property
    def packed_dim(self) -> int:
        return int(self.packed_indices.shape[1])

    def _get_indices(self) -> np.ndarray:
        if self._indices_cache is None:
            self._indices_cache = _unpack_indices(
                self.packed_indices,
                d=self.d,
                b=self.b,
            )
        return self._indices_cache

    def _get_rotated_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._rotated_vectors_cache is None or self._rotated_norms_cache is None:
            indices = self._get_indices()
            rotated_vectors = self.quantizer.codebook[indices.astype(np.int64)].astype(np.float32)
            norms = np.linalg.norm(rotated_vectors, axis=1).astype(np.float32)
            norms = np.where(norms > 0.0, norms, 1.0).astype(np.float32)

            self._rotated_vectors_cache = rotated_vectors
            self._rotated_norms_cache = norms

        return self._rotated_vectors_cache, self._rotated_norms_cache

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search compressed vectors and return Chroma-like result dicts."""
        if top_k <= 0 or self.count == 0:
            return []

        expanded_query = expand_query(query_text)
        query_embedding = np.asarray(
            self._embedding_fn([expanded_query])[0],
            dtype=np.float32,
        )

        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []

        query_unit = query_embedding / query_norm
        query_rot = query_unit @ self.quantizer.rotation.T

        rotated_vectors, denom = self._get_rotated_vectors()
        similarities = (rotated_vectors @ query_rot) / denom

        k = min(int(top_k), self.count)
        if k == self.count:
            top_indices = np.argsort(similarities)[::-1]
        else:
            candidate = np.argpartition(similarities, -k)[-k:]
            top_indices = candidate[np.argsort(similarities[candidate])[::-1]]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            meta = self.metadata[int(idx)]
            results.append(
                {
                    "content": meta.get("content"),
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "similarity": float(similarities[int(idx)]),
                }
            )
        return results

    def save(self, directory: str) -> Dict[str, Any]:
        """Persist compressed vectors, metadata, and quantizer state to disk."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        vectors_path = target / "vectors_uint8.bin"
        norms_path = target / "norms_f32.bin"
        state_path = target / "quantizer_state.npz"
        metadata_path = target / "metadata.jsonl"
        manifest_path = target / "manifest.json"

        self.packed_indices.tofile(vectors_path)
        self.norms.astype(np.float32).tofile(norms_path)
        np.savez(
            state_path,
            rotation=self.quantizer.rotation.astype(np.float32),
            codebook=self.quantizer.codebook.astype(np.float32),
        )

        with open(metadata_path, "w", encoding="utf-8") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        manifest = {
            "d": self.d,
            "b": self.b,
            "seed": int(self.quantizer.seed),
            "count": self.count,
            "packed_dim": self.packed_dim,
            "embedding_model": self.embedding_model,
            "files": {
                "vectors": vectors_path.name,
                "norms": norms_path.name,
                "state": state_path.name,
                "metadata": metadata_path.name,
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        payload_bytes = os.path.getsize(vectors_path) + os.path.getsize(norms_path)
        total_bytes = (
            payload_bytes
            + os.path.getsize(state_path)
            + os.path.getsize(metadata_path)
            + os.path.getsize(manifest_path)
        )

        return {
            "directory": str(target),
            "payload_bytes": int(payload_bytes),
            "total_bytes": int(total_bytes),
            "files": {
                "vectors": str(vectors_path),
                "norms": str(norms_path),
                "state": str(state_path),
                "metadata": str(metadata_path),
                "manifest": str(manifest_path),
            },
        }

    @classmethod
    def load(cls, directory: str) -> "CompressedVectorStore":
        """Load a CompressedVectorStore from disk."""
        base = Path(directory)
        manifest_path = base / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

        d = int(manifest["d"])
        b = int(manifest["b"])
        seed = int(manifest.get("seed", 42))
        count = int(manifest["count"])
        packed_dim = int(manifest["packed_dim"])
        embedding_model = manifest.get("embedding_model", EMBEDDING_MODEL)

        file_map = manifest["files"]
        vectors_path = base / file_map["vectors"]
        norms_path = base / file_map["norms"]
        state_path = base / file_map["state"]
        metadata_path = base / file_map["metadata"]

        packed = np.fromfile(vectors_path, dtype=np.uint8)
        packed = packed.reshape(count, packed_dim)

        norms = np.fromfile(norms_path, dtype=np.float32)
        if norms.shape[0] != count:
            raise ValueError("Norm vector length does not match manifest count")

        state = np.load(state_path)
        rotation = np.asarray(state["rotation"], dtype=np.float32)
        codebook = np.asarray(state["codebook"], dtype=np.float32)

        metadata: List[Dict[str, Any]] = []
        with open(metadata_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                metadata.append(json.loads(line))

        if len(metadata) != count:
            raise ValueError("Metadata row count does not match manifest count")

        quantizer = TurboQuantMSE(d=d, b=b, seed=seed)
        quantizer.rotation = rotation
        quantizer.codebook = codebook
        quantizer.thresholds = ((codebook[:-1] + codebook[1:]) * 0.5).astype(np.float32)

        return cls(
            quantizer=quantizer,
            packed_indices=packed,
            norms=norms,
            metadata=metadata,
            embedding_model=embedding_model,
        )


def compress_collection(
    collection,
    bit_width: int,
    seed: int = 42,
    embedding_model: str = EMBEDDING_MODEL,
) -> CompressedVectorStore:
    """Build a CompressedVectorStore from an existing Chroma collection."""
    embeddings, metadata_records = _extract_collection_embeddings(collection)

    quantizer = TurboQuantMSE(d=embeddings.shape[1], b=bit_width, seed=seed)
    indices, norms = quantizer.quantize_batch(embeddings)
    packed = _pack_indices(indices, b=bit_width)

    return CompressedVectorStore(
        quantizer=quantizer,
        packed_indices=packed,
        norms=norms,
        metadata=metadata_records,
        embedding_model=embedding_model,
    )