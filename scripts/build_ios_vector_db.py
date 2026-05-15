#!/usr/bin/env python3
"""Build iOS vector SQLite DB from Chroma data.

Source:
    data/chroma_db (collection: guidelines)

Target:
    GuidelineAssistant/GuidelineAssistant/Resources/guidelines.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
import sys

import chromadb
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.turboquant import TurboQuantMSE


CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "guidelines"
RESOURCES_DIR = Path("GuidelineAssistant/GuidelineAssistant/Resources")
CODEBOOK_PATH = RESOURCES_DIR / "codebook.json"
OUTPUT_DB_PATH = RESOURCES_DIR / "guidelines.db"


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_image_refs(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=True)
    return json.dumps([str(value)], ensure_ascii=True)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs32 = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs32 = np.asarray(rhs, dtype=np.float32).reshape(-1)
    lhs_norm = float(np.linalg.norm(lhs32))
    rhs_norm = float(np.linalg.norm(rhs32))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return float(np.dot(lhs32, rhs32) / (lhs_norm * rhs_norm))


def _load_quantizer_from_codebook(path: Path) -> TurboQuantMSE:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dimension = int(payload["dimension"])
    bits = int(payload["bits_per_coordinate"])
    seed = int(payload["seed"])

    quantizer = TurboQuantMSE(d=dimension, b=bits, seed=seed)

    centroids = np.asarray(payload["centroids"], dtype=np.float32)
    rotation_col_major = np.asarray(payload["rotation_matrix_flat"], dtype=np.float32)
    rotation = rotation_col_major.reshape(dimension, dimension, order="F")

    # Force exact parity with serialized codebook used by iOS.
    quantizer.codebook = centroids
    quantizer.rotation = rotation
    quantizer.thresholds = ((centroids[:-1] + centroids[1:]) * 0.5).astype(np.float32)

    return quantizer


def _create_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS compressed_vectors;
        DROP TABLE IF EXISTS guideline_chunks;
        DROP TABLE IF EXISTS metadata;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE guideline_chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            page INTEGER NOT NULL,
            section TEXT,
            chunk_type TEXT NOT NULL,
            content TEXT NOT NULL,
            table_image TEXT,
            image_refs TEXT,
            chunk_index INTEGER NOT NULL
        );

        CREATE TABLE compressed_vectors (
            chunk_id INTEGER PRIMARY KEY REFERENCES guideline_chunks(id),
            indices BLOB NOT NULL,
            norm REAL NOT NULL
        );

        CREATE INDEX idx_chunks_source ON guideline_chunks(source);
        CREATE INDEX idx_chunks_type ON guideline_chunks(chunk_type);
        """
    )
    connection.commit()


def main() -> None:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing Chroma directory: {CHROMA_DIR}")

    if not CODEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Missing codebook.json at {CODEBOOK_PATH}. Run scripts/build_codebook.py first."
        )

    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB_PATH.exists():
        OUTPUT_DB_PATH.unlink()

    quantizer = _load_quantizer_from_codebook(CODEBOOK_PATH)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    total = int(collection.count())
    if total <= 0:
        raise ValueError(f"Collection '{COLLECTION_NAME}' is empty")

    connection = sqlite3.connect(str(OUTPUT_DB_PATH))
    try:
        _create_schema(connection)
        cursor = connection.cursor()

        chunk_rows: list[tuple[Any, ...]] = []
        vector_rows: list[tuple[Any, ...]] = []
        original_embeddings: list[np.ndarray] = []

        next_chunk_id = 1
        batch_size = 256

        for offset in tqdm(
            range(0, total, batch_size),
            desc="Exporting Chroma to SQLite",
            unit="batch",
        ):
            batch = collection.get(
                limit=min(batch_size, total - offset),
                offset=offset,
                include=["embeddings", "metadatas", "documents"],
            )

            embeddings_raw = batch.get("embeddings")
            metadatas_raw = batch.get("metadatas")
            documents_raw = batch.get("documents")

            if embeddings_raw is None:
                raise ValueError("Chroma returned no embeddings for batch")

            embeddings = (
                embeddings_raw.tolist()
                if isinstance(embeddings_raw, np.ndarray)
                else list(embeddings_raw)
            )
            metadatas = []
            if metadatas_raw is not None:
                metadatas = (
                    metadatas_raw.tolist()
                    if isinstance(metadatas_raw, np.ndarray)
                    else list(metadatas_raw)
                )
            documents = []
            if documents_raw is not None:
                documents = (
                    documents_raw.tolist()
                    if isinstance(documents_raw, np.ndarray)
                    else list(documents_raw)
                )

            for idx, embedding_values in enumerate(embeddings):
                embedding = np.asarray(embedding_values, dtype=np.float32).reshape(-1)
                if embedding.shape[0] != quantizer.d:
                    raise ValueError(
                        f"Expected {quantizer.d}-dim embedding, got {embedding.shape[0]}"
                    )

                metadata = {}
                if idx < len(metadatas) and metadatas[idx] is not None:
                    metadata = dict(metadatas[idx])

                document = documents[idx] if idx < len(documents) else ""
                content = _normalize_text(metadata.get("display_text", document))

                source = _normalize_text(metadata.get("source", "unknown"))
                page = _parse_int(metadata.get("page"), default=0)
                section = _normalize_optional_text(metadata.get("section"))
                chunk_type = _normalize_text(metadata.get("chunk_type", "text"))
                table_image = _normalize_optional_text(metadata.get("table_image"))
                image_refs = _normalize_image_refs(metadata.get("image_refs"))
                chunk_index = _parse_int(metadata.get("chunk_index"), default=offset + idx)

                indices, norm = quantizer.quantize(embedding)
                chunk_id = next_chunk_id
                next_chunk_id += 1

                chunk_rows.append(
                    (
                        chunk_id,
                        source,
                        page,
                        section,
                        chunk_type,
                        content,
                        table_image,
                        image_refs,
                        chunk_index,
                    )
                )
                vector_rows.append(
                    (
                        chunk_id,
                        sqlite3.Binary(np.asarray(indices, dtype=np.uint8).tobytes()),
                        float(norm),
                    )
                )
                original_embeddings.append(embedding)

        cursor.execute("BEGIN")
        cursor.executemany(
            """
            INSERT INTO guideline_chunks (
                id, source, page, section, chunk_type, content, table_image, image_refs, chunk_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )
        cursor.executemany(
            """
            INSERT INTO compressed_vectors (chunk_id, indices, norm)
            VALUES (?, ?, ?)
            """,
            vector_rows,
        )
        # Insert DB version metadata
        cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("version", "1"))
        connection.commit()

        cursor.execute("VACUUM")
        connection.commit()

        cursor.execute("SELECT COUNT(*) FROM guideline_chunks")
        db_count = int(cursor.fetchone()[0])
        if db_count != total:
            raise ValueError(f"Chunk count mismatch: sqlite={db_count}, chroma={total}")

        sample_size = min(10, total)
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(np.arange(1, total + 1), size=sample_size, replace=False)

        cosines: list[float] = []
        for chunk_id in sample_ids.tolist():
            cursor.execute(
                "SELECT indices, norm FROM compressed_vectors WHERE chunk_id = ?",
                (int(chunk_id),),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Missing compressed vector for chunk_id={chunk_id}")

            blob, norm = row
            indices = np.frombuffer(blob, dtype=np.uint8)
            reconstructed = quantizer.dequantize(indices, float(norm))
            original = original_embeddings[int(chunk_id) - 1]
            cosine = _cosine_similarity(reconstructed, original)
            cosines.append(cosine)

        min_cosine = min(cosines)
        mean_cosine = float(np.mean(cosines))
        if min_cosine <= 0.95:
            raise ValueError(
                f"Validation failed: minimum cosine similarity {min_cosine:.6f} <= 0.95"
            )

    finally:
        connection.close()

    db_size_mb = OUTPUT_DB_PATH.stat().st_size / (1024 * 1024)

    print("iOS vector DB build complete")
    print(f"  Output: {OUTPUT_DB_PATH}")
    print(f"  Chroma chunks: {total}")
    print(f"  SQLite chunks: {db_count}")
    print(f"  Sample cosine min: {min_cosine:.6f}")
    print(f"  Sample cosine mean: {mean_cosine:.6f}")
    print(f"  Database size: {db_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
