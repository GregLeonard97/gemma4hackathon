#!/usr/bin/env python3
"""Validate generated iOS runtime resources."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Any
import sys
import subprocess

import chromadb
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.turboquant import TurboQuantMSE


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_LENGTH = 512
COLLECTION_NAME = "guidelines"

CHROMA_DIR = Path("data/chroma_db")
RESOURCES_DIR = Path("GuidelineAssistant/GuidelineAssistant/Resources")

MLPACKAGE_PATH = RESOURCES_DIR / "BGESmall.mlpackage"
VOCAB_PATH = RESOURCES_DIR / "bge_vocab.txt"
TOKENIZER_CONFIG_PATH = RESOURCES_DIR / "bge_tokenizer_config.json"
CODEBOOK_PATH = RESOURCES_DIR / "codebook.json"
DB_PATH = RESOURCES_DIR / "guidelines.db"
TABLES_DIR = RESOURCES_DIR / "extracted_tables"
GUIDELINES_DIR = RESOURCES_DIR / "Guidelines"


class BGEEmbedder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(torch.long)
        mask = attention_mask.to(torch.long)
        outputs = self.model(input_ids=ids, attention_mask=mask)
        cls = outputs.last_hidden_state[:, 0]
        return F.normalize(cls, p=2, dim=1)


def _require_path(path: Path, is_dir: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    if is_dir and not path.is_dir():
        raise ValueError(f"Expected directory, got file: {path}")
    if not is_dir and not path.is_file():
        raise ValueError(f"Expected file, got directory: {path}")


def _directory_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _directory_file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


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

    quantizer.codebook = centroids
    quantizer.rotation = rotation
    quantizer.thresholds = ((centroids[:-1] + centroids[1:]) * 0.5).astype(np.float32)
    return quantizer


def _validate_presence() -> None:
    _require_path(MLPACKAGE_PATH, is_dir=True)
    _require_path(VOCAB_PATH, is_dir=False)
    _require_path(TOKENIZER_CONFIG_PATH, is_dir=False)
    _require_path(CODEBOOK_PATH, is_dir=False)
    _require_path(DB_PATH, is_dir=False)
    _require_path(TABLES_DIR, is_dir=True)
    _require_path(GUIDELINES_DIR, is_dir=True)

    if _directory_file_count(TABLES_DIR) == 0:
        raise ValueError(f"{TABLES_DIR} is empty")
    if _directory_file_count(GUIDELINES_DIR) == 0:
        raise ValueError(f"{GUIDELINES_DIR} is empty")


def _validate_vocab() -> dict[str, Any]:
    lines = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    count = len(lines)
    if count < 30000:
        raise ValueError(f"Unexpected vocab size: {count}")

    if not lines:
        raise ValueError("Vocabulary file is empty")

    token_0 = lines[0]
    token_101 = lines[101] if count > 101 else None
    token_102 = lines[102] if count > 102 else None

    if token_0 != "[PAD]":
        raise ValueError(f"Expected token id 0 to be [PAD], found {token_0}")
    if token_101 != "[CLS]":
        raise ValueError(f"Expected token id 101 to be [CLS], found {token_101}")
    if token_102 != "[SEP]":
        raise ValueError(f"Expected token id 102 to be [SEP], found {token_102}")

    return {
        "count": count,
        "token_0": token_0,
        "token_101": token_101,
        "token_102": token_102,
    }


def _load_chroma_embeddings() -> list[np.ndarray]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    total = int(collection.count())
    if total <= 0:
        raise ValueError(f"Collection '{COLLECTION_NAME}' is empty")

    embeddings: list[np.ndarray] = []
    batch_size = 256
    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=min(batch_size, total - offset),
            offset=offset,
            include=["embeddings"],
        )
        raw = batch.get("embeddings")
        if raw is None:
            raise ValueError("Missing embeddings from Chroma batch")
        rows = raw.tolist() if isinstance(raw, np.ndarray) else list(raw)
        for row in rows:
            emb = np.asarray(row, dtype=np.float32).reshape(-1)
            embeddings.append(emb)
    return embeddings


def _validate_db_against_chroma(quantizer: TurboQuantMSE) -> dict[str, Any]:
    chroma_embeddings = _load_chroma_embeddings()

    connection = sqlite3.connect(str(DB_PATH))
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM guideline_chunks")
        chunk_count = int(cursor.fetchone()[0])

        cursor.execute("SELECT COUNT(*) FROM compressed_vectors")
        vector_count = int(cursor.fetchone()[0])

        if chunk_count != len(chroma_embeddings):
            raise ValueError(
                f"Chunk count mismatch: sqlite={chunk_count}, chroma={len(chroma_embeddings)}"
            )
        if vector_count != chunk_count:
            raise ValueError(
                f"Vector row mismatch: compressed_vectors={vector_count}, chunks={chunk_count}"
            )

        sample_size = min(10, chunk_count)
        sample_ids = random.sample(range(1, chunk_count + 1), sample_size)
        cosines: list[float] = []

        for chunk_id in sample_ids:
            cursor.execute(
                "SELECT indices, norm FROM compressed_vectors WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Missing compressed vector for chunk_id={chunk_id}")

            blob, norm = row
            indices = np.frombuffer(blob, dtype=np.uint8)
            reconstructed = quantizer.dequantize(indices, float(norm))
            original = chroma_embeddings[chunk_id - 1]
            cosines.append(_cosine_similarity(reconstructed, original))

        min_cosine = min(cosines)
        mean_cosine = float(np.mean(cosines))
        if min_cosine <= 0.95:
            raise ValueError(
                f"Compression validation failed: min cosine {min_cosine:.6f} <= 0.95"
            )

        return {
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "sample_size": sample_size,
            "min_cosine": min_cosine,
            "mean_cosine": mean_cosine,
        }
    finally:
        connection.close()


def _validate_coreml() -> dict[str, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        torch_model = BGEEmbedder(AutoModel.from_pretrained(MODEL_NAME)).eval()
        mlmodel = ct.models.MLModel(str(MLPACKAGE_PATH))

        sample = tokenizer(
            "neonatal sepsis early-onset",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.no_grad():
            torch_out = torch_model(
                sample["input_ids"].to(torch.int32),
                sample["attention_mask"].to(torch.int32),
            ).cpu().numpy()

        prediction = mlmodel.predict(
            {
                "input_ids": sample["input_ids"].cpu().numpy().astype(np.int32),
                "attention_mask": sample["attention_mask"].cpu().numpy().astype(np.int32),
            }
        )

        if "embedding" in prediction:
            coreml_out = np.asarray(prediction["embedding"], dtype=np.float32)
        else:
            first_value = next(iter(prediction.values()))
            coreml_out = np.asarray(first_value, dtype=np.float32)

        cosine = _cosine_similarity(torch_out, coreml_out)
        if cosine <= 0.9999:
            raise ValueError(f"Core ML validation failed: cosine {cosine:.8f} <= 0.9999")

        return {"cosine": cosine}
    except Exception as error:
        fallback_python = REPO_ROOT / ".venv-coreml" / "bin" / "python"
        if not fallback_python.exists():
            raise RuntimeError(
                "Core ML validation failed in current environment and no fallback "
                f"interpreter found at {fallback_python}: {error}"
            ) from error

        code = f"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = {MODEL_NAME!r}
MAX_LENGTH = {MAX_LENGTH}
MLPACKAGE_PATH = {str(MLPACKAGE_PATH)!r}

class BGEEmbedder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return F.normalize(cls, p=2, dim=1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
torch_model = BGEEmbedder(AutoModel.from_pretrained(MODEL_NAME)).eval()
mlmodel = ct.models.MLModel(MLPACKAGE_PATH)

sample = tokenizer(
    "neonatal sepsis early-onset",
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)

with torch.no_grad():
    torch_out = torch_model(
        sample["input_ids"].to(torch.int32),
        sample["attention_mask"].to(torch.int32),
    ).cpu().numpy()

prediction = mlmodel.predict({{
    "input_ids": sample["input_ids"].cpu().numpy().astype(np.int32),
    "attention_mask": sample["attention_mask"].cpu().numpy().astype(np.int32),
}})

if "embedding" in prediction:
    coreml_out = np.asarray(prediction["embedding"], dtype=np.float32)
else:
    first_value = next(iter(prediction.values()))
    coreml_out = np.asarray(first_value, dtype=np.float32)

lhs = torch_out.astype(np.float32).reshape(-1)
rhs = coreml_out.astype(np.float32).reshape(-1)
cosine = float(np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs)))
print(json.dumps({{"cosine": cosine}}))
"""

        result = subprocess.run(
            [str(fallback_python), "-c", code],
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Core ML validation failed in current and fallback environments. "
                f"Current error: {error}; Fallback stderr: {result.stderr.strip()}"
            ) from error

        stdout = result.stdout.strip().splitlines()
        if not stdout:
            raise RuntimeError("Fallback Core ML validation returned no output") from error

        payload = json.loads(stdout[-1])
        cosine = float(payload["cosine"])
        if cosine <= 0.9999:
            raise ValueError(
                f"Core ML validation failed via fallback: cosine {cosine:.8f} <= 0.9999"
            )

        return {"cosine": cosine, "fallback_python": str(fallback_python)}


def _build_size_summary() -> list[tuple[str, float]]:
    rows = [
        ("BGESmall.mlpackage", _directory_size_bytes(MLPACKAGE_PATH) / (1024 * 1024)),
        ("bge_vocab.txt", VOCAB_PATH.stat().st_size / (1024 * 1024)),
        (
            "bge_tokenizer_config.json",
            TOKENIZER_CONFIG_PATH.stat().st_size / (1024 * 1024),
        ),
        ("codebook.json", CODEBOOK_PATH.stat().st_size / (1024 * 1024)),
        ("guidelines.db", DB_PATH.stat().st_size / (1024 * 1024)),
        ("extracted_tables/", _directory_size_bytes(TABLES_DIR) / (1024 * 1024)),
        ("Guidelines/", _directory_size_bytes(GUIDELINES_DIR) / (1024 * 1024)),
    ]
    return rows


def main() -> None:
    _validate_presence()

    quantizer = _load_quantizer_from_codebook(CODEBOOK_PATH)
    vocab_summary = _validate_vocab()
    db_summary = _validate_db_against_chroma(quantizer)
    coreml_summary = _validate_coreml()
    size_summary = _build_size_summary()

    print("Resource validation complete")
    print("\nSizes (MB):")
    for name, size_mb in size_summary:
        print(f"  {name:26s} {size_mb:10.2f}")

    print("\nVocabulary checks:")
    print(f"  Line count: {vocab_summary['count']}")
    print(f"  id 0 token: {vocab_summary['token_0']}")
    print(f"  id 101 token: {vocab_summary['token_101']}")
    print(f"  id 102 token: {vocab_summary['token_102']}")

    print("\nVector DB checks:")
    print(f"  Chunk count: {db_summary['chunk_count']}")
    print(f"  Vector count: {db_summary['vector_count']}")
    print(f"  Sample size: {db_summary['sample_size']}")
    print(f"  Min cosine: {db_summary['min_cosine']:.6f}")
    print(f"  Mean cosine: {db_summary['mean_cosine']:.6f}")

    print("\nCore ML checks:")
    print(f"  PyTorch/CoreML cosine: {coreml_summary['cosine']:.8f}")


if __name__ == "__main__":
    main()
