#!/usr/bin/env python3
"""Convert BAAI/bge-small-en-v1.5 to Core ML for iOS bundling.

Run from project root with Python 3.11:
    python3.11 -m venv .venv-coreml
    source .venv-coreml/bin/activate
    pip install -r scripts/requirements_coreml_conversion.txt
    python scripts/convert_bge_to_coreml.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


OUTPUT_PATH = Path("GuidelineAssistant/GuidelineAssistant/Resources/BGESmall.mlpackage")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_LENGTH = 512
COSINE_THRESHOLD = 0.9999


class BGEEmbedder(nn.Module):
    """Wrap BGE model to return L2-normalized CLS embeddings."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_embedding, p=2, dim=1)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_vec = lhs.astype(np.float32).reshape(-1)
    rhs_vec = rhs.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(lhs_vec) * np.linalg.norm(rhs_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(lhs_vec, rhs_vec) / denom)


def _directory_size_mb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def main() -> None:
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    wrapper = BGEEmbedder(model)
    wrapper.eval()

    print("Tracing model...")
    sample_input_ids = torch.zeros((1, MAX_LENGTH), dtype=torch.int32)
    sample_attention_mask = torch.ones((1, MAX_LENGTH), dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (sample_input_ids, sample_attention_mask),
            strict=False,
        )

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_LENGTH), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_LENGTH), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.short_description = "BGE-small-en-v1.5 sentence embedder"
    mlmodel.author = "BAAI / converted for clinical RAG"
    mlmodel.version = "1.0"
    mlmodel.input_description["input_ids"] = "Tokenized input sequence"
    mlmodel.input_description["attention_mask"] = "Attention mask (1=token, 0=padding)"
    mlmodel.output_description["embedding"] = "L2-normalized 384-dim sentence embedding"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        if OUTPUT_PATH.is_dir():
            shutil.rmtree(OUTPUT_PATH)
        else:
            OUTPUT_PATH.unlink()

    mlmodel.save(str(OUTPUT_PATH))
    print(f"Saved to {OUTPUT_PATH}")

    print("\nValidating conversion...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_text = "neonatal sepsis early-onset gentamicin dose"
    encoded = tokenizer(
        test_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        pytorch_output = wrapper(
            encoded["input_ids"].to(torch.int32),
            encoded["attention_mask"].to(torch.int32),
        ).numpy()[0]

    coreml_output = mlmodel.predict(
        {
            "input_ids": encoded["input_ids"].to(torch.int32).numpy(),
            "attention_mask": encoded["attention_mask"].to(torch.int32).numpy(),
        }
    )["embedding"][0]

    cosine = _cosine_similarity(pytorch_output, coreml_output)
    pytorch_norm = float(np.linalg.norm(pytorch_output))
    coreml_norm = float(np.linalg.norm(coreml_output))

    print(f"PyTorch L2 norm: {pytorch_norm:.6f}")
    print(f"Core ML L2 norm: {coreml_norm:.6f}")
    print(f"Cosine similarity (PyTorch vs Core ML): {cosine:.6f}")

    if cosine < COSINE_THRESHOLD:
        raise RuntimeError(
            f"Validation failed: cosine {cosine:.6f} < {COSINE_THRESHOLD:.4f}"
        )

    print("\nValidation PASSED")
    print(f"Package size: {_directory_size_mb(OUTPUT_PATH):.2f} MB")


if __name__ == "__main__":
    main()
