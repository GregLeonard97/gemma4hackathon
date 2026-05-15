#!/usr/bin/env python3
"""Generate embedding_reference.json for iOS embedding parity tests.

Usage:
    python scripts/generate_embedding_reference.py

Optional flags:
    --text "neonatal sepsis early-onset"
    --output GuidelineAssistant/GuidelineAssistantTests/Fixtures/embedding_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BGE-small embedding reference JSON")
    parser.add_argument(
        "--text",
        default="neonatal sepsis early-onset",
        help="Input text to embed",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-Transformers model name",
    )
    parser.add_argument(
        "--output",
        default="GuidelineAssistant/GuidelineAssistantTests/Fixtures/embedding_reference.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Embedding text: {args.text!r}")
    embedding = model.encode(
        args.text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    values = embedding.astype("float32").tolist()
    if len(values) != 384:
        raise ValueError(f"Expected 384 dimensions, got {len(values)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)

    print(f"Wrote {len(values)}-dim embedding to {output_path}")


if __name__ == "__main__":
    main()
