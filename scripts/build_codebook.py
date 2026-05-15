#!/usr/bin/env python3
"""Build TurboQuant codebook JSON for iOS resources.

Outputs:
    GuidelineAssistant/GuidelineAssistant/Resources/codebook.json

The JSON includes both:
- snake_case keys requested in the handoff
- camelCase compatibility keys consumed by Swift TurboQuantCodebook decoding
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.turboquant import TurboQuantMSE


DIMENSION = 384
BITS_PER_COORDINATE = 3
SEED = 42
OUTPUT_PATH = Path("GuidelineAssistant/GuidelineAssistant/Resources/codebook.json")


def _assert_float32_bit_equal(lhs: np.ndarray, rhs: np.ndarray, label: str) -> None:
    lhs32 = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs32 = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if lhs32.shape != rhs32.shape:
        raise ValueError(f"{label} shape mismatch: {lhs32.shape} vs {rhs32.shape}")
    if not np.array_equal(lhs32.view(np.uint32), rhs32.view(np.uint32)):
        raise ValueError(f"{label} changed after serialization round-trip")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    quantizer = TurboQuantMSE(d=DIMENSION, b=BITS_PER_COORDINATE, seed=SEED)
    centroids = np.asarray(quantizer.codebook, dtype=np.float32)
    rotation_col_major = np.asarray(quantizer.rotation, dtype=np.float32).ravel(order="F")

    payload = {
        "version": 1,
        "dimension": DIMENSION,
        "bits_per_coordinate": BITS_PER_COORDINATE,
        "seed": SEED,
        "centroids": [float(x) for x in centroids.tolist()],
        "rotation_matrix_flat": [float(x) for x in rotation_col_major.tolist()],
        "rotation_matrix_layout": "column_major",
        # Swift compatibility keys for TurboQuantCodebook decoding.
        "bitsPerCoordinate": BITS_PER_COORDINATE,
        "rotationMatrix": [float(x) for x in rotation_col_major.tolist()],
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    loaded_centroids = np.asarray(loaded["centroids"], dtype=np.float32)
    loaded_rotation = np.asarray(loaded["rotation_matrix_flat"], dtype=np.float32)

    _assert_float32_bit_equal(centroids, loaded_centroids, "centroids")
    _assert_float32_bit_equal(rotation_col_major, loaded_rotation, "rotation_matrix_flat")

    print("Codebook generation complete")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Dimension: {DIMENSION}")
    print(f"  Bits/coord: {BITS_PER_COORDINATE}")
    print(f"  Seed: {SEED}")
    print(f"  Centroids: {centroids.shape[0]}")
    print(f"  Rotation entries: {rotation_col_major.shape[0]}")


if __name__ == "__main__":
    main()
