"""TurboQuantMSE (Algorithm 1) for embedding compression.

This module implements the MSE-optimized TurboQuant variant described in:
Zandieh et al., "TurboQuant", ICLR 2026 (arXiv:2504.19874).

Core ideas:
1. L2-normalize each vector and store original norm separately.
2. Apply a random Haar rotation.
3. Quantize each rotated coordinate with a Lloyd-Max scalar quantizer
   trained for N(0, 1/d) coordinates.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import norm as gaussian


def _haar_random_rotation(d: int, seed: int) -> np.ndarray:
    """Sample a Haar-uniform random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d), dtype=np.float64)
    q, r = np.linalg.qr(a)

    # Sign correction gives a uniform draw from the orthogonal group.
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs
    return q.astype(np.float32)


def _lloyd_max_gaussian_codebook(
    d: int,
    levels: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fit Lloyd-Max reconstruction levels for N(0, 1/d)."""
    sigma = 1.0 / math.sqrt(float(d))

    # Quantile initialization is stable and converges quickly for symmetric
    # unimodal distributions.
    probs = (np.arange(levels, dtype=np.float64) + 0.5) / float(levels)
    codebook = sigma * gaussian.ppf(probs)

    for _ in range(max_iter):
        boundaries = np.empty(levels + 1, dtype=np.float64)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        boundaries[1:-1] = 0.5 * (codebook[:-1] + codebook[1:])

        updated = np.empty_like(codebook)
        for i in range(levels):
            a = boundaries[i] / sigma
            b = boundaries[i + 1] / sigma
            mass = gaussian.cdf(b) - gaussian.cdf(a)

            if mass <= 1e-15:
                updated[i] = codebook[i]
                continue

            # E[Z | a < Z < b] for Z ~ N(0,1) is (phi(a)-phi(b))/(Phi(b)-Phi(a))
            truncated_mean = (gaussian.pdf(a) - gaussian.pdf(b)) / mass
            updated[i] = sigma * truncated_mean

        delta = float(np.max(np.abs(updated - codebook)))
        codebook = updated
        if delta < tol:
            break

    return codebook.astype(np.float32)


class TurboQuantMSE:
    """TurboQuantMSE quantizer for d-dimensional vectors at b bits/coordinate."""

    def __init__(self, d: int, b: int, seed: int = 42):
        if d <= 0:
            raise ValueError("d must be positive")
        if b <= 0 or b > 8:
            raise ValueError("b must be in [1, 8]")

        self.d = int(d)
        self.b = int(b)
        self.seed = int(seed)
        self.levels = 1 << self.b

        self.codebook = _lloyd_max_gaussian_codebook(
            d=self.d,
            levels=self.levels,
        )
        self.thresholds = (
            (self.codebook[:-1] + self.codebook[1:]) * 0.5
        ).astype(np.float32)
        self.rotation = _haar_random_rotation(self.d, self.seed)

    def quantize(self, vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize a single vector into codebook indices plus original L2 norm."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.d:
            raise ValueError(f"Expected vector of length {self.d}, got {vec.shape[0]}")

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return np.zeros(self.d, dtype=np.uint8), 0.0

        unit = vec / norm
        rotated = unit @ self.rotation.T
        indices = np.searchsorted(self.thresholds, rotated, side="right").astype(np.uint8)
        return indices, norm

    def dequantize(self, indices: np.ndarray, norm: float) -> np.ndarray:
        """Reconstruct a single vector from indices and stored norm."""
        idx = np.asarray(indices).reshape(-1)
        if idx.shape[0] != self.d:
            raise ValueError(f"Expected indices of length {self.d}, got {idx.shape[0]}")
        if norm == 0.0:
            return np.zeros(self.d, dtype=np.float32)

        rotated = self.codebook[idx.astype(np.int64)]
        unit = rotated @ self.rotation
        return (unit * float(norm)).astype(np.float32)

    def quantize_batch(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized quantization for shape (n, d)."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.d:
            raise ValueError(f"Expected shape (n, {self.d}), got {arr.shape}")

        norms = np.linalg.norm(arr, axis=1).astype(np.float32)
        safe = np.where(norms > 0.0, norms, 1.0).astype(np.float32)
        unit = arr / safe[:, None]
        unit[norms == 0.0] = 0.0

        rotated = unit @ self.rotation.T
        indices = np.searchsorted(self.thresholds, rotated, side="right").astype(np.uint8)
        return indices, norms

    def dequantize_batch(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Vectorized dequantization for shape (n, d)."""
        idx = np.asarray(indices)
        arr_norms = np.asarray(norms, dtype=np.float32).reshape(-1)

        if idx.ndim != 2 or idx.shape[1] != self.d:
            raise ValueError(f"Expected indices shape (n, {self.d}), got {idx.shape}")
        if arr_norms.shape[0] != idx.shape[0]:
            raise ValueError("norm count must match number of rows in indices")

        rotated = self.codebook[idx.astype(np.int64)]
        unit = rotated @ self.rotation
        return (unit * arr_norms[:, None]).astype(np.float32)


def estimate_turboquant_mse(
    d: int = 1536,
    b: int = 2,
    samples: int = 4096,
    seed: int = 123,
    quantizer_seed: int = 42,
) -> float:
    """Estimate E[||x - x_hat||^2] on random unit vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((samples, d), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    quantizer = TurboQuantMSE(d=d, b=b, seed=quantizer_seed)
    indices, norms = quantizer.quantize_batch(vectors)
    recon = quantizer.dequantize_batch(indices, norms)

    sq_errors = np.sum((vectors - recon) ** 2, axis=1)
    return float(np.mean(sq_errors))


if __name__ == "__main__":
    mse = estimate_turboquant_mse(d=1536, b=2, samples=4096)
    print(f"TurboQuantMSE sanity check (d=1536, b=2): {mse:.6f}")