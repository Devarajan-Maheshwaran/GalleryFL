"""Framework-free federated aggregation primitives.

The server aggregates client *deltas* with sample/quality weighting. Each client
delta is clipped to a public L2 bound before averaging, limiting the influence
of one device without discarding legitimate Non-IID directionality.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def l2_norm(layers: Sequence[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.square(layer, dtype=np.float64))) for layer in layers)))


def clipped_fedavg(
    global_weights: Sequence[np.ndarray],
    client_weights: Sequence[Sequence[np.ndarray]],
    client_weights_scalar: Sequence[float],
    max_delta_norm: float,
) -> List[np.ndarray]:
    """Return ``global + weighted_mean(clipped_client_delta)``.

    ``client_weights_scalar`` should represent effective trusted sample counts,
    not raw device demand. Values must be finite and non-negative.
    """

    if not client_weights:
        raise ValueError("At least one client update is required")
    if len(client_weights) != len(client_weights_scalar):
        raise ValueError("Client updates and scalar weights must have equal length")
    if max_delta_norm <= 0 or not np.isfinite(max_delta_norm):
        raise ValueError("max_delta_norm must be finite and positive")

    reference = [np.asarray(layer, dtype=np.float32) for layer in global_weights]
    if not reference:
        raise ValueError("Global weights cannot be empty")

    scalars = np.asarray(client_weights_scalar, dtype=np.float64)
    if np.any(~np.isfinite(scalars)) or np.any(scalars < 0) or scalars.sum() <= 0:
        raise ValueError("Client scalar weights must be finite, non-negative, and non-zero")
    scalars /= scalars.sum()

    clipped_deltas: list[list[np.ndarray]] = []
    for update in client_weights:
        if len(update) != len(reference):
            raise ValueError("Client layer count mismatch")
        delta: list[np.ndarray] = []
        for client_layer, global_layer in zip(update, reference):
            client_array = np.asarray(client_layer, dtype=np.float32)
            if client_array.shape != global_layer.shape:
                raise ValueError(f"Client shape {client_array.shape} != global shape {global_layer.shape}")
            if not np.all(np.isfinite(client_array)):
                raise ValueError("Client update contains non-finite values")
            delta.append(client_array - global_layer)
        norm = l2_norm(delta)
        scale = min(1.0, max_delta_norm / max(norm, 1e-12))
        clipped_deltas.append([(layer * scale).astype(np.float32) for layer in delta])

    result: list[np.ndarray] = []
    for layer_index, global_layer in enumerate(reference):
        mean_delta = np.zeros_like(global_layer, dtype=np.float64)
        for client_index, delta in enumerate(clipped_deltas):
            mean_delta += scalars[client_index] * delta[layer_index]
        result.append((global_layer + mean_delta).astype(np.float32))
    return result
