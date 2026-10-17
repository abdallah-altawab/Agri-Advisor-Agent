"""Confidence calibration utilities (NumPy only, no deep-learning framework needed)."""

import numpy as np

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of `logits / temperature`, numerically stable."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def _validate(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("logits must have shape (N, K) and labels shape (N,)")
    if len(labels) == 0:
        raise ValueError("at least one sample is required")
    if labels.dtype.kind not in "iu":
        raise ValueError("labels must be integer class indices")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("labels must be class indices in [0, K)")
    return logits, labels.astype(np.int64)


def negative_log_likelihood(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0
) -> float:
    """Mean negative log-likelihood of the true labels under softmax(logits / T)."""
    logits, labels = _validate(logits, labels)
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error with equal-width confidence bins (0 = perfectly calibrated)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or labels.ndim != 1 or len(labels) != len(probs):
        raise ValueError("probs must have shape (N, K) and labels shape (N,)")

    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    beta_bounds: tuple[float, float] = (0.05, 20.0),
    tol: float = 1e-4,
) -> float:
    """Fit the temperature T that minimises NLL on held-out data.

    NLL is convex in beta = 1/T, so a golden-section search over beta is enough.
    T is searched in roughly [0.05, 20].
    """
    logits, labels = _validate(logits, labels)
    low, high = beta_bounds

    def loss(beta: float) -> float:
        return negative_log_likelihood(logits, labels, temperature=1.0 / beta)

    c = high - _GOLDEN * (high - low)
    d = low + _GOLDEN * (high - low)
    fc, fd = loss(c), loss(d)
    while high - low > tol:
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - _GOLDEN * (high - low)
            fc = loss(c)
        else:
            low, c, fc = c, d, fd
            d = low + _GOLDEN * (high - low)
            fd = loss(d)
    return 1.0 / ((low + high) / 2.0)