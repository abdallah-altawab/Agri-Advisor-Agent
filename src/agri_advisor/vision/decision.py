"""Turn calibrated class probabilities into a decision the agent can act on."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DecisionStatus(StrEnum):
    """How the agent should proceed after seeing the classifier output."""

    CONFIDENT = "confident"  # one class clearly leads: answer directly
    AMBIGUOUS = "ambiguous"  # two classes share most of the mass: bring in context evidence
    UNCERTAIN = "uncertain"  # no clear picture: ask for a better photo or more information


@dataclass(frozen=True)
class DecisionPolicy:
    """Thresholds of the decision rule.

    These are hyperparameters to be tuned on field-like data, not universal constants.
    """

    top_k: int = 3
    margin_threshold: float = 0.2  # top-1 minus top-2 probability
    min_confidence: float = 0.5  # minimum top-1 probability to be CONFIDENT
    pair_mass_min: float = 0.7  # top-1 + top-2 mass needed to call a close race AMBIGUOUS


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


@dataclass(frozen=True)
class ClassifierOutput:
    """What the agent receives: ranked candidates plus a decision status."""

    top_k: tuple[Prediction, ...]
    margin: float
    entropy: float  # normalised to [0, 1]
    status: DecisionStatus


def normalized_entropy(probs: np.ndarray) -> float:
    """Shannon entropy divided by log(K): 0 = one-hot, 1 = uniform."""
    probs = np.asarray(probs, dtype=np.float64)
    nonzero = probs[probs > 0]
    value = -(nonzero * np.log(nonzero)).sum() / np.log(len(probs))
    return float(max(0.0, value))


def decide(
    probs: np.ndarray,
    class_names: Sequence[str],
    policy: DecisionPolicy | None = None,
) -> ClassifierOutput:
    """Rank classes and classify the situation as confident, ambiguous or uncertain."""
    policy = policy or DecisionPolicy()
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 1 or len(probs) != len(class_names):
        raise ValueError("probs must be a 1-D vector with one entry per class")
    if len(probs) < 2:
        raise ValueError("at least two classes are required")
    if (probs < 0).any() or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        raise ValueError("probs must be non-negative and sum to 1")

    order = np.argsort(probs)[::-1]
    top_k = tuple(
        Prediction(label=class_names[i], probability=float(probs[i]))
        for i in order[: policy.top_k]
    )
    p1, p2 = float(probs[order[0]]), float(probs[order[1]])
    margin = p1 - p2

    if margin < policy.margin_threshold:
        close_race = (p1 + p2) >= policy.pair_mass_min
        status = DecisionStatus.AMBIGUOUS if close_race else DecisionStatus.UNCERTAIN
    elif p1 < policy.min_confidence:
        status = DecisionStatus.UNCERTAIN
    else:
        status = DecisionStatus.CONFIDENT

    return ClassifierOutput(
        top_k=top_k, margin=margin, entropy=normalized_entropy(probs), status=status
    )