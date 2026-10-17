"""Scripted classifier for demos and tests. NOT a real model: its output is meaningless."""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

STUB_CLASS_NAMES = ("candidate_A", "candidate_B", "candidate_C", "candidate_D")

SCENARIOS: dict[str, tuple[float, ...]] = {
    "confident": (0.90, 0.05, 0.03, 0.02),
    "ambiguous": (0.45, 0.40, 0.10, 0.05),
    "uncertain": (0.30, 0.28, 0.22, 0.20),
}


class StubClassifier:
    """Returns fixed logits regardless of the image."""

    def __init__(
        self,
        logits: Sequence[float],
        class_names: Sequence[str] = STUB_CLASS_NAMES,
    ) -> None:
        self._logits = np.asarray(logits, dtype=np.float64)
        self._class_names = tuple(class_names)

    @classmethod
    def from_scenario(cls, scenario: str) -> Self:
        try:
            probs = SCENARIOS[scenario]
        except KeyError as exc:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Choose from {sorted(SCENARIOS)}."
            ) from exc
        return cls(np.log(np.asarray(probs)))

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._class_names

    def predict_logits(self, image_path: Path) -> np.ndarray:
        return self._logits.copy()