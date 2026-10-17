"""Interfaces between the decision layer and concrete image models."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class ImageClassifier(Protocol):
    """Anything that can turn an image file into raw class logits."""

    @property
    def class_names(self) -> Sequence[str]: ...

    def predict_logits(self, image_path: Path) -> np.ndarray:
        """Return a 1-D array of raw (uncalibrated) logits, one per class."""
        ...