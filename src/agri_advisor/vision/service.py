"""Glue between an image classifier, calibration and the decision rule."""

from pathlib import Path

import numpy as np

from agri_advisor.vision.calibration import softmax
from agri_advisor.vision.decision import ClassifierOutput, DecisionPolicy, decide
from agri_advisor.vision.protocols import ImageClassifier


class DiagnosisService:
    """image -> logits -> temperature-scaled probabilities -> decision."""

    def __init__(
        self,
        classifier: ImageClassifier,
        temperature: float = 1.0,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self._classifier = classifier
        self._temperature = temperature
        self._policy = policy

    def diagnose(self, image_path: Path) -> ClassifierOutput:
        logits = np.asarray(self._classifier.predict_logits(image_path), dtype=np.float64)
        names = self._classifier.class_names
        if logits.ndim != 1 or len(logits) != len(names):
            raise ValueError(
                f"classifier returned {logits.size} logits for {len(names)} classes"
            )
        probs = softmax(logits, self._temperature)
        return decide(probs, names, self._policy)