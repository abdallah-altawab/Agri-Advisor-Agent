import numpy as np
import pytest

from agri_advisor.vision.decision import (
    DecisionPolicy,
    DecisionStatus,
    decide,
    normalized_entropy,
)


def test_confident_when_one_class_clearly_leads():
    out = decide(np.array([0.90, 0.05, 0.03, 0.02]), ["A", "B", "C", "D"])
    assert out.status == DecisionStatus.CONFIDENT
    assert [p.label for p in out.top_k] == ["A", "B", "C"]
    assert out.margin == pytest.approx(0.85)


def test_ambiguous_when_two_classes_are_close_and_share_the_mass():
    out = decide(np.array([0.15, 0.40, 0.45]), ["a", "b", "c"])
    assert out.status == DecisionStatus.AMBIGUOUS
    assert [p.label for p in out.top_k] == ["c", "b", "a"]
    assert out.margin == pytest.approx(0.05)


def test_uncertain_when_mass_is_spread_over_many_classes():
    out = decide(np.array([0.30, 0.25, 0.20, 0.15, 0.10]), list("abcde"))
    assert out.status == DecisionStatus.UNCERTAIN


def test_uncertain_when_top_confidence_is_low_even_with_clear_margin():
    out = decide(np.array([0.45, 0.20, 0.20, 0.15]), list("abcd"))
    assert out.status == DecisionStatus.UNCERTAIN


def test_top_k_is_respected():
    out = decide(np.array([0.7, 0.2, 0.1]), list("abc"), DecisionPolicy(top_k=2))
    assert len(out.top_k) == 2


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        decide(np.array([0.5, 0.3, 0.2]), ["a", "b"])  # names/probs length mismatch
    with pytest.raises(ValueError):
        decide(np.array([0.5, 0.3]), ["a", "b"])  # does not sum to 1


def test_entropy_is_normalised():
    assert normalized_entropy(np.full(4, 0.25)) == pytest.approx(1.0)
    assert normalized_entropy(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)