import numpy as np
import pytest

from agri_advisor.vision.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    softmax,
)


def synthetic(true_temperature, n=20000, k=5, seed=0):
    """Logits whose labels are sampled from softmax(logits / true_temperature)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(n, k))
    probs = softmax(logits, temperature=true_temperature)
    u = rng.random((n, 1))
    labels = np.minimum((probs.cumsum(axis=1) < u).sum(axis=1), k - 1)
    return logits, labels


def test_softmax_rows_sum_to_one_and_higher_temperature_flattens():
    logits = np.array([[2.0, 1.0, 0.0]])
    sharp, flat = softmax(logits, 1.0), softmax(logits, 2.0)
    assert sharp.sum() == pytest.approx(1.0)
    assert flat.sum() == pytest.approx(1.0)
    assert flat.max() < sharp.max()


def test_softmax_is_numerically_stable():
    probs = softmax(np.array([[1000.0, 1001.0]]))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("true_temperature", [0.7, 1.0, 2.0])
def test_fit_temperature_recovers_known_value(true_temperature):
    logits, labels = synthetic(true_temperature)
    assert fit_temperature(logits, labels) == pytest.approx(true_temperature, abs=0.15)


def test_fitted_temperature_lowers_nll():
    logits, labels = synthetic(2.0)
    fitted = fit_temperature(logits, labels)
    assert negative_log_likelihood(logits, labels, fitted) < negative_log_likelihood(logits, labels, 1.0)


def test_ece_small_when_calibrated_and_larger_when_overconfident():
    logits, labels = synthetic(2.0)
    calibrated = expected_calibration_error(softmax(logits, 2.0), labels)
    overconfident = expected_calibration_error(softmax(logits, 1.0), labels)
    assert calibrated < 0.03
    assert overconfident > calibrated + 0.02


def test_validation_errors():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 4)), np.array([0, 1]))  # length mismatch
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 4)), np.array([0, 4]))  # label out of range
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 4)), np.array([0.0, 1.0]))  # non-integer labels