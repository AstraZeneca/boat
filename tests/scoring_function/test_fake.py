"""Tests for the FakeMultimodalScoringFunction class."""

import pytest

from boat.scoring_function.fake import FakeMultimodalScoringFunction

"""Tests for FakeMultimodalScoringFunction."""


def test_fake_scoring_function_single_sequence():
    """Test scoring a single sequence."""
    sequence = ["AKQAKQ"]
    weights = {"A": 1.2, "K": 1.0, "Q": 0.8}
    scoring_function = FakeMultimodalScoringFunction(weights=weights)
    scores = scoring_function(sequence)
    scores = list(scores.values())[0]

    assert scores == [pytest.approx(2.4)], f"Expected score [2.4], got {scores}"


def test_fake_scoring_function_multiple_sequences():
    """Test scoring multiple sequences."""
    sequences = ["AKQAKQ", "AKKAKQ", "QQQQQQ"]
    weights = {"A": 1.2, "K": 1.0, "Q": 0.8}
    scoring_function = FakeMultimodalScoringFunction(weights=weights)
    scores = scoring_function(sequences)
    scores = list(scores.values())[0]

    expected = [pytest.approx(exp) for exp in [2.4, 3.0, 4.8]]
    assert scores == expected, f"Expected scores [2.4, 3.0, 4.8], got {scores}"


def test_fake_scoring_function_no_weights():
    """Test scoring with no weights provided."""
    sequences = ["AKQAKQ", "AKKAKQ"]
    scoring_function = FakeMultimodalScoringFunction()
    scores = scoring_function(sequences)
    expected = [pytest.approx(exp) for exp in [2.4, 3.0]]
    scores = list(scores.values())[0]
    assert scores == expected, f"Expected scores [2.4, 3.0], got {scores}"


def test_fake_scoring_function_empty_sequence():
    """Test scoring an empty sequence."""
    sequences = [""]
    scoring_function = FakeMultimodalScoringFunction()
    scores = scoring_function(sequences)
    scores = list(scores.values())[0]
    assert scores == [0.0], f"Expected score 0.0, got {scores}"


def test_fake_scoring_function_unknown_amino_acids():
    """Test scoring sequences with unknown amino acids."""
    sequences = ["XYZXYZ"]
    weights = {"A": 1.2, "K": 1.0, "Q": 0.8}
    scoring_function = FakeMultimodalScoringFunction(weights=weights)
    scores = scoring_function(sequences)
    scores = list(scores.values())[0]
    assert scores == [0.0], f"Expected score 0.0, got {scores}"
