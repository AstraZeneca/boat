"""Tests for the MultiObjectiveScoringFunction class."""

import pytest

from boat.scoring_function.fake import FakeMultimodalScoringFunction
from boat.scoring_function.interface import MultiObjectiveScoringFunction


@pytest.fixture
def mo_scoring_function():
    """Fixture for the MultiObjectiveScoringFunction instance."""
    sf1 = FakeMultimodalScoringFunction()
    sf2 = FakeMultimodalScoringFunction(weights={"A": 2.4, "K": 2.0, "Q": 1.6})
    return MultiObjectiveScoringFunction(
        scoring_functions=[sf1, sf2],
        # Arguments specific to each function
        args_list=[(), ()],  # No positional args needed
        kwargs_list=[{}, {}],  # No kwargs
    )


def test_fake_scoring_function_single_sequence(mo_scoring_function):
    """Test scoring a single sequence."""
    sequence = ["AKQAKQ"]
    scores = mo_scoring_function(sequence)
    key1, key2 = scores.keys()

    # Each key should have a list with one value
    assert len(scores[key1]) == 1, f"Expected one score for 'One', got {len(scores[key1])}"
    assert len(scores[key2]) == 1, f"Expected one score for 'Two', got {len(scores[key2])}"

    # Convert dictionary to list
    scores_list = [[scores[key1][0], scores[key2][0]]]

    expected = [[pytest.approx(2.4), pytest.approx(4.8)]]
    assert scores_list == expected, f"Expected scores [2.4, 4.8], got {scores_list}"


def test_fake_scoring_function_multi_sequence(mo_scoring_function):
    """Test scoring multiple sequences."""
    sequences = ["AKQAKQ", "AKKAKQ", "QQQQQQ"]
    scores = mo_scoring_function(sequences)
    key1, key2 = scores.keys()

    # Each key should have a list with three values
    assert len(scores[key1]) == 3, f"Expected three scores for 'One', got {len(scores[key1])}"
    assert len(scores[key2]) == 3, f"Expected three scores for 'Two', got {len(scores[key2])}"

    # Convert dictionary to the expected format
    scores_list = []
    for i in range(len(sequences)):
        scores_list.append([scores[key1][i], scores[key2][i]])

    expected = [
        [pytest.approx(2.4), pytest.approx(4.8)],
        [pytest.approx(3.0), pytest.approx(6.0)],
        [pytest.approx(4.8), pytest.approx(9.6)],
    ]
    assert scores_list == expected, f"Expected scores [[2.4, 4.8], [3.0, 6.0], [4.8, 9.6], got {scores_list}"
