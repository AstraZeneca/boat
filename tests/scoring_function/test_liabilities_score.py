"""Tests for LiabilityScoringFunction in boat.scoring_function.liabilities_score."""

from boat.scoring_function.liabilities_score import LiabilityScoringFunction


def test_liability_scoring_function_invert_true():
    """Test LiabilityScoringFunction with invert=True."""
    scoring_function = LiabilityScoringFunction(invert=True)
    sequences = ["CAVDGVV", "AKGDKGD", "AAAAKM"]
    scores = scoring_function(sequences)
    assert isinstance(scores, dict)
    assert scoring_function.objective_names[0] in scores
    # Scores should be negative (inverted)
    for score in scores[scoring_function.objective_names[0]]:
        assert score <= 0


def test_liability_scoring_function_invert_false():
    """Test LiabilityScoringFunction with invert=False."""
    scoring_function = LiabilityScoringFunction(invert=False)
    sequences = ["CAVDGVV", "AKGDKGD", "AAAAKM"]
    scores = scoring_function(sequences)
    assert isinstance(scores, dict)
    assert scoring_function.objective_names[0] in scores
    # Scores should be positive (not inverted)
    for score in scores[scoring_function.objective_names[0]]:
        assert score >= 0


def test_liability_scoring_function_empty_input():
    """Test LiabilityScoringFunction with empty input list."""
    scoring_function = LiabilityScoringFunction()
    scores = scoring_function([])
    assert isinstance(scores, dict)
    assert scores[scoring_function.objective_names[0]] == []


def test_liability_scoring_function_consistency():
    """Test that LiabilityScoringFunction gives consistent results for the same input."""
    scoring_function = LiabilityScoringFunction(invert=False)
    seq = "CAVDGVV"
    score1 = scoring_function([seq])[scoring_function.objective_names[0]][0]
    score2 = scoring_function([seq])[scoring_function.objective_names[0]][0]
    assert score1 == score2
