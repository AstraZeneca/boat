"""Tests for the OASis scoring function."""
import random

import pytest

from boat.scoring_function.oasis_interface import OASisScoringFunction

random.seed(42)


@pytest.fixture
def oasis_score(parental):
    """Fixture for OASiS scoring function."""
    return OASisScoringFunction(parental=parental)


@pytest.fixture
def oasis_score_swissprot(parental):
    """Fixture for OASiS scoring function with SwissProt database."""
    return OASisScoringFunction(parental=parental, reference_db="human-swissprot")


def test_init(oasis_score, parental):
    """Test initialization of OASis scoring function."""
    assert oasis_score.parental == parental, "Failed to initialize OASis scoring function with parental sequence."
    assert oasis_score.current_cdr is None, "Failed to initialize OASis scoring function with no current CDR."
    assert oasis_score.db is not None, "Failed to initialize database"
    assert hasattr(oasis_score.db, "compute_peptide_content"), "Database missing required method"


def test_swissprot_db_initialization(oasis_score_swissprot):
    """Test initialization with SwissProt database."""
    assert oasis_score_swissprot.db is not None
    assert hasattr(oasis_score_swissprot.db, "compute_peptide_content")


def test_call_with_full_sequences(oasis_score, test_sequences):
    """Test scoring a list of the full sequences."""
    scores = oasis_score(test_sequences)
    scores = list(scores.values())[0]

    assert len(scores) == len(test_sequences), "Failed to score the sequences."
    for score in scores:
        assert isinstance(score, float), "Scores should be floats."
        assert 0.0 <= score <= 1.0, "Scores should be between 0 and 1."


def test_call_fails_with_cdr_sequences(oasis_score, test_cdrs):
    """Test scoring a list of CDR sequences when CDR is not given."""
    with pytest.raises(ValueError):
        oasis_score(test_cdrs)


def test_set_cdr(oasis_score, fake_cdr):
    """Test setting the current CDR in the OASis scoring function."""
    oasis_score.set_cdr(fake_cdr)
    assert oasis_score.current_cdr == fake_cdr, "Failed to set the current CDR in OASis scoring function."


def test_call_with_cdr_sequences(oasis_score, fake_cdr, test_cdrs):
    """Test scoring a list of CDR sequences."""
    oasis_score.set_cdr(fake_cdr)
    scores = oasis_score(test_cdrs)
    scores = list(scores.values())[0]
    assert len(scores) == len(test_cdrs), "Failed to score the sequences."
    for score in scores:
        assert isinstance(score, float), "Scores should be floats."
        assert 0.0 <= score <= 1.0, "Scores should be between 0 and 1."


def test_call_with_full_sequences_fails_when_cdr_given(oasis_score, fake_cdr, test_sequences):
    """Test that scoring a list of full sequences when CDR is given will fail."""
    oasis_score.set_cdr(fake_cdr)
    with pytest.raises(ValueError):
        oasis_score(test_sequences)


def test_invalid_sequences(oasis_score):
    """Test scoring with invalid sequences."""
    # Test with non-amino acid characters
    with pytest.raises(ValueError):
        oasis_score(["ACDEFGXYZ"])
