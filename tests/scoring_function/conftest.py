"""Fixtures for scoring function tests."""

import random

import pytest

from boat.biologics.sequence import CDR, AbVSeq
from boat.genetic_algorithm.vocabularies import AA_VOCABULARY


@pytest.fixture
def target_sequence():
    """Fixture for a target sequence (even length for splitting into chains)."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    # use an even number of amino acids for equal splitting into heavy and light chains
    seq = "".join(random.choices(amino_acids, k=120))
    return seq


@pytest.fixture
def parental(target_sequence):
    """Fixture to create a parental AbVSeq from the target sequence."""
    return AbVSeq(heavy_chain=target_sequence, light_chain="")


@pytest.fixture
def test_sequences(target_sequence):
    """Fixture to generate an initial population of sequences."""
    population_size = 20
    seq_length = len(target_sequence)
    return ["".join(random.choices(AA_VOCABULARY, k=seq_length)) for _ in range(population_size)]


@pytest.fixture
def fake_cdr(target_sequence):
    """Fixture for a fake CDR object."""
    pos = (9, 18)
    return CDR(id="H1", sequence=target_sequence[pos[0] : pos[1] + 1], pos=pos)


@pytest.fixture
def test_cdrs(fake_cdr):
    """Fixture to generate an initial population of sequences."""
    population_size = 10
    seq_length = len(fake_cdr.sequence)
    return ["".join(random.choices(AA_VOCABULARY, k=seq_length)) for _ in range(population_size)]
