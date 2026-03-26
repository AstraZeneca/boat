"""Unit tests for genetic algorithm utilities in mlab-oneshot-active-learning."""

import random

import pytest

from boat.genetic_algorithm.utils import (
    count_mutations,
    generate_random_point_mutations,
)
from boat.genetic_algorithm.vocabularies import AA_VOCABULARY, aa_vocabulary_complete


@pytest.fixture
def target_sequence():
    """Fixture for the target sequence used as wild type."""
    return "ACACACACAC"


def test_count_mutations_identical():
    """Identical sequences should have 0 mutations."""
    seq1 = "ACDEFGHIK"
    seq2 = "ACDEFGHIK"
    assert count_mutations(seq1, seq2) == 0, "Identical sequences should return 0 mutations."


def test_count_mutations_all_different():
    """Completely different sequences should have mutations equal to the sequence length."""
    seq1 = "AAAAAAAAAA"
    seq2 = "CCCCCCCCCC"
    expected = len(seq1)
    assert count_mutations(seq1, seq2) == expected, f"Expected {expected} mutations, got {count_mutations(seq1, seq2)}."


def test_count_mutations_partial_difference():
    """Test count_mutations with sequences that differ at only one position."""
    seq1 = "ACDEFGHIKL"
    seq2 = "ACDEYGHIKL"  # Only the 5th character differs: F vs Y.
    assert count_mutations(seq1, seq2) == 1, "Should detect one mutation between the sequences."


def test_count_mutations_different_lengths():
    """
    Test count_mutations with sequences of different lengths.

    If sequences have different lengths, count_mutations will only compare up to the shorter length.
    For example:
        seq1 = "ACDEFG"
        seq2 = "ACXYFGZ"  -> Compares first 6 characters.
            A vs A => same
            C vs C => same
            D vs X => diff
            E vs Y => diff
            F vs F => same
            G vs G => same
    Expected differences = 2.
    """
    seq1 = "ACDEFG"
    seq2 = "ACXYFGZ"
    expected = 2
    assert count_mutations(seq1, seq2) == expected, f"Expected {expected} mutations, got {count_mutations(seq1, seq2)}."


@pytest.mark.parametrize(
    "n_mutations",
    [1, 2, 3],
)
def test_generate_random_point_mutations(target_sequence, n_mutations):
    """Test generating a population by introducing point mutations."""
    population_size = 25
    seq_length = len(target_sequence)
    population = generate_random_point_mutations(
        sequence=target_sequence,
        aa_vocabulary=aa_vocabulary_complete(),
        population_size=population_size,
        max_point_mutations=n_mutations,
        rng=random.Random(42),
    )

    # Check population size.
    assert len(population) == population_size

    # Verify each sequence's length.
    assert all(len(seq) == seq_length for seq in population)

    # Verify that sequences only contain valid amino acids.
    assert all(all(aa in AA_VOCABULARY for aa in seq) for seq in population)

    # Verify that none of the generated sequences are identical to the wild-type.
    assert all(seq != target_sequence for seq in population)

    # Check that the number of mutations does not exceed max_mutations.
    for seq in population:
        mutations = sum(1 for a, b in zip(seq, target_sequence) if a != b)
        assert mutations <= n_mutations, f"Sequence {seq} has {mutations} mutations; expected at most {n_mutations}"
