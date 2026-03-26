"""Tests for the basic genetic operators: single_point_crossover and mutate_sequence."""


import random

import pytest

from boat.genetic_algorithm.genetic_operators import (
    _get_mutable_positions,
    batch_crossover,
    mutate_batch,
    mutate_sequence,
    single_point_crossover,
)
from boat.genetic_algorithm.vocabularies import (
    aa_to_positional_vocabulary,
    aa_vocabulary_complete,
    aa_vocabulary_reduced,
)


# Dummy RNG with fixed crossover point for testing single_point_crossover.
class DummyRNG:
    """A dummy random number generator that always returns a fixed crossover point."""

    def randint(self, a, b):
        """Override the randint method to return a fixed crossover point."""
        # Always return a fixed crossover point of 3.
        return 3

    def random(self):
        """Override the random method to return a fixed value."""
        return 0.0  # always trigger crossover

    def sample(self, population, k):
        """Override the sample method to return a fixed subset."""
        return population[:k]


def test_single_point_crossover_correctness():
    """Test that single_point_crossover recombines sequences correctly."""
    seq1 = "AAAAAA"
    seq2 = "CCCCCC"
    dummy_rng = DummyRNG()
    new_seq1, new_seq2 = single_point_crossover(seq1, seq2, rng=dummy_rng)
    # With a crossover point of 3, expect:
    # new_seq1 = seq1[:3] + seq2[3:] => "AAA" + "CCC" = "AAACCC"
    # new_seq2 = seq2[:3] + seq1[3:] => "CCC" + "AAA" = "CCCAAA"
    assert new_seq1 == "AAACCC", f"Expected AAACCC, got {new_seq1}"
    assert new_seq2 == "CCCAAA", f"Expected CCCAAA, got {new_seq2}"


def test_single_point_crossover_different_length():
    """Test that single_point_crossover raises ValueError for sequences of unequal lengths."""
    seq1 = "AAAA"
    seq2 = "CCCCCC"
    with pytest.raises(ValueError):
        single_point_crossover(seq1, seq2)


def test_mutate_sequence_no_mutation():
    """Test that mutate_sequence returns the original sequence if mutation_rate is 0."""
    original_seq = "ACDEFGHIKL"
    vocab = aa_vocabulary_complete()
    vocab = aa_to_positional_vocabulary(original_seq, vocab)
    mutated_seq = mutate_sequence(original_seq, mutation_rate=0.0, aa_vocabulary=vocab)
    assert mutated_seq == original_seq, "Sequence should be unchanged when mutation_rate is 0."


def test_mutate_sequence_probability_matrix():
    """Test that mutate_sequence respects the provided probability matrix."""
    original_seq = "AAA"
    # position-based vocabulary: each position can be A, B or C
    vocab = {0: "ABC", 1: "ABC", 2: "ABC"}

    # Construct a probability matrix that strongly favors a specific alternative
    # For position 0 favor 'C', for position 1 favor 'B', for position 2 favor 'B'
    probability_matrix = {
        0: {"B": 0.0, "C": 1.0},
        1: {"B": 1.0, "C": 0.0},
        2: {"B": 1.0, "C": 0.0},
    }

    rng = random.Random(42)
    mutated_seq = mutate_sequence(
        original_seq, mutation_rate=1.0, aa_vocabulary=vocab, rng=rng, probability_matrix=probability_matrix
    )

    # With the extreme weights above, the expected mutated sequence is "CBB"
    assert mutated_seq == "CBB", f"Expected 'CBB' but got '{mutated_seq}'"


def test_mutate_sequence_full_mutation():
    """
    Test that mutate_sequence changes every amino acid when mutation_rate is 1.

    Since the function ensures a mutated amino acid differs from the original,
    each position should differ.
    """
    original_seq = "ACDEFGHIKL"
    vocab = aa_vocabulary_complete()
    vocab = aa_to_positional_vocabulary(original_seq, vocab)
    mutated_seq = mutate_sequence(original_seq, mutation_rate=1.0, aa_vocabulary=vocab)
    assert len(mutated_seq) == len(original_seq), "Mutated sequence should have the same length."
    for i, (orig, new) in enumerate(zip(original_seq, mutated_seq)):
        assert new in vocab[i], f"Invalid amino acid {new} in mutated sequence at position {i}."
        assert new != orig, f"Position did not mutate: {orig} remained unchanged at position {i}."


@pytest.mark.parametrize(
    "original_seq, vocab",
    [
        ("ACDEFGHIKL", {aa: aa + "X" for aa in "ACDEFGHIKL"}),  # AA-based
        ("ACDEFGHIKL", {i: aa + "X" for i, aa in enumerate("ACDEFGHIKL")}),  # position-based
    ],
)
def test_mutate_sequence_vocabularies(original_seq, vocab):
    """
    Test mutate_sequence for different vocabularies.

    This checks that the function can handle both positional and amino acid-based vocabularies.
    """
    mutated_seq = mutate_sequence(original_seq, mutation_rate=0.5, aa_vocabulary=vocab)
    # Verify same length:
    assert len(mutated_seq) == len(original_seq)

    for i, aa in enumerate(mutated_seq):
        # Check that each mutated amino acid is in the vocabulary
        assert aa in ["X", original_seq[i]], f"Mutated amino acid {aa} not in vocabulary."


def test_get_mutable_positions():
    """Test the _get_mutable_positions helper function."""
    original_seq = "ACDEFG"
    sequence_list = list(original_seq)

    # Test with complete vocabulary
    complete_vocab = aa_vocabulary_complete()
    complete_vocab = aa_to_positional_vocabulary(original_seq, complete_vocab)
    mutable_positions = _get_mutable_positions(sequence_list, complete_vocab)
    assert set(mutable_positions) == set(
        range(len(sequence_list))
    ), "All positions should be mutable with complete vocabulary"

    # Test with reduced vocabulary
    reduced_vocab = aa_vocabulary_reduced()
    reduced_vocab = aa_to_positional_vocabulary(original_seq, reduced_vocab)
    mutable_positions = _get_mutable_positions(sequence_list, reduced_vocab)
    expected_mutable = [0, 2, 3, 4, 5]  # All positions except index 1 (C)
    assert set(mutable_positions) == set(
        expected_mutable
    ), "All positions except C should be mutable with reduced vocabulary"

    # Test with empty vocabulary
    empty_vocab = {}
    mutable_positions = _get_mutable_positions(sequence_list, empty_vocab)
    assert mutable_positions == [], "Empty vocabulary should yield no mutable positions"

    # Test with single-option vocabulary (no mutations possible)
    single_option_vocab = {"A": "A", "C": "C", "D": "D"}
    single_option_vocab = aa_to_positional_vocabulary(original_seq, single_option_vocab)
    mutable_positions = _get_mutable_positions(sequence_list, single_option_vocab)
    assert mutable_positions == [], "Single-option vocabulary should yield no mutable positions"

    # Test with custom AA vocabulary
    custom_vocab = {"A": "ABC", "C": "C", "D": "DE", "E": "E", "F": "FG", "G": "G"}
    custom_vocab = aa_to_positional_vocabulary(original_seq, custom_vocab)
    mutable_positions = _get_mutable_positions(sequence_list, custom_vocab)
    expected_mutable = [0, 2, 4]
    assert set(mutable_positions) == set(expected_mutable), "Only positions with multiple options should be mutable"

    # Test with custom numeric vocabulary
    custom_vocab = {
        0: "ABC",
        1: "C",
        2: "DE",
        3: "E",
        4: "FG",
        5: "G",
    }
    mutable_positions = _get_mutable_positions(sequence_list, custom_vocab)
    expected_mutable = [0, 2, 4]
    assert set(mutable_positions) == set(expected_mutable), "Only positions with multiple options should be mutable"


def test_batch_crossover_single_element_batches():
    """Test batch_crossover with single-element batches."""
    seq1 = "AAAAAA"
    seq2 = "CCCCCC"
    batch1 = [seq1]
    batch2 = [seq2]
    rng = DummyRNG()
    out1, out2 = batch_crossover(batch1, batch2, rng=rng)
    assert out1 == ["AAACCC"]
    assert out2 == ["CCCAAA"]


def test_batch_crossover_no_crossover():
    """Test batch_crossover when crossover rates are 0 (no crossover occurs)."""
    batch1 = ["AAAA", "BBBB"]
    batch2 = ["CCCC", "DDDD"]
    out1, out2 = batch_crossover(batch1, batch2, single_crossover_rate=0.0, batch_crossover_rate=0.0)
    # Should be unchanged except for possible shuffling
    assert sorted(out1) == sorted(batch1)
    assert sorted(out2) == sorted(batch2)


def test_batch_crossover_full_batch_swap():
    """Test batch_crossover with batch_crossover_rate=1 (all pairs swapped)."""
    batch1 = ["AAAA", "BBBB"]
    batch2 = ["CCCC", "DDDD"]
    out1, out2 = batch_crossover(batch1, batch2, single_crossover_rate=0.0, batch_crossover_rate=1.0)
    # All should be swapped
    assert sorted(out1) == sorted(batch2)
    assert sorted(out2) == sorted(batch1)


def test_batch_crossover_single_point_and_batch_crossover():
    """Test batch_crossover with both single and batch crossover enabled."""
    batch1 = ["AAAA", "BBBB"]
    batch2 = ["CCCC", "DDDD"]
    rng = DummyRNG()
    out1, out2 = batch_crossover(batch1[:], batch2[:], single_crossover_rate=1.0, batch_crossover_rate=1.0, rng=rng)
    # Each pair: single_point_crossover at point 2, then swap
    # "AAAA" + "CCCC" -> "AAAC", "CCCA" then swap -> out1: "CCCA", out2: "AAAC"
    # "BBBB" + "DDDD" -> "BBBD", "DDDB" then swap -> out1: "DDDB", out2: "BBBD"
    assert sorted(out1) == sorted(["CCCA", "DDDB"])
    assert sorted(out2) == sorted(["AAAC", "BBBD"])


def test_batch_crossover_options():
    """Test that batch_crossover shuffling does not change the set of possible outputs."""
    batch1 = ["A" * 5 for _ in range(10)]
    batch2 = ["C" * 5 for _ in range(10)]
    all_possible = ["A" * 5, "C" * 5, "A" * 3 + "C" * 2, "C" * 3 + "A" * 2]
    rng = DummyRNG()
    out1, out2 = batch_crossover(batch1[:], batch2[:], rng=rng)
    assert len(out1) == len(batch1)
    assert len(out2) == len(batch2)
    assert set(out1).issubset(all_possible)
    assert set(out2).issubset(all_possible)


def test_mutate_batch_no_mutation():
    """Test that mutate_batch returns the original batch when mutation_rate is 0."""
    original_batch = ["AAAA", "CCCC", "GGGG"]
    # Use a simple vocabulary where each amino acid has alternatives.
    custom_vocab = {
        "A": "AB",
        "C": "CD",
        "G": "GH",
    }
    rng = random.Random(42)
    mutated_batch = mutate_batch(
        original_batch, mutation_rate=0.0, aa_vocabularies=[custom_vocab] * len(original_batch), rng=rng
    )
    assert mutated_batch == original_batch, "Batch should be unchanged when mutation_rate is 0."


def test_mutate_batch_full_mutation():
    """Test that mutate_batch fully mutates each mutable position in each sequence when mutation_rate is 1."""
    original_batch = ["AAAA", "CCCC", "GGGG"]
    # Define a vocabulary such that every letter has exactly one alternative (different from itself)
    custom_vocab = {
        "A": "AB",  # alternative for A is B
        "C": "CD",  # alternative for C is D
        "G": "GH",  # alternative for G is H
    }
    rng = random.Random(42)
    mutated_batch = mutate_batch(
        original_batch, mutation_rate=1.0, aa_vocabularies=[custom_vocab] * len(original_batch), rng=rng
    )
    assert sorted(mutated_batch) == sorted(
        ["BBBB", "DDDD", "HHHH"]
    ), "Each sequence should be fully mutated to its alternative amino acids."


def test_mutate_batch_empty():
    """Test that mutate_batch returns an empty batch when given an empty list."""
    rng = random.Random(42)
    mutated_batch = mutate_batch([], mutation_rate=0.5, aa_vocabularies=[], rng=rng)
    assert mutated_batch == [], "Empty batch should return an empty list."
