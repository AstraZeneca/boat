"""Tests for the amino acid vocabularies."""
import blosum
import pytest

from boat.genetic_algorithm.vocabularies import (
    AA_VOCABULARY,
    aa_to_positional_vocabulary,
    aa_vocabulary_blosum,
    aa_vocabulary_complete,
    aa_vocabulary_reduced,
    count_mutation_permutations,
    positional_vocabulary,
)

# --- Tests for AA-based vocabularies ---------------------------------------


def test_aa_vocabulary_complete():
    """Test that aa_vocabulary_complete returns the complete alphabet for each key."""
    vocab = aa_vocabulary_complete()
    for aa in AA_VOCABULARY:
        assert aa in vocab, f"Missing key: {aa}"
        assert vocab[aa] == AA_VOCABULARY, f"For {aa}, expected {AA_VOCABULARY} but got {vocab[aa]}"


def test_aa_vocabulary_reduced_default():
    """Test that aa_vocabulary_reduced by default excludes 'C'."""
    vocab = aa_vocabulary_reduced()
    # Expected: for letters not "CM", value is AA_VOCABULARY without "C" and "M".
    expected = "".join([a for a in AA_VOCABULARY if a not in ["C", "M"]])
    for aa in AA_VOCABULARY:
        if aa == "C":
            assert vocab[aa] == aa, f"Expected {aa} to map to itself."
        elif aa == "M":
            assert vocab[aa] == aa, f"Expected {aa} to map to itself."
        else:
            assert vocab[aa] == expected, f"For {aa}, expected {expected} but got {vocab[aa]}"


def test_aa_vocabulary_reduced_custom():
    """Test aa_vocabulary_reduced with custom excluded amino acids."""
    exclude = "DE"
    vocab = aa_vocabulary_reduced(exclude_aas=exclude)
    expected = "".join([a for a in AA_VOCABULARY if a not in exclude])
    for aa in AA_VOCABULARY:
        if aa in exclude:
            assert vocab[aa] == aa, f"Expected {aa} to map to itself when excluded."
        else:
            assert vocab[aa] == expected, f"For {aa}, expected {expected} but got {vocab[aa]}"


@pytest.mark.parametrize("similarity", [45, 50, 62, 80, 90])
def test_aa_vocabulary_blosum(similarity):
    """Test that aa_vocabulary_blosum returns keys and values only from AA_VOCABULARY and excludes default letters."""
    vocab = aa_vocabulary_blosum(similarity, exclude_aas="BJZX*")
    blosum_matrix = blosum.BLOSUM(similarity, default=0)
    for key, value in vocab.items():
        assert key in AA_VOCABULARY, f"Key {key} not in AA_VOCABULARY."
        for aa in value:
            assert aa in AA_VOCABULARY, f"Value {aa} for key {key} not in AA_VOCABULARY."
            assert blosum_matrix[key][aa] > 0, f"Expected positive similarity for {key} and {aa} in BLOSUM matrix."


# --- Tests for position-based vocabularies ------------------------------------


def test_positional_vocabulary():
    """
    Test positional_vocabulary.

    Given a sequence and mutations dict (keys are 0-indexed positions),
    ensure that each position's vocabulary is the set union of the mutation letters (if any)
    and the original amino acid at that position.
    """
    sequence = "ACDE"
    mutations = {
        0: "XYZ",  # For position 0: union of "XYZ" and "A"
        2: "PQ",  # For position 2: union of "PQ" and "D"
    }
    vocab = positional_vocabulary(sequence, mutations)
    # position 0: mutations.get(0, "") returns "XYZ", plus "A"
    expected0 = set("XYZA")
    assert set(vocab[0]) == expected0, f"Expected {expected0} at pos0, got {vocab[0]}"
    # position 1: key 1 not in mutations => only "C"
    expected1 = set("C")
    assert set(vocab[1]) == expected1, f"Expected {expected1} at pos1, got {vocab[1]}"
    # position 2: key=2 returns "PQ" plus letter "D"
    expected2 = set("PQD")
    assert set(vocab[2]) == expected2, f"Expected {expected2} at pos2, got {vocab[2]}"
    # position 3: key=3 not provided, so only "E"
    expected3 = set("E")
    assert set(vocab[3]) == expected3, f"Expected {expected3} at pos3, got {vocab[3]}"


# --- Tests for AA to positional vocabulary transformation --------------------


def test_aa_to_positional_vocabulary():
    """
    Test aa_to_positional_vocabulary by ensuring it converts an AA-based vocabulary into a positional one.

    Given a sequence and a dictionary mapping each amino acid to a list of alternatives,
    the output at each position should correspond to the list for the amino acid at that position.
    """
    sequence = "ACDE"
    aa_vocab = {"A": ["X", "Y"], "C": ["Z"], "D": ["W"], "E": ["V"]}
    vocab = aa_to_positional_vocabulary(sequence, aa_vocab)
    assert vocab[0] == aa_vocab["A"], f"Expected {aa_vocab['A']} at pos0, got {vocab[0]}"
    assert vocab[1] == aa_vocab["C"], f"Expected {aa_vocab['C']} at pos1, got {vocab[1]}"
    assert vocab[2] == aa_vocab["D"], f"Expected {aa_vocab['D']} at pos2, got {vocab[2]}"
    assert vocab[3] == aa_vocab["E"], f"Expected {aa_vocab['E']} at pos3, got {vocab[3]}"


def test_count_mutation_permutations_aa_based():
    """
    Test count_mutation_permutations using an AA-based vocabulary.

    For an AA-based vocabulary, the function first converts it into a positional vocabulary.
    For each position, the number of mutation choices is len(aa_vocab[pos]) - 1.
    """
    sequence = "AC"
    # AA-based vocabulary: keys are amino acids and values are lists of alternatives.
    # For pos0 ('A'): list=["X","Y"] -> choices = 2
    # For pos1 ('C'): list=["Z"] -> choices = 1
    aa_vocab = {"A": ["A", "X", "Y"], "C": ["C", "Z"]}
    # With n_muts=1, the possible combinations are:
    #  - Mutate position 0: 1 possibility.
    #  - Mutate position 1: 0 possibility.
    # Total = 1.
    result = count_mutation_permutations(sequence, aa_vocab, 1)
    assert result == 3, f"Expected 3 but got {result}"


def test_count_mutation_permutations_positional():
    """
    Test count_mutation_permutations using a positional vocabulary.

    Here the vocabulary keys are positions and values are strings.
    For each position, mutation choices equal len(vocab[pos]) - 1.
    """
    sequence = "XYZ"
    # Positional vocabulary: each string includes the original letter and one alternative.
    # For each pos: choices = 1.
    # For n_muts=2, the combinations will be:
    #   (0, 1): 1
    #   (0, 2): 1
    #   (1, 2): 1
    # Total = 3.
    vocab = {0: "XA", 1: "YB", 2: "ZC"}
    result = count_mutation_permutations(sequence, vocab, 2)
    assert result == 3, f"Expected 3 but got {result}"


def test_count_mutation_permutations_no_mutations():
    """Test `count_mutation_permutations` when n_muts is greater than the sequence length."""
    sequence = "AC"
    # Positional vocabulary: each position has two letters, so choices = 1 for each.
    vocab = {0: "AB", 1: "CD"}
    # With n_muts=3 and sequence length=2, there are no valid combinations.
    result = count_mutation_permutations(sequence, vocab, 3)
    assert result == 0, f"Expected 0 but got {result}"
