"""Tests for the liabilities functions."""

import numpy as np

from boat.biologics.liabilities import (
    GLYCOLISATION_PAYLOAD,
    LIABILITIES_GLYCO,
    LIABILITIES_NGRAM,
    LIABILITIES_SINGLE,
    filter_by_liability,
    glycolisation_liabilities,
    ngram_liabilities,
    score_sequence,
    score_sequence_list,
    single_residue_liabilities,
)


def test_single_residue_liabilities():
    """
    Test that single_residue_liabilities computes the weighted sum for risky residues.

    For example, for sequence "CCM", expect:
      Count('C') = 2, with weight 15 each  => 2 * 15 = 30
      Count('M') = 1, with weight 5.4        => 1 * 5.4 = 5.4
    Total expected = 35.4.
    """
    sequence = "CCM"
    expected = 2 * LIABILITIES_SINGLE.get("C", 0) + 1 * LIABILITIES_SINGLE.get("M", 0)
    result = single_residue_liabilities(sequence)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_ngram_liabilities():
    """
    Test ngram_liabilities.

    For sequence "NGNG", the n-gram "NG" has weight 15.
    Occurrences: "NG" appears twice (non-overlapping, using count method).
    Expected = 2 * 15 = 30.
    """
    sequence = "NGNG"
    weight_ng = LIABILITIES_NGRAM.get("NG", 0)
    expected = 2 * weight_ng
    result = ngram_liabilities(sequence)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_glycolisation_liabilities():
    """
    Test glycolisation_liabilities using a controlled pattern.

    LIABILITIES_GLYCO is defined as:
         [ "N" + t for t in "ARNDCQEGHILKMFSTWYV" ]
    For a sequence that contains "NAS", if we assume "NA" is in LIABILITIES_GLYCO:
         For glyc = "NA" and t="S", "NAS" occurs once.
    Expected = GLYCOLISATION_PAYLOAD * 1.
    """
    # Construct a sequence that contains the pattern "NAS".
    sequence = "NAS"
    # Verify that "NA" is in LIABILITIES_GLYCO.
    assert "NA" in LIABILITIES_GLYCO, "Expected 'NA' in LIABILITIES_GLYCO"
    # For glyc "NA", when t = "S", f"{glyc}{'S'}" yields "NAS".
    expected = GLYCOLISATION_PAYLOAD * sequence.count("NAS")
    result = glycolisation_liabilities(sequence)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_score_sequence():
    """
    Test score_sequence by summing up contributions from single residue, ngram, and glycolisation liabilities.

    For a sequence with no ngram or glycol patterns, e.g., "CCM":
      Expected score = single_residue_liabilities("CCM").
    """
    sequence = "CCM"
    expected = single_residue_liabilities(sequence) + ngram_liabilities(sequence) + glycolisation_liabilities(sequence)
    result = score_sequence(sequence)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_score_sequence_list():
    """
    Test score_sequence_list by providing a list of sequences.

    For example, for ["CCM", "AAA"]:
      Expected scores = [score_sequence("CCM"), score_sequence("AAA")].
    """
    sequences = ["CCM", "AAA"]
    expected = [score_sequence(seq) for seq in sequences]
    result = score_sequence_list(sequences)
    assert all(np.isclose(r, e) for r, e in zip(result, expected)), f"Expected {expected}, got {result}"


def test_filter_by_liability():
    """
    Test filter_by_liability by providing a list of sequences.

    For a given threshold (e.g., 1.0 per residue), a risky sequence such as "CCM"
    (score >> 1.0 * len("CCM") = 3) should be filtered out,
    while a safe sequence like "AAA" (score 0) should be retained.
    """
    risky_seq = "CCM"  # Expected score ~35.4 > 3 (if threshold=1.0)
    safe_seq = "AAA"  # Expected score 0 < 3
    sequence_list = [risky_seq, safe_seq]
    filtered = filter_by_liability(sequence_list, aa_threshold=1.0)
    assert risky_seq not in filtered, f"Risky sequence {risky_seq} was not filtered out."
    assert safe_seq in filtered, f"Safe sequence {safe_seq} should be retained."
