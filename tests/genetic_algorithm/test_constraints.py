"""Tests for genetic algorithm constraint handlers."""

import pytest

from boat.genetic_algorithm.constraints import (
    CDRMutationConstraintHandler,
    CompositeConstraintHandler,
    MaxMutationConstraintHandler,
    MutationConstraintHandler,
    create_constraint_handler,
)


def test_base_constraint_handler():
    """Test that base constraint handler returns vocabulary unchanged."""
    handler = MutationConstraintHandler()
    aa_vocabulary = {0: "XYZ", 1: "ABC", 2: "DEF"}
    result = handler.adjust_vocabulary("", aa_vocabulary)

    assert result == aa_vocabulary
    assert result is aa_vocabulary  # Should return same object


### Test cases for MaxMutationConstraintHandler ###


def test_max_mutation_constraint_handler_below_limit():
    """Test max mutation constraint when mutations are below limit."""
    parental = "AAAAAA"
    sequence = "AAXAAA"  # 1 mutation at position 2

    handler = MaxMutationConstraintHandler(parental_sequence=parental, max_mutations=2)

    aa_vocabulary = {i: "XYZ" for i in range(6)}
    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    # Should return vocabulary unchanged since we're below limit
    assert result == aa_vocabulary


def test_max_mutation_constraint_handler_at_limit():
    """Test max mutation constraint when mutations are at the limit."""
    parental = "AAAAAA"
    sequence = "XXYAAA"  # 3 mutations at positions 0,1,2

    handler = MaxMutationConstraintHandler(parental_sequence=parental, max_mutations=3)

    aa_vocabulary = {i: "XYZ" for i in range(6)}
    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    # invalid positions should be removed
    invalid_positions = [3, 4, 5]
    for pos in invalid_positions:
        assert pos not in result

    # the other positions should remain
    for i in range(6):
        if i not in invalid_positions:
            assert i in result


def test_max_mutation_constraint_handler_check_constraint_met():
    """Test check_constraint_met method of MaxMutationConstraintHandler."""
    parental = "AAAAAA"
    sequence_good = "AAXAAA"  # 1 mutation at position 2
    sequence_bad = "XXYAAA"  # 3 mutations at positions 0,1,2
    max_mutations = 2

    handler = MaxMutationConstraintHandler(parental_sequence=parental, max_mutations=max_mutations)

    assert handler.check_constraint_met(sequence_good) is True
    assert handler.check_constraint_met(sequence_bad) is False


def test_max_mutation_constraint_handler_repair_sequence():
    """Test repair_sequence method of MaxMutationConstraintHandler."""
    parental = "AAAAAA"
    sequence_good = "AAXAAA"  # 1 mutation at position 2
    sequence_bad = "XXYAAA"  # 3 mutations at positions 0
    max_mutations = 2

    handler = MaxMutationConstraintHandler(parental_sequence=parental, max_mutations=max_mutations)

    repaired_sequence_good = handler.repair_sequence(sequence_good)
    repaired_sequence_bad = handler.repair_sequence(sequence_bad)

    # Good sequence should remain unchanged
    assert repaired_sequence_good == sequence_good

    # Count mutations in repaired bad sequence
    mutation_count = sum(1 for p, s in zip(parental, repaired_sequence_bad) if p != s)
    assert mutation_count <= 2  # Should have at most 2 mutations


### Test cases for CDRMutationConstraintHandler ###


def test_cdr_constraint_handler_below_limit():
    """Test CDR constraint when mutations are below limit."""
    parental = "AAAAAAAAAA"  # positions 0-9
    sequence = "AXAAAAAAAA"  # 1 mutation at position 1

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 3}, cdr_positions={"CDR1": [0, 1, 2, 3, 4]}
    )

    aa_vocabulary = {i: "AXYZ" for i in range(10)}
    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    # Should return vocabulary unchanged since we're below limit
    assert result == aa_vocabulary


def test_cdr_constraint_handler_at_limit():
    """Test CDR constraint when mutations are at the limit."""
    parental = "AAAAAAAAAA"
    sequence = "XXYAAAAAAAA"  # 3 mutations at positions 0,1,2

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 3}, cdr_positions={"CDR1": [0, 1, 2, 3, 4]}
    )

    aa_vocabulary = {i: "AXYZ" for i in range(10)}
    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    # invalid positions should be removed
    invalid_positions = [3, 4]
    for pos in invalid_positions:
        assert pos not in result

    # the other positions should remain
    for i in range(10):
        if i not in invalid_positions:
            assert i in result


def test_cdr_constraint_handler_multiple_cdrs():
    """Test CDR constraint with multiple CDRs."""
    parental = "AAAAAAAAAAAAAA"  # 14 positions
    sequence = "XXYAAAAXAAAXYA"  # CDR1: 3 muts (0,1,2), CDR2: 1 mut (7), CDR3: 2 muts (10,11)

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations_per_cdr={"CDR1": 3, "CDR2": 2, "CDR3": 2},
        cdr_positions={"CDR1": [0, 1, 2, 3], "CDR2": [6, 7, 8, 9], "CDR3": [10, 11, 12, 13]},
    )

    aa_vocabulary = {i: "XYZ" for i in range(14)}
    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    # invalid positions should be removed
    invalid_positions = [3, 10, 13]
    for pos in invalid_positions:
        assert pos not in result

    # the other positions should remain
    for i in range(14):
        if i not in invalid_positions:
            assert i in result


def test_cdr_constraint_handler_preserves_vocabulary():
    """Test that CDR constraint doesn't modify the original vocabulary."""
    parental = "AAAA"
    sequence = "XXAA"

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 2}, cdr_positions={"CDR1": [0, 1, 2, 3]}
    )

    original_vocab = {0: "X", 1: "Y", 2: "Z", 3: "W"}
    vocab_copy = original_vocab.copy()

    result = handler.adjust_vocabulary(sequence, original_vocab)

    # Original vocabulary should be unchanged
    assert original_vocab == vocab_copy
    # Result should be different
    assert result != original_vocab


def test_cdr_constraint_handler_check_constraints_met():
    """Test check_constraint_met method of CDRMutationConstraintHandler."""
    parental = "AAAAAA"
    sequence_good = "AAXAYA"  # meets constraints
    sequence_bad = "XXYAXX"  # does not meet constraints
    max_mutations_per_cdr = {"CDR1": 2, "CDR2": 1}
    cdr_positions = {"CDR1": [0, 1, 2], "CDR2": [3, 4, 5]}

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations_per_cdr=max_mutations_per_cdr,
        cdr_positions=cdr_positions,
    )

    assert handler.check_constraint_met(sequence_good) is True
    assert handler.check_constraint_met(sequence_bad) is False


def test_cdr_constraint_handler_repair_sequence():
    """Test repair_sequence method of CDRMutationConstraintHandler."""
    parental = "AAAAAA"
    sequence_good = "AAXAYA"  # meets constraints
    sequence_bad = "XXYAXX"  # does not meet constraints
    max_mutations_per_cdr = {"CDR1": 2, "CDR2": 1}
    cdr_positions = {"CDR1": [0, 1, 2], "CDR2": [3, 4, 5]}

    handler = CDRMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations_per_cdr=max_mutations_per_cdr,
        cdr_positions=cdr_positions,
    )

    repaired_sequence_good = handler.repair_sequence(sequence_good)
    repaired_sequence_bad = handler.repair_sequence(sequence_bad)

    # Good sequence should remain unchanged
    assert repaired_sequence_good == sequence_good

    # Check that repaired bad sequence meets constraints
    assert handler.check_constraint_met(repaired_sequence_bad) is True


### Test cases for CompositeConstraintHandler ###


def test_composite_constraint_handler_empty():
    """Test composite handler with no constraints."""
    handler = CompositeConstraintHandler([])
    sequence = "ACDEFG"
    aa_vocabulary = {i: "XYZ" for i in range(6)}

    result = handler.adjust_vocabulary(sequence, aa_vocabulary)

    assert result == aa_vocabulary


def test_composite_constraint_handler_single():
    """Test composite handler with a single constraint."""
    parental = "AAAA"
    sequence = "XXAA"

    cdr_handler = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 2}, cdr_positions={"CDR1": [0, 1, 2, 3]}
    )

    composite = CompositeConstraintHandler([cdr_handler])
    aa_vocabulary = {i: "XYZ" for i in range(4)}

    result = composite.adjust_vocabulary(sequence, aa_vocabulary)

    # Should behave same as single handler
    assert 2 not in result
    assert 3 not in result
    assert 0 in result
    assert 1 in result


def test_composite_constraint_handler_multiple():
    """Test composite handler with multiple constraints applied sequentially."""
    parental = "AAAAAA"
    sequence = "XXYAAA"

    # First constraint: removes position 3 and 4
    cdr_handler1 = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 3}, cdr_positions={"CDR1": [0, 1, 2, 3, 4]}
    )

    # Second constraint: removes position 5 (different CDR)
    cdr_handler2 = CDRMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations_per_cdr={"CDR2": 0},  # 0 mutations allowed
        cdr_positions={"CDR2": [5]},
    )

    composite = CompositeConstraintHandler([cdr_handler1, cdr_handler2])
    aa_vocabulary = {i: "XYZ" for i in range(6)}

    result = composite.adjust_vocabulary(sequence, aa_vocabulary)

    # Both constraints should be applied
    assert 3 not in result  # Removed by first constraint
    assert 4 not in result  # Removed by first constraint
    assert 5 not in result  # Removed by second constraint
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_composite_constraint_handler_check_constraints_met():
    """Test check_constraint_met method of CompositeConstraintHandler."""
    parental = "AAAAAA"
    sequence_bad = "XXYAXX"  # does not meet constraints
    sequence_good = "AAXAYA"  # meets constraints

    handler1 = MaxMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations=2,
    )

    handler2 = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 2}, cdr_positions={"CDR1": [0, 1, 2]}
    )

    composite = CompositeConstraintHandler([handler1, handler2])

    assert composite.check_constraint_met(sequence_good) is True
    assert composite.check_constraint_met(sequence_bad) is False


def test_composite_constraint_handler_repair_sequence():
    """Test repair_sequence method of CompositeConstraintHandler."""
    parental = "AAAAAA"
    sequence_bad_1 = "XXYAXX"  # does not meet either constraint
    sequence_bad_2 = "AAXAXX"  # does not meet max_mutation constraint
    sequence_bad_3 = "XXYAAA"  # does not meet CDR constraint
    sequence_good = "AAXAYA"  # meets constraints

    handler1 = MaxMutationConstraintHandler(
        parental_sequence=parental,
        max_mutations=3,
    )

    handler2 = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 2}, cdr_positions={"CDR1": [0, 1, 2]}
    )

    composite = CompositeConstraintHandler([handler1, handler2])

    repaired_sequence_bad_1 = composite.repair_sequence(sequence_bad_1)
    repaired_sequence_bad_2 = composite.repair_sequence(sequence_bad_2)
    repaired_sequence_bad_3 = composite.repair_sequence(sequence_bad_3)

    # Check that repaired sequence meets both constraints
    assert composite.repair_sequence(sequence_good) == sequence_good
    assert composite.check_constraint_met(repaired_sequence_bad_1) is True
    assert composite.check_constraint_met(repaired_sequence_bad_2) is True
    assert composite.check_constraint_met(repaired_sequence_bad_3) is True


def test_composite_with_cdr_and_max_mutations():
    """Test composite handler with both CDR and max mutation constraints."""
    parental = "AAAAAA"
    sequence = "XXAAAA"

    max_mutation_handler = MaxMutationConstraintHandler(parental_sequence=parental, max_mutations=2)

    cdr_handler = CDRMutationConstraintHandler(
        parental_sequence=parental, max_mutations_per_cdr={"CDR1": 3}, cdr_positions={"CDR1": [0, 1, 2, 3, 4]}
    )

    composite = CompositeConstraintHandler([max_mutation_handler, cdr_handler])
    aa_vocabulary = {i: "XYZ" for i in range(6)}

    result = composite.adjust_vocabulary(sequence, aa_vocabulary)

    # Max mutations overrule CDR constraint here
    invalid_positions = [2, 3, 4, 5]
    for pos in invalid_positions:
        assert pos not in result

    for i in range(6):
        if i not in invalid_positions:
            assert i in result


@pytest.mark.parametrize(
    "parental_sequence,max_mutations,max_mutations_per_cdr,cdr_positions,expected_handler_type",
    [
        (None, None, None, None, MutationConstraintHandler),
        ("AAAA", 2, None, None, MaxMutationConstraintHandler),
        ("AAAA", None, {"CDR1": 2}, {"CDR1": [0, 1, 2, 3]}, CDRMutationConstraintHandler),
        ("AAAA", 2, {"CDR1": 2}, {"CDR1": [0, 1, 2, 3]}, CompositeConstraintHandler),
    ],
)
def test_create_constraint_handler(
    parental_sequence, max_mutations, max_mutations_per_cdr, cdr_positions, expected_handler_type
):
    """Test create_constraint_handler function for various configurations."""
    handler = create_constraint_handler(
        parental_sequence=parental_sequence,
        max_mutations=max_mutations,
        max_mutations_per_cdr=max_mutations_per_cdr,
        cdr_positions=cdr_positions,
    )

    assert isinstance(handler, expected_handler_type)
