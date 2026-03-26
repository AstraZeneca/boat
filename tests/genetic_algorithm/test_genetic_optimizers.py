"""Test genetic algorithm functionalities with refactored API."""
import random

import pytest

from boat.genetic_algorithm.genetic_optimizers import (
    BatchGeneticAlgorithm,
    GeneticAlgorithm,
)
from boat.genetic_algorithm.utils import count_mutations
from boat.genetic_algorithm.vocabularies import (
    AA_VOCABULARY,
    aa_to_positional_vocabulary,
    aa_vocabulary_complete,
)

rng = random.Random(42)


@pytest.fixture
def target_sequence():
    """Fixture for the target sequence used as wild type."""
    return "ACACACACAC"


@pytest.fixture
def evaluate_sequences(target_sequence):
    """Fixture to provide a simple scoring function for sequences.

    The score is defined as the number of positions matching the target.
    """

    def _evaluate_sequences(sequences, target=target_sequence):
        scores = [sum(1 for a, b in zip(seq, target) if a == b) for seq in sequences]
        return scores

    return _evaluate_sequences


@pytest.fixture
def initial_population(target_sequence):
    """Fixture to generate an initial population of sequences."""
    population_size = 100
    seq_length = len(target_sequence)
    return ["".join(rng.choices(AA_VOCABULARY, k=seq_length)) for _ in range(population_size)]


def make_genetic_algorithm(
    ga_class,
    initial_population,
    evaluate_sequences,
    target_sequence,
    batch_size=None,
    max_mutations=None,
    liability_filtering=False,
    liability_threshold=None,
):
    """Fixture to initialise a genetic algorithm."""
    vocab = aa_vocabulary_complete()
    vocab = aa_to_positional_vocabulary(target_sequence, vocab)
    params = {
        "initial_population": initial_population,
        "scoring_function": evaluate_sequences,
        "aa_vocabulary": vocab,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "repetitions": 2,
        "tournament_size": 5,
        "rng": rng,
        "probability_matrix": None,
    }
    if batch_size is not None:
        params["batch_size"] = batch_size
    if max_mutations is not None:
        params["parental_sequence"] = target_sequence
        params["max_mutations"] = max_mutations
    if liability_filtering:
        params["liability_filtering"] = True
        params["liability_threshold"] = liability_threshold
    return ga_class(**params)


def _get_final_sequences_for_class(genetic_algorithm, ga_class):
    """Run GA and extract final sequences based on class type."""
    class_configs = {
        GeneticAlgorithm: ({"max_rounds": 10}, "all_sequences"),
        BatchGeneticAlgorithm: ({"max_rounds": 10}, "all_batches"),
    }

    run_params, result_key = class_configs[ga_class]
    results = genetic_algorithm.run(**run_params)
    return results[result_key]


def _verify_sequences_clean(sequences, liability_patterns, ga_class):
    """Verify sequences don't contain liability patterns."""
    # Handle batch vs individual sequences
    seq_iterator = (seq for batch in sequences for seq in batch) if ga_class == BatchGeneticAlgorithm else sequences

    for seq in seq_iterator:
        for pattern in liability_patterns:
            assert (
                pattern not in seq
            ), f"Sequence {seq} contains liability pattern '{pattern}' that should have been filtered out"


def test_genetic_algorithm(initial_population, evaluate_sequences, target_sequence):
    """Test the evolutionary improvement using a fixed number of generations."""
    genetic_algorithm = make_genetic_algorithm(
        ga_class=GeneticAlgorithm,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
    )

    results = genetic_algorithm.run(max_rounds=50)

    # Check that keys are returned as expected.
    for key in ["all_sequences", "all_scores"]:
        assert key in results, f"Missing key '{key}' in results."

    # Assert that the number of sequences and scores match.
    assert len(results["all_sequences"]) == len(results["all_scores"])

    # As a simple convergence test, the wild-type sequence (target_sequence) should eventually appear.
    assert target_sequence in results["all_sequences"], "Wild-type sequence not found in final population."


def test_genetic_algorithm_max_mutations(initial_population, evaluate_sequences, target_sequence):
    """Test that the GA returns sequences with at most n mutations w.r.t. the wild type."""
    max_mut = 4

    genetic_algorithm = make_genetic_algorithm(
        ga_class=GeneticAlgorithm,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
        max_mutations=max_mut,
    )

    results = genetic_algorithm.run(max_rounds=30)

    # Check that keys are returned as expected.
    for key in ["all_sequences", "all_scores"]:
        assert key in results, f"Missing key '{key}' in results."

    # Assert that the number of evaluated sequences and scores match.
    assert len(results["all_sequences"]) == len(results["all_scores"])

    # Assert that every generated sequence has exactly fixed_n_mut mutations relative to the wild-type.
    recent_sequences = results["all_sequences"][-30:]
    for seq in recent_sequences:
        muts = count_mutations(seq, target_sequence)
        assert muts <= max_mut, f"Sequence {seq} has {muts} mutations; expected at most {max_mut}"


def test_batch_genetic_algorithm(initial_population, evaluate_sequences, target_sequence):
    """Test batch GA evolutionary improvement using a fixed number of generations."""
    batch_size = 3

    batch_ga = make_genetic_algorithm(
        ga_class=BatchGeneticAlgorithm,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
        batch_size=batch_size,
    )

    results = batch_ga.run(max_rounds=10)

    # Check that keys are returned as expected.
    for key in ["all_batches", "all_scores"]:
        assert key in results, f"Missing key '{key}' in results."

    # Assert that the number of batches and scores match.
    assert len(results["all_batches"]) == len(results["all_scores"])

    # Check that each batch has the correct batch size
    for batch in results["all_batches"]:
        assert len(batch) == batch_size


def test_batch_genetic_algorithm_size_1(initial_population, evaluate_sequences, target_sequence):
    """Test batch GA evolutionary improvement for a batch size of 1."""
    batch_size = 1

    batch_ga = make_genetic_algorithm(
        ga_class=BatchGeneticAlgorithm,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
        batch_size=batch_size,
    )

    results = batch_ga.run(max_rounds=10)

    # Check that keys are returned as expected.
    for key in ["all_batches", "all_scores"]:
        assert key in results, f"Missing key '{key}' in results."

    # Assert that the number of batches and scores match.
    assert len(results["all_batches"]) == len(results["all_scores"])

    # Check that each batch has the correct batch size
    for batch in results["all_batches"]:
        assert len(batch) == batch_size


def test_batch_genetic_algorithm_max_mutations(initial_population, evaluate_sequences, target_sequence):
    """Test that the batch GA returns sequences with at most n mutations w.r.t. the wild type."""
    max_mut = 4
    batch_size = 3

    batch_ga = make_genetic_algorithm(
        ga_class=BatchGeneticAlgorithm,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
        batch_size=batch_size,
        max_mutations=max_mut,
    )

    results = batch_ga.run(max_rounds=10)

    # Check that every sequence in every batch has at most max_mut mutations
    for batch in results["all_batches"][-10:]:
        for seq in batch:
            muts = count_mutations(seq, target_sequence)
            assert muts <= max_mut, f"Sequence {seq} has {muts} mutations; expected at most {max_mut}"


@pytest.mark.parametrize(
    "ga_class",
    [
        GeneticAlgorithm,
        BatchGeneticAlgorithm,
    ],
)
def test_genetic_algorithm_with_liability_filtering(ga_class, initial_population, evaluate_sequences, target_sequence):
    """Test that liability filtering correctly removes sequences."""
    liability_patterns = ["C", "M", "NG", "DS"]

    # Common parameters for all GA classes
    genetic_algorithm = make_genetic_algorithm(
        ga_class=ga_class,
        initial_population=initial_population,
        evaluate_sequences=evaluate_sequences,
        target_sequence=target_sequence,
        batch_size=2 if ga_class == BatchGeneticAlgorithm else None,
        max_mutations=None,
        liability_filtering=True,
        liability_threshold=0.5,  # No liability patterns above 5 (10 * 0.5)
    )

    # Run and get final sequences
    final_sequences = _get_final_sequences_for_class(genetic_algorithm, ga_class)

    # Verify no liability patterns exist
    _verify_sequences_clean(final_sequences, liability_patterns, ga_class)
