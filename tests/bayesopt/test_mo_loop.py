"""Integration test for the Multi-Objective BO loop."""

import logging
import random

import pytest
import torch

from boat.bayesopt.loop.utils import compute_hypervolume
from boat.bayesopt.mo_loop import MOBayesOptOnSequences
from boat.scoring_function.fake import FakeMultimodalScoringFunction
from boat.scoring_function.interface import MultiObjectiveScoringFunction


@pytest.fixture
def so_scoring_function():
    """Fixture to create a fresh single-objective scoring function for each test."""
    sf = FakeMultimodalScoringFunction(weights={"A": 1.2, "K": 1.0, "Q": 0.8})
    return MultiObjectiveScoringFunction([sf])


@pytest.fixture
def mo_scoring_function():
    """Fixture to create a fresh multi-objective scoring function for each test."""
    weights1 = {"A": 1.2, "K": 1.0, "Q": 0.8}
    weights2 = {"A": 1.0, "G": 2.1, "C": 1.5}
    sf1 = FakeMultimodalScoringFunction(weights=weights1)
    sf2 = FakeMultimodalScoringFunction(weights=weights2)
    return MultiObjectiveScoringFunction([sf1, sf2])


@pytest.fixture(params=["one-hot", "ablang2", "blosum", "bag_of_aas"])
def encoding_str(request):
    """Fixture to parametrize over different encoding strategies."""
    return request.param


@pytest.mark.parametrize("acquisition_str, batch_size", [("EI", 1), ("qEI", 2)])  # Standard EI  # Batched qEI
def test_so_bayesopt_loop_integration(acquisition_str, batch_size, so_scoring_function, encoding_str):
    """Test the integration of the single-objective Bayesian optimization loop on sequences."""
    initial_sequences = ["AKTAKT", "TCCTCQ", "TTTAAA", "CCCTTT"]
    initial_length = len(initial_sequences)
    initial_scores = so_scoring_function(initial_sequences)

    # Create a simple vocabulary mapping (for test, just use all AA letters)
    vocab = {c: "ACGKQT" for c in "ACGKQT"}

    # GA parameters for a quick run
    ga_params = {
        "aa_vocabulary": vocab,
        "max_rounds": 10,
        "tournament_size": 4,
        "population_size": 20,
        "mutation_rate": 0.8,
        "crossover_rate": 0.2,
        "repetitions": 2,
        "logging_level": logging.CRITICAL,
        "probability_matrix": None,
    }

    # Add batch-specific parameters if needed
    batch_params = {}
    if acquisition_str.startswith("q"):
        batch_params["batch_size"] = batch_size

    # Create the loop instance
    loop = MOBayesOptOnSequences(
        initial_sequences=initial_sequences,
        initial_scores=initial_scores,
        objective_function=so_scoring_function,
        encoding_str=encoding_str,
        model_dict={"regression": "TanimotoGP"},
        acquisition_str=acquisition_str,
        vocab=vocab,
        ga_params=ga_params,
        dd_dict={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float64},
        validate_surrogates=False,  # Skip expensive validation in tests
        **batch_params,
    )

    initial_best = loop.best_observed

    # Run a few iterations of the loop
    n_iter = 6
    sequences, scores = loop.run(n_iter)

    later_best = loop.best_observed

    # Check that new sequences were added
    assert len(sequences) == initial_length + n_iter * loop.batch_size
    assert scores.shape[0] == initial_length + n_iter * loop.batch_size
    assert scores.shape[1] == 1  # single objective

    # Ensure the iteration counter is as expected
    assert loop.iteration == n_iter

    # Check that the __repr__ output contains key information
    repr_str = repr(loop)
    assert "MOBayesOptOnSequences" in repr_str
    assert f"iteration={loop.iteration}" in repr_str
    assert "num_objectives=1" in repr_str
    assert later_best >= initial_best, "Best observed should not decrease in a valid optimization run"


@pytest.mark.parametrize("acquisition_str, batch_size", [("EHVI", 1), ("qEHVI", 2), ("qNEHVI", 2)])
def test_mo_bayesopt_loop_integration(mo_scoring_function, acquisition_str, batch_size, encoding_str):
    """Test the integration of the multi-objective Bayesian optimization loop on sequences."""
    initial_sequences = ["AKTAKT", "TCCTCQ", "TTTAAA", "CCCTTT"]
    initial_length = len(initial_sequences)
    initial_scores = mo_scoring_function(initial_sequences)

    # Create a simple vocabulary mapping (for test, just use all AA letters)
    vocab = {c: "ACGKQT" for c in "ACGKQT"}

    # GA parameters for a quick run
    ga_params = {
        "aa_vocabulary": vocab,
        "max_rounds": 10,
        "tournament_size": 4,
        "population_size": 20,
        "mutation_rate": 0.8,
        "crossover_rate": 0.2,
        "repetitions": 2,
        "logging_level": logging.CRITICAL,
    }

    model_str = "TanimotoGP"

    # Create the loop instance
    loop = MOBayesOptOnSequences(
        initial_sequences=initial_sequences,
        initial_scores=initial_scores,
        objective_function=mo_scoring_function,
        encoding_str=encoding_str,
        model_dict={"regression": model_str},
        acquisition_str=acquisition_str,
        vocab=vocab,
        ga_params=ga_params,
        dd_dict={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float64},
        batch_size=batch_size,
    )

    initial_hv = compute_hypervolume(loop.pareto_front, loop.ref_point)

    # Run a few iterations of the loop
    n_iter = 6
    sequences, scores = loop.run(n_iter)

    # Check that new sequences were added
    assert len(sequences) == initial_length + n_iter * loop.batch_size
    assert scores.shape[0] == initial_length + n_iter * loop.batch_size
    assert scores.shape[1] == 2  # two objectives

    # Check that the Pareto front is not empty and has correct shape
    assert loop.pareto_front.shape[1] == 2
    assert loop.pareto_front.shape[0] >= 1

    # Ensure the iteration counter is as expected
    assert loop.iteration == n_iter

    # Check that the __repr__ output contains key information
    repr_str = repr(loop)
    assert "MOBayesOptOnSequences" in repr_str
    assert f"iteration={loop.iteration}" in repr_str
    assert "num_objectives=2" in repr_str

    # Check that hypervolume is a non-negative float
    hv = compute_hypervolume(loop.pareto_front, loop.ref_point)
    assert isinstance(hv, float)
    assert hv >= 0.0
    assert hv >= initial_hv, "Hypervolume should not decrease in a valid optimization run"


@pytest.mark.parametrize(
    "acquisition_str,batch_size", [("EI", 1), ("qEI", 2), ("EHVI", 1), ("qEHVI", 2), ("qNEHVI", 2)]
)
def test_bayesopt_loop_seeding(acquisition_str, batch_size, mo_scoring_function, so_scoring_function, encoding_str):
    """Test that seeding produces reproducible results for both regular and batch modes."""
    # Define initial sequences and scores
    initial_sequences = ["AKTAKT", "TCCTCQ", "TTTAAA", "CCCTTT"]

    if "HV" in acquisition_str:
        scoring_function = mo_scoring_function
    else:
        scoring_function = so_scoring_function

    initial_scores = scoring_function(initial_sequences)

    # Create a simple vocabulary mapping (for test, just use all AA letters)
    vocab = {c: "ACGKQT" for c in "ACGKQT"}

    # GA parameters for a quick run
    ga_params = {
        "aa_vocabulary": vocab,
        "max_rounds": 5,
        "tournament_size": 4,
        "population_size": 10,
        "mutation_rate": 0.8,
        "crossover_rate": 0.2,
        "repetitions": 5,
        "logging_level": logging.CRITICAL,
    }

    # Add batch-specific parameters if needed
    batch_params = {}
    if acquisition_str.startswith("q"):
        batch_params["batch_size"] = batch_size

    # First run
    loop1 = MOBayesOptOnSequences(
        initial_sequences=initial_sequences,
        initial_scores=initial_scores,
        objective_function=scoring_function,
        encoding_str=encoding_str,
        model_dict={"regression": "TanimotoGP"},
        acquisition_str=acquisition_str,
        vocab=vocab,
        rng=random.Random(101),
        ga_params=ga_params.copy(),
        dd_dict={"device": torch.device("cpu"), "dtype": torch.float64},
        **batch_params,
    )
    seqs1, scores1 = loop1.run(3)

    initial_sequences = ["AKTAKT", "TCCTCQ", "TTTAAA", "CCCTTT"]

    # Second run (should be identical)
    loop2 = MOBayesOptOnSequences(
        initial_sequences=initial_sequences,
        initial_scores=initial_scores,
        objective_function=scoring_function,
        encoding_str=encoding_str,
        model_dict={"regression": "TanimotoGP"},
        acquisition_str=acquisition_str,
        vocab=vocab,
        rng=random.Random(101),
        ga_params=ga_params.copy(),
        dd_dict={"device": torch.device("cpu"), "dtype": torch.float64},
        **batch_params,
    )
    seqs2, scores2 = loop2.run(3)

    # Assert reproducibility
    assert seqs1 == seqs2, f"Sequences not identical for {acquisition_str}"
    assert torch.equal(scores1, scores2), f"Scores not identical for {acquisition_str}"
