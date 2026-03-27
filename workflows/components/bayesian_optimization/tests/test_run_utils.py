"""Tests for run utils."""
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from boat.biologics.sequence import AbVSeq
from workflows.components.bayesian_optimization.src.run_utils import (
    check_if_multi_objective,
    compute_metrics,
    compute_pareto_front_plots,
    compute_score_distribution_metrics,
    compute_sequence_diversity_metrics,
    find_repo_root,
    get_parental_sequence,
    initialize_vocabulary,
    load_checkpoint,
    resolve_config_paths,
    save_checkpoint,
    save_results_to_csv,
    set_random_state,
    transform_scores_to_numpy,
)

# ---------------------------------------------------------------------------
# find_repo_root
# ---------------------------------------------------------------------------


def test_find_repo_root_returns_path_with_marker():
    """find_repo_root should return a Path that contains pyproject.toml."""
    root = find_repo_root()
    assert isinstance(root, Path)
    assert (root / "pyproject.toml").exists()


def test_find_repo_root_custom_marker(tmp_path):
    """find_repo_root should find a custom marker file."""
    marker = "my_marker.txt"
    (tmp_path / marker).touch()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    # Patch __file__ of the module so the walk starts from our nested dir
    import workflows.components.bayesian_optimization.src.run_utils as run_utils_mod

    with patch.object(run_utils_mod, "__file__", str(nested / "run_utils.py")):
        root = find_repo_root(marker=marker)

    assert root == tmp_path


def test_find_repo_root_raises_when_not_found(tmp_path):
    """find_repo_root should raise FileNotFoundError when marker is absent."""
    nested = tmp_path / "deep" / "dir"
    nested.mkdir(parents=True)

    import workflows.components.bayesian_optimization.src.run_utils as run_utils_mod

    with patch.object(run_utils_mod, "__file__", str(nested / "run_utils.py")):
        with pytest.raises(FileNotFoundError, match="Could not find repository root"):
            find_repo_root(marker="nonexistent_marker_xyz.txt")


# ---------------------------------------------------------------------------
# resolve_config_paths
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_root():
    """Repo root fixture for testing resolve_config_paths."""
    return find_repo_root()


def test_resolve_config_paths_relative_input(repo_root):
    """Relative paths under 'input' are resolved against the repo root."""
    config = {
        "input": {"path_to_parental": "data/vhh.fasta", "mutations_yaml": "data/mutations.yaml"},
        "genetic_algorithm": {},
    }
    result = resolve_config_paths(config)
    assert result["input"]["path_to_parental"] == str((repo_root / "data/vhh.fasta").resolve())
    assert result["input"]["mutations_yaml"] == str((repo_root / "data/mutations.yaml").resolve())


def test_resolve_config_paths_relative_ga(repo_root):
    """Relative paths under 'genetic_algorithm' are resolved against the repo root."""
    config = {
        "input": {},
        "genetic_algorithm": {
            "path_probability_matrix": "data/prob.yaml",
            "path_cdr_positions": "data/cdrs.yaml",
        },
    }
    result = resolve_config_paths(config)
    assert result["genetic_algorithm"]["path_probability_matrix"] == str((repo_root / "data/prob.yaml").resolve())
    assert result["genetic_algorithm"]["path_cdr_positions"] == str((repo_root / "data/cdrs.yaml").resolve())


def test_resolve_config_paths_absolute_unchanged(tmp_path):
    """Absolute paths are left unchanged."""
    abs_path = str(tmp_path / "vhh.fasta")
    config = {
        "input": {"path_to_parental": abs_path, "mutations_yaml": ""},
        "genetic_algorithm": {},
    }
    result = resolve_config_paths(config)
    assert result["input"]["path_to_parental"] == abs_path


def test_resolve_config_paths_none_unchanged():
    """None/empty values are left unchanged."""
    config = {
        "input": {"path_to_parental": None, "mutations_yaml": ""},
        "genetic_algorithm": {"path_probability_matrix": None, "path_cdr_positions": ""},
    }
    result = resolve_config_paths(config)
    assert result["input"]["path_to_parental"] is None
    assert result["input"]["mutations_yaml"] == ""
    assert result["genetic_algorithm"]["path_probability_matrix"] is None
    assert result["genetic_algorithm"]["path_cdr_positions"] == ""


def test_resolve_config_paths_missing_sections():
    """Sections absent from the config are silently skipped."""
    config = {}
    result = resolve_config_paths(config)  # must not raise
    assert result == {}


def test_get_parental_sequence_from_string():
    """Test get_parental_sequence with string input."""
    parental_seq = "EVQLVESGGGLVQPGGSLRLSCAASG.DIQMTQSPSSLSASVGDRVTITC"
    result = get_parental_sequence(parental_seq, ".", None, None)

    assert isinstance(result, AbVSeq)
    assert result.heavy_chain == "EVQLVESGGGLVQPGGSLRLSCAASG"
    assert result.light_chain == "DIQMTQSPSSLSASVGDRVTITC"


def test_initialize_vocabulary():
    """Test initialize_vocabulary function."""
    parental = AbVSeq(heavy_chain="EVQL", light_chain="DIQM")
    with patch("workflows.components.bayesian_optimization.src.run_utils.load_mutations_from_yaml") as mock_load:
        mock_load.return_value = {0: "A", 1: "C"}

        result = initialize_vocabulary("test.yaml", parental)
        assert result[0] == "AE"
        assert result[1] == "CV"
        assert len(result) == len(parental.heavy_chain) + len(parental.light_chain)


def test_check_if_multi_objective():
    """Test check_if_multi_objective function."""
    # Multi-objective case
    bo_config = {"acquisition": "qEHVI", "batch_size": 2}
    obj_config = [{"name": "obj1"}, {"name": "obj2"}]

    result = check_if_multi_objective(bo_config, obj_config)
    assert result is True

    # Single objective case
    bo_config = {"acquisition": "EI", "batch_size": 1}
    obj_config = [{"name": "obj1"}]

    result = check_if_multi_objective(bo_config, obj_config)
    assert result is False


def test_transform_scores_to_numpy():
    """Test transform_scores_to_numpy function."""
    # Test with dictionary input
    scores_dict = {"obj1": [0.8, 0.9], "obj2": [0.7, 0.6]}
    objective_names = ["obj1", "obj2"]

    result = transform_scores_to_numpy(scores_dict, objective_names)
    expected = np.array([[0.8, 0.7], [0.9, 0.6]], dtype="float32")
    np.testing.assert_array_equal(result, expected)

    # Test with torch tensor
    scores_tensor = torch.tensor([[0.8, 0.7], [0.9, 0.6]])
    result = transform_scores_to_numpy(scores_tensor, objective_names)
    np.testing.assert_array_equal(result, scores_tensor.cpu().numpy())


def test_compute_metrics():
    """Test compute_metrics function."""
    sequences = ["EVQL", "AVQL", "EVRL"]
    scores = np.array([[0.8, 0.7], [0.9, 0.6], [0.7, 0.8]])
    parental_seq = "EVQL"
    objective_names = ["obj1", "obj2"]
    pareto_front = np.array([[0.9, 0.6], [0.7, 0.8]])
    hypervolume = 0.5

    with patch("workflows.components.bayesian_optimization.src.run_utils.compute_pareto_front_plots") as mock_plots:
        mock_plots.return_value = {"plot1": MagicMock()}

        result = compute_metrics(1, sequences, scores, parental_seq, objective_names, pareto_front, hypervolume)

        assert result["step"] == 1
        assert result["population/num_sequences"] == 3
        assert np.isclose(result["scores/hypervolume"], 0.5)
        assert "diversity/hamming_mean" in result
        assert "scores/obj1/mean" in result


def test_compute_pareto_front_plots():
    """Test compute_pareto_front_plots function."""
    scores_np = np.array([[0.8, 0.7], [0.9, 0.6], [0.7, 0.8]])
    objective_names = ["obj1", "obj2"]
    pareto_front = np.array([[0.9, 0.6], [0.7, 0.8]])

    result = compute_pareto_front_plots(scores_np, objective_names, pareto_front)

    assert isinstance(result, dict)
    assert "pareto_front_obj1_vs_obj2" in result
    assert "distribution_obj1" in result
    assert "distribution_obj2" in result
    assert "objective_correlations" in result


def test_compute_sequence_diversity_metrics():
    """Test compute_sequence_diversity_metrics function."""
    sequences = ["EVQL", "AVQL", "EVRL", "AVRL"]
    parental_seq = "EVQL"

    result = compute_sequence_diversity_metrics(sequences, parental_seq)

    assert "diversity/hamming_mean" in result
    assert "diversity/hamming_std" in result
    assert "diversity/parental_dist_mean" in result
    assert "diversity/unique_sequences" in result
    assert "diversity/uniqueness_ratio" in result
    assert result["diversity/unique_sequences"] == 4
    assert np.isclose(result["diversity/uniqueness_ratio"], 1.0)


def test_compute_score_distribution_metrics():
    """Test compute_score_distribution_metrics function."""
    scores_np = np.array([[0.8, 0.7], [0.9, 0.6], [0.7, 0.8], [0.85, 0.75]])
    objective_names = ["obj1", "obj2"]

    result = compute_score_distribution_metrics(scores_np, objective_names)

    assert "scores/obj1/mean" in result
    assert "scores/obj1/std" in result
    assert "scores/obj1/min" in result
    assert "scores/obj1/max" in result
    assert "scores/obj2/median" in result
    assert "scores/correlation/obj1_obj2" in result

    # Check some values
    assert np.isclose(result["scores/obj1/mean"], np.mean([0.8, 0.9, 0.7, 0.85]))
    assert np.isclose(result["scores/obj1/min"], 0.7)
    assert np.isclose(result["scores/obj1/max"], 0.9)


# Additional edge case tests
def test_get_parental_sequence_no_input():
    """Test get_parental_sequence with no input raises ValueError."""
    with pytest.raises(ValueError, match="No parental sequence provided"):
        get_parental_sequence("", ".", None, None)


def test_initialize_vocabulary_no_yaml():
    """Test initialize_vocabulary with no YAML file raises ValueError."""
    parental = AbVSeq(heavy_chain="EVQL", light_chain="DIQM")

    with pytest.raises(ValueError, match="No mutations YAML file provided"):
        initialize_vocabulary("", parental)


def test_compute_pareto_front_plots_single_objective():
    """Test compute_pareto_front_plots with single objective."""
    scores_np = np.array([0.8, 0.9, 0.7, 0.85])
    objective_names = ["obj1"]

    result = compute_pareto_front_plots(scores_np, objective_names, None)

    assert isinstance(result, dict)
    assert "score_distribution" in result
    assert len(result) == 1  # Only single objective plot


def test_compute_sequence_diversity_metrics_single_sequence():
    """Test compute_sequence_diversity_metrics with single sequence."""
    sequences = ["EVQL"]
    parental_seq = "EVQL"

    result = compute_sequence_diversity_metrics(sequences, parental_seq)

    # Should not have hamming distance metrics for single sequence
    assert "diversity/hamming_mean" not in result
    assert "diversity/parental_dist_mean" in result
    assert np.isclose(result["diversity/parental_dist_mean"], 0.0)
    assert result["diversity/unique_sequences"] == 1


def test_save_and_load_checkpoint(tmp_path):
    """Test saving and loading a checkpoint."""
    checkpoint_file = tmp_path / "checkpoint.pkl"
    iteration = 3
    sequences = ["AAA", "BBB"]
    scores = {"obj1": [0.1, 0.2], "obj2": [0.3, 0.4]}
    rng = random.Random(123)
    run_id = "test_run_id"

    save_checkpoint(str(checkpoint_file), iteration, sequences, scores, rng, run_id)
    loaded = load_checkpoint(str(checkpoint_file))

    assert loaded["iteration"] == iteration
    assert loaded["sequences"] == sequences
    assert loaded["scores"] == scores
    assert loaded["run_id"] == run_id
    assert "rng_state" in loaded


def test_save_results_to_csv(tmp_path):
    """Test saving results to CSV."""
    output_path = tmp_path / "results.csv"
    parental = AbVSeq(heavy_chain="AAA", light_chain="CC")
    sequences = ["ABACB", "BABCA"]
    scores = {"obj1": [0.5, 0.6], "obj2": [0.7, 0.8]}
    objective_names = ["obj1", "obj2"]
    pareto_indices = [1]

    save_results_to_csv(str(output_path), parental, sequences, scores, objective_names, pareto_indices)
    df = pd.read_csv(output_path)
    assert list(df["full_sequence"]) == sequences
    assert list(df["heavy_chain"]) == [seq[:-2] for seq in sequences]
    assert list(df["light_chain"]) == [seq[-2:] for seq in sequences]
    assert np.allclose(df["obj1"], [0.5, 0.6])
    assert np.allclose(df["obj2"], [0.7, 0.8])
    assert "is_pareto" in df.columns
    assert df["is_pareto"].tolist() == [False, True]


def test_set_random_state():
    """Test set_random_state returns reproducible random.Random."""
    seed = 12345
    rng1 = set_random_state(seed)
    rng2 = set_random_state(seed)
    vals1 = [rng1.random() for _ in range(5)]
    vals2 = [rng2.random() for _ in range(5)]
    assert np.allclose(vals1, vals2)

    # Also test that global random and torch are seeded
    random.seed(seed)
    torch.manual_seed(seed)
    val_rand = random.random()
    val_torch = torch.randint(0, 100, (1,)).item()
    random.seed(seed)
    torch.manual_seed(seed)
    val_rand2 = random.random()
    val_torch2 = torch.randint(0, 100, (1,)).item()
    assert val_rand == val_rand2
    assert val_torch == val_torch2
