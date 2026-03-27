"""Utility functions for running experiments with Bayesian Optimization."""

import logging
import os
import pickle
import random
from contextlib import contextmanager
from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import sleep
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import torch
import wandb
from logomaker import Logo

from boat.biologics.sequence import AbVSeq, str_to_abvseq
from boat.data_utils import load_mutations_from_yaml, read_in_fasta
from boat.genetic_algorithm.utils import generate_random_point_mutations
from boat.genetic_algorithm.vocabularies import (
    positional_vocabulary,
)
from boat.scoring_function.interface import MultiObjectiveScoringFunction
from boat.scoring_function.oasis_interface import OASisScoringFunction
from boat.scoring_function.plm_interface import ESMInterface


def find_repo_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until a marker file is found; that directory is the repo root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find repository root (no '{marker}' found in any parent directory).")


def resolve_config_paths(config: dict) -> dict:
    """Resolve relative paths in a config dict against the repository root.

    Paths that are already absolute are left unchanged.
    Any falsy value (None, empty string) is also left unchanged.
    """
    repo_root = find_repo_root()
    for section_key, path_keys in [
        ("input", ["path_to_parental", "mutations_yaml"]),
        ("genetic_algorithm", ["path_probability_matrix", "path_cdr_positions"]),
    ]:
        section = config.get(section_key, {})
        for key in path_keys:
            val = section.get(key)
            if val and not Path(val).is_absolute():
                section[key] = str((repo_root / val).resolve())
    return config


def set_basic_logging_config(
    name: str,
    path: str | None = "./",
    level: str = "DEBUG",
    format: str = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    verbose: bool = False,
    level_file: str = None,
    level_stdout: str = None,
) -> logging.Logger:
    """Set up the default python logger.

    Parameters
    ----------
    name : str
        Name of the log file.
    path : str
        Path to a storage location for the log file. Use None if you don't wish to write log to disk.
    format : str, optional
        Log print format, by default "%(asctime)s %(levelname)-8s %(message)s".
    datefmt : str, optional
        Log datetime fomar, by default "%Y-%m-%d %H:%M:%S".
    level : str, optional
        Log level. See https://docs.python.org/3/library/logging.html, for more details, by default "DEBUG".
    verbose : bool, optional
        If True, will also print all logs to stdout, be default False.
    level_file : str, optional
        Log level for the file log. If not given, uses general level.
    level_stdout : str, optional
        Log level for the stdout log. If not given, uses general level.

    Returns
    -------
    logger: logging.Logger
        a logger with the required handles and format.
    """
    if not path and not verbose:
        raise ValueError(f"At least one of `path` or `verbose` must be set. Got path={path} and verbose={verbose}.")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(name)
    handlers = []

    if path:
        log_file = os.path.join(path, name + ".log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.upper() if not level_file else level_file.upper())
        handlers.append(file_handler)

    if verbose:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(level.upper() if not level_stdout else level_stdout.upper())
        handlers.append(stdout_handler)

    logging.basicConfig(
        level=level.upper(),
        format=format,
        datefmt=datefmt,
        handlers=handlers,
    )

    return logger


class NoOpWandbRun:
    """A no-op mock object for wandb run when logging is disabled."""

    def log(self, data):
        """Do nothing."""
        pass

    def log_artifact(self, artifact):
        """Do nothing and return the artifact for compatibility."""
        return artifact

    def use_artifact(self, artifact, type=None):
        """Return the input artifact unchanged for compatibility."""
        return artifact

    def __enter__(self):
        """Return self for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


@contextmanager
def get_wandb_context(log_to_wandb, entity, project, name, config, run_id):
    """
    Get a wandb context manager, or a no-op mock if logging is disabled.

    Args:
        log_to_wandb: Whether to log to wandb
        entity: Wandb entity
        project: Wandb project
        name: Wandb run name
        config: Config dict to pass to wandb

    Yields
    ------
        Either a real wandb run or a no-op mock object
    """
    if not log_to_wandb:
        yield NoOpWandbRun()
        return

    try:
        with wandb.init(
            entity=entity, project=project, name=name, config=config, resume="allow", id=str(run_id)
        ) as run:
            yield run
    except wandb.errors.CommError as err:
        logging.warning(
            "W&B initialization failed (%s). Falling back to no-op logging. "
            "Check logging.wandb_entity/logging.wandb_project and your W&B account access.",
            err,
        )
        yield NoOpWandbRun()


def _wandb_upload_artifact(
    artifact_name: str,
    artifact_path: str,
    artifact_type: str,
    run: Optional[wandb.run] = None,
    logger: Optional[logging.Logger] = None,
    wait=True,
) -> wandb.sdk.artifacts.artifact.Artifact:
    """Upload artifact to wandb.

    Parameters
    ----------
    artifact_name : str
        The name for the artifact.
    artifact_path : str
        Path or filename of the data to upload
    artifact_type : str
        The type of artifact: data, config, model
    run : wandb.run
        The run to log the artifact to. Defaults to None if this is called from within a wandb context manager
    logger : logging.Logger
        The logger to use (can be None to ignore logging) for INFO
    wait : bool, defaults to True
        Whether to wait for the artifact to be fully uploaded.

    Returns
    -------
    wandb.sdk.artifacts.artifact.Artifact : the logged artifact
    """
    logger = logging if logger is None else logger
    run = wandb.run if run is None else run

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    if os.path.isdir(artifact_path):
        artifact.add_dir(local_path=artifact_path)
    else:
        artifact.add_file(local_path=artifact_path)

    logged_artifact = run.log_artifact(artifact)
    if wait and not isinstance(run, NoOpWandbRun):
        artifact.wait()

    logger.info(f"Uploaded artifacts of type {artifact_type} in {artifact_path} to {artifact_name}.")
    return logged_artifact


def wandb_upload_artifact(
    artifact_name: str, artifact_path: str, artifact_type: str, run: wandb.run, logger: logging.Logger, retry: int = 5
):
    """Retry option to W&B upload."""
    retry = max(1, retry)
    retryable_errors = (
        requests.exceptions.ProxyError,
        requests.exceptions.ConnectionError,
        ConnectionResetError,
    )
    while retry:
        try:
            _wandb_upload_artifact(
                artifact_name=artifact_name,
                artifact_path=artifact_path,
                artifact_type=artifact_type,
                run=run,
                logger=logger,
            )
            return
        except wandb.errors.CommError as e:
            err = e.exc
            if isinstance(err, retryable_errors):
                retry -= 1
                logger.error(f"Network/proxy error uploading artifact ({err}). Retries left: {retry}")
                sleep(5)
                continue
            logger.error(f"Failed to push data. err={err}")
            raise
        except Exception as e:
            if isinstance(e, retryable_errors):
                retry -= 1
                logger.error(f"Network/proxy error uploading artifact ({e}). Retries left: {retry}")
                sleep(5)
                continue
            logger.error(f"Failed to push data. err={e}")
            raise


def get_parental_sequence(
    parental_seq: str, linker: str, path_to_parental: Optional[str] = None, chains: Optional[list[str]] = None
) -> AbVSeq:
    """Process parental sequence either from a string or a CSV file.

    Args:
        parental_seq (str): The parental sequence as a string. If the string is empty,
            it will attempt to read from a CSV or a fasta file.
        linker (str): The linker between heavy and light chain. If there is no linker, it is
            assumed that the given chain is a heavy chain. To give a light chain, pass the string of
            amino acids with a preceding "."
        path_to_parental (str): Path to file with parental sequence.
        chains (list): Chain names. Assumes heavy chain is listed first.

    Returns
    -------
        AbVSeq: An instance of AbVSeq containing the heavy and light chains.

    Raises
    ------
        ValueError: If no parental sequence is provided and no path to parental file is found.
    """
    if parental_seq:
        parental_chains = parental_seq.split(linker) if linker in parental_seq else [parental_seq, ""]

    elif path_to_parental:
        if path_to_parental.endswith(".csv"):
            # Read parental sequence from CSV file.
            if not chains:
                raise ValueError("No chains provided for CSV input.")
            try:
                parent_df = pd.read_csv(path_to_parental, usecols=[chain for chain in chains])
            except ValueError:
                raise ValueError(f"CSV file must contain the specified chains: {chains}")

            parental_chains = sorted(parent_df.iloc[0].to_list(), key=len, reverse=True)  # Heavy chain first
            if len(parental_chains) == 1 and "l" in chains[0].lower():
                # Assume that there is an "l" in the chain name for light chain
                parental_chains = ["", parental_chains[0]]  # Only light chain provided
            else:
                parental_chains.append("")  # Ensure light chain is present
        else:
            return read_in_fasta(path_to_parental)

    else:
        logging.error("No parental sequence provided.")
        raise ValueError(
            "No parental sequence provided. "
            "Please provide a parental sequence or a path to a csv or fasta file containing the parental sequence."
        )

    return AbVSeq(heavy_chain=parental_chains[0], light_chain=parental_chains[1])


def initialize_vocabulary(mutations_yaml: str, parental_abvseq):
    """Initialize vocabulary.

    Args:
        mutations_yaml (str): Path to the YAML file defining mutations.
        parental_abvseq (AbVSeq): An instance of AbVSeq containing the heavy and light chains.

    Returns
    -------
        dict: A dictionary representing the positional vocabulary.
    """
    if not mutations_yaml:
        raise ValueError("No mutations YAML file provided for vocabulary initialization.")

    # Get mutations
    mutations = load_mutations_from_yaml(mutations_yaml)

    # Build vocabulary
    vocab = {}
    parental_seq = parental_abvseq.heavy_chain + parental_abvseq.light_chain
    vocab.update(positional_vocabulary(parental_seq, mutations))

    return vocab


def load_model_wb(
    artifact_path: str, destination: str, entity: str, run_name: str, project: str, logger: logging.Logger
) -> None:
    """Load model and config artifacts from wandb."""
    model_artifact = artifact_path
    config_artifact = artifact_path.split(":")[0] + "_config:latest"  # assumes :v0, :v1 tag
    retry = 10
    with wandb.init(entity=entity, name=run_name, project=project) as run:
        while retry:
            err = None
            try:
                model_artifact = run.use_artifact(model_artifact, type="model")
                model_artifact.download(
                    root=destination,
                )

                config_artifact = run.use_artifact(config_artifact, type="config")
                config_artifact.download(
                    root=destination,
                )
                return
            except wandb.errors.CommError as e:
                err = e.exc
            except Exception as e:
                err = e
            retry -= 1

            logger.error(f"Failed to get data from {artifact_path}. Remaining retries: {retry}. err={err}")
            if not retry:
                raise (err)
            sleep(5)


def check_if_multi_objective(bo_config, obj_config):
    """Check if run is multi-objective."""
    is_multi_objective = True if bo_config["acquisition"] in ["EHVI", "qEHVI", "qNEHVI"] else False

    # is any of the scoring functions multi-output?
    any_multioutput_function = any(
        [(len(cfg["metrics_for_scoring"]) > 1) if "metrics_for_scoring" in cfg else False for cfg in obj_config]
    )

    assert ((is_multi_objective and len(obj_config) > 1) or (is_multi_objective and any_multioutput_function)) or (
        not is_multi_objective and len(obj_config) == 1
    ), f"Using {bo_config['acquisition']} as acquisition function which is \
        incompatible with {len(obj_config)} objectives."
    assert (bo_config["acquisition"].startswith("q") and bo_config["batch_size"] > 1) or (
        not bo_config["acquisition"].startswith("q") and bo_config["batch_size"] == 1
    ), "Using a batched acquisition function with a batch size of 1."

    return is_multi_objective


#### Scoring & Objective function ####


def generate_objective_function(obj_config, parental_abvseq):
    """Populate scoring function and build final objective function."""
    scoring_fcts = []
    for scoring_fct_cfg in obj_config:
        logging.info(f"Adding scoring function {scoring_fct_cfg['description']}.")

        if scoring_fct_cfg["name"] == "plm":
            scoring_fct = ESMInterface(
                parental=parental_abvseq,
                checkpoint_path=scoring_fct_cfg["checkpoint_path"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        elif scoring_fct_cfg["name"] == "oasis":
            scoring_fct = OASisScoringFunction(
                parental=parental_abvseq, threshold=scoring_fct_cfg.get("threshold", None)
            )
        else:
            raise ValueError(f"Scoring function of name {scoring_fct_cfg['name']} not known.")
        scoring_fcts.append(scoring_fct)

    mo_scoring_function = MultiObjectiveScoringFunction(scoring_functions=scoring_fcts)
    return mo_scoring_function


#### Metric utils ####


def transform_scores_to_numpy(scores, objective_names):
    """Transform the scores to a numpy object."""
    if isinstance(scores, dict):
        score_lists = [scores[name] for name in objective_names]
        scores = np.array(
            list(zip(*score_lists)),  # Transpose to get [n_sequences, n_objectives]
            dtype="float32",
        )
    elif isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    elif isinstance(scores, list):
        scores = np.asarray(scores, dtype="float32").reshape(-1, len(objective_names))
    return scores


def compute_metrics(step, sequences, scores, parental_seq, objective_names, pareto_front, hypervolume):
    """Compute metrics for the current step."""
    # Close all open matplotlib figures to prevent memory leaks from figures created in previous iterations,
    # especially those generated by sequence_logo_plot() below.
    plt.close()

    scores = transform_scores_to_numpy(scores, objective_names=objective_names)

    metrics = {
        "step": step,
        "population/num_sequences": len(sequences),
    }
    metrics.update(compute_sequence_diversity_metrics(sequences, parental_seq))
    metrics.update(compute_score_distribution_metrics(scores, objective_names))

    if (hypervolume is not None) and (pareto_front is not None):
        metrics["scores/pareto_front_size"] = len(pareto_front)
        metrics["scores/pareto_front_ratio"] = len(pareto_front) / len(scores)
        metrics["scores/hypervolume"] = hypervolume

    # Add Pareto front plots
    pareto_plots = compute_pareto_front_plots(scores, objective_names, pareto_front)
    metrics.update(pareto_plots)

    # Sequence logo plot
    logo_fig = sequence_logo_plot(sequences)
    if logo_fig is not None:
        metrics["sequence_logo"] = wandb.Image(logo_fig)  # wandb will log this as an image

    return metrics


def compute_pareto_front_plots(scores_np: np.array, objective_names, pareto_front=None):
    """Create plotly plots of Pareto front slices for multi-objective optimization."""
    plots = {}

    if scores_np.ndim == 1 or scores_np.shape[1] < 2:
        plots["score_distribution"] = _plot_single_objective_histogram(scores_np, objective_names)
        return plots

    n_objectives = scores_np.shape[1]

    # Pairwise 2D projections
    for i, j in combinations(range(n_objectives), 2):
        plots[f"pareto_front_{objective_names[i]}_vs_{objective_names[j]}"] = _plot_pareto_2d(
            scores_np, objective_names, pareto_front, i, j
        )

    # 3D plot for exactly 3 objectives
    if n_objectives == 3:
        plots["pareto_front_3d"] = _plot_pareto_3d(scores_np, objective_names, pareto_front)

    # Objective correlation heatmap
    if n_objectives >= 2:
        plots["objective_correlations"] = _plot_objective_correlation(scores_np, objective_names)

    # Individual histograms
    for i, obj_name in enumerate(objective_names):
        plots[f"distribution_{obj_name}"] = _plot_objective_histogram(scores_np, pareto_front, i, obj_name)

    return plots


def _plot_single_objective_histogram(scores_np, objective_names):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores_np.flatten(), nbinsx=20, opacity=0.7, name="Score Distribution"))
    fig.update_layout(
        title=f'Distribution of {objective_names[0] if len(objective_names) > 0 else "Scores"}',
        xaxis_title=objective_names[0] if len(objective_names) > 0 else "Score",
        yaxis_title="Frequency",
        showlegend=False,
    )
    return fig


def _plot_pareto_2d(scores_np, objective_names, pareto_front, i, j):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores_np[:, i],
            y=scores_np[:, j],
            mode="markers",
            marker=dict(size=6, color="lightblue", opacity=0.6, line=dict(width=1, color="gray")),
            name="All points",
        )
    )
    if pareto_front is not None and len(pareto_front) > 0:
        pareto_np = pareto_front.cpu().numpy() if hasattr(pareto_front, "cpu") else pareto_front
        fig.add_trace(
            go.Scatter(
                x=pareto_np[:, i],
                y=pareto_np[:, j],
                mode="markers",
                marker=dict(size=10, color="red", opacity=0.9, line=dict(width=2, color="darkred")),
                name="Pareto front",
            )
        )
        if len(pareto_np) > 1:
            sorted_indices = np.argsort(pareto_np[:, i])
            sorted_pareto = pareto_np[sorted_indices]
            fig.add_trace(
                go.Scatter(
                    x=sorted_pareto[:, i],
                    y=sorted_pareto[:, j],
                    mode="lines",
                    line=dict(color="red", width=1, dash="dash"),
                    opacity=0.7,
                    name="Pareto trend",
                    showlegend=False,
                )
            )
    fig.update_layout(
        title=f"Pareto Front: {objective_names[i]} vs {objective_names[j]}",
        xaxis_title=objective_names[i],
        yaxis_title=objective_names[j],
        showlegend=True,
        width=800,
        height=600,
    )
    return fig


def _plot_pareto_3d(scores_np, objective_names, pareto_front):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=scores_np[:, 0],
            y=scores_np[:, 1],
            z=scores_np[:, 2],
            mode="markers",
            marker=dict(size=4, color="lightblue", opacity=0.6),
            name="All points",
        )
    )
    if pareto_front is not None and len(pareto_front) > 0:
        pareto_np = pareto_front.cpu().numpy() if hasattr(pareto_front, "cpu") else pareto_front
        fig.add_trace(
            go.Scatter3d(
                x=pareto_np[:, 0],
                y=pareto_np[:, 1],
                z=pareto_np[:, 2],
                mode="markers",
                marker=dict(size=8, color="red", opacity=0.9),
                name="Pareto front",
            )
        )
    fig.update_layout(
        title="3D Pareto Front Visualization",
        scene=dict(
            xaxis_title=f"{objective_names[0]}",
            yaxis_title=f"{objective_names[1]}",
            zaxis_title=f"{objective_names[2]}",
        ),
        width=900,
        height=700,
    )
    return fig


def _plot_objective_correlation(scores_np, objective_names):
    correlation_matrix = np.corrcoef(scores_np.T)
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=objective_names,
            y=objective_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=[
                [f"{correlation_matrix[i, j]:.2f}" for j in range(len(objective_names))]
                for i in range(len(objective_names))
            ],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Objective Correlation Matrix",
        xaxis_title="Objectives",
        yaxis_title="Objectives",
        width=600,
        height=500,
    )
    return fig


def _plot_objective_histogram(scores_np, pareto_front, i, obj_name):
    fig = go.Figure()
    obj_scores = scores_np[:, i] if scores_np.ndim > 1 else scores_np
    fig.add_trace(
        go.Histogram(
            x=obj_scores,
            nbinsx=20,
            opacity=0.7,
            marker_color="skyblue",
            marker_line_color="black",
            marker_line_width=1,
            name=obj_name,
        )
    )
    if pareto_front is not None and len(pareto_front) > 0:
        pareto_np = pareto_front.cpu().numpy() if hasattr(pareto_front, "cpu") else pareto_front
        pareto_scores = pareto_np[:, i] if pareto_np.ndim > 1 else pareto_np
        pareto_mean = np.mean(pareto_scores)
        fig.add_vline(
            x=pareto_mean,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="Pareto mean",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"{obj_name} Distribution",
        xaxis_title=obj_name,
        yaxis_title="Frequency",
        showlegend=False,
        width=500,
        height=400,
    )
    return fig


def compute_sequence_diversity_metrics(
    sequences: list[str], parental_seq: str, max_seq_for_diversity: int = 500, seed: Optional[int] = None
):
    """Compute sequence diversity metrics."""
    metrics = {}

    # Prevent blow up on computational time on hamming distance
    if len(sequences) > max_seq_for_diversity:
        if seed is not None:
            rng = random.Random(seed)
            sequences = rng.choices(sequences, k=max_seq_for_diversity)
        else:
            sequences = random.choices(sequences, k=max_seq_for_diversity)

    # Hamming distance metrics
    if len(sequences) > 1:
        hamming_distances = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(sequences[i], sequences[j]))
                hamming_distances.append(hamming_dist)

        metrics["diversity/hamming_mean"] = np.mean(hamming_distances)
        metrics["diversity/hamming_std"] = np.std(hamming_distances)
        metrics["diversity/hamming_max"] = np.max(hamming_distances)
        metrics["diversity/hamming_min"] = np.min(hamming_distances)

    # Distance from parental sequence
    parental_distances = [sum(c1 != c2 for c1, c2 in zip(seq, parental_seq)) for seq in sequences]
    metrics["diversity/parental_dist_mean"] = np.mean(parental_distances)
    metrics["diversity/parental_dist_std"] = np.std(parental_distances)
    metrics["diversity/parental_dist_max"] = np.max(parental_distances)

    # Number of unique sequences
    metrics["diversity/unique_sequences"] = len(set(sequences))
    metrics["diversity/uniqueness_ratio"] = len(set(sequences)) / len(sequences)

    return metrics


def sequence_logo_plot(sequences: list[str], max_sequences: int = 300):
    """
    Create a sequence logo (matplotlib figure) from a list of equal-length sequences.

    Returns a matplotlib Figure or None if it cannot be created.
    """
    if not sequences:
        return None

    # Ensure all sequences same length
    seq_len = len(sequences[0])
    if any(len(s) != seq_len for s in sequences):
        return None  # Skip if inconsistent

    # Downsample for speed if large
    if len(sequences) > max_sequences:
        sequences = random.sample(sequences, k=max_sequences)

    seq_df = pd.DataFrame([list(s) for s in sequences])
    counts_df = seq_df.apply(pd.Series.value_counts).fillna(0)

    # Probabilities per position (positions x amino acids)
    prob_df = counts_df.T.div(counts_df.T.sum(axis=1), axis=0).fillna(0)

    # Entropy
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log2(prob_df.replace(0, np.nan))
    entropy = -np.nansum(prob_df.values * logp.values, axis=1)

    # Normalize entropy for coloring
    if entropy.max() - entropy.min() == 0:
        entropy_norm = np.zeros_like(entropy)
    else:
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min())

    cmap = plt.get_cmap("gist_heat_r")
    colors = [mpl.colors.to_hex(cmap(e)) for e in entropy_norm]

    fig = plt.figure(figsize=(min(20, max(6, seq_len / 3)), 2.5))
    logo = Logo(prob_df, ax=plt.gca())
    ax = logo.ax

    for i, color in enumerate(colors):
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.6, zorder=0)

    ax.set_facecolor("white")
    logo.style_spines(visible=False)
    logo.style_xticks(rotation=90, fmt="%d", anchor=0)
    logo.style_glyphs(color="k", alpha=1.0, linewidth=0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_title("Sequence Logo (entropy-colored)")
    fig.tight_layout()
    return fig


def compute_score_distribution_metrics(scores_np, objective_names):
    """Compute score distribution metrics."""
    metrics = {}

    for i, obj_name in enumerate(objective_names):
        obj_scores = scores_np[:, i] if scores_np.ndim > 1 else scores_np

        metrics[f"scores/{obj_name}/mean"] = np.mean(obj_scores)
        metrics[f"scores/{obj_name}/std"] = np.std(obj_scores)
        metrics[f"scores/{obj_name}/min"] = np.min(obj_scores)
        metrics[f"scores/{obj_name}/max"] = np.max(obj_scores)
        metrics[f"scores/{obj_name}/median"] = np.median(obj_scores)
        metrics[f"scores/{obj_name}/q25"] = np.percentile(obj_scores, 25)
        metrics[f"scores/{obj_name}/q75"] = np.percentile(obj_scores, 75)

        # Range and interquartile range
        metrics[f"scores/{obj_name}/range"] = np.max(obj_scores) - np.min(obj_scores)
        metrics[f"scores/{obj_name}/iqr"] = np.percentile(obj_scores, 75) - np.percentile(obj_scores, 25)

    # Multi-objective specific metrics
    if scores_np.ndim > 1 and scores_np.shape[1] > 1:
        # Correlation between objectives
        for i in range(scores_np.shape[1]):
            for j in range(i + 1, scores_np.shape[1]):
                corr = np.corrcoef(scores_np[:, i], scores_np[:, j])[0, 1]
                metrics[f"scores/correlation/{objective_names[i]}_{objective_names[j]}"] = corr

    return metrics


def save_results_to_csv(
    output_path: str,
    parental_abvseq: AbVSeq,
    sequences: list[str],
    scores: dict | torch.Tensor,
    objective_names: list[str],
    pareto_indices: torch.Tensor | None = None,
):
    """Save sequences and scores to a CSV file."""
    if os.path.exists(output_path):
        os.remove(output_path)
    scores = transform_scores_to_numpy(scores, objective_names=objective_names)
    scores_dict = {objective_names[i]: scores[:, i] for i in range(len(objective_names))}

    if parental_abvseq.heavy_chain and parental_abvseq.light_chain:
        abvseqs = str_to_abvseq(sequences, parental_abvseq)
        seq_dict = {}
        seq_dict["full_sequence"] = sequences
        seq_dict["heavy_chain"] = [abv.heavy_chain for abv in abvseqs]
        seq_dict["light_chain"] = [abv.light_chain for abv in abvseqs]
        results_df = pd.DataFrame({**seq_dict, **scores_dict})

    else:
        results_df = pd.DataFrame({"sequence": sequences, **scores_dict})

    if pareto_indices is not None:
        if isinstance(pareto_indices, torch.Tensor):
            pareto_indices = pareto_indices.cpu().numpy()
        results_df["is_pareto"] = False
        results_df.loc[pareto_indices, "is_pareto"] = True

    results_df.to_csv(output_path, index=False)


def save_checkpoint(
    checkpoint_file: str, iteration: int, sequences: list, scores: dict, rng: random.Random, run_id: str
):
    """Save a checkpoint of the current BO state to a file and logs it as a wandb Artifact.

    Args:
        checkpoint_file (str): Path to the checkpoint file.
        iteration (int): Current iteration number.
        sequences (list): List of sequences evaluated so far.
        scores (dict): Dictionary of scores for each objective.
        rng (random.Random): Random number generator state.
        run_id (str): Unique identifier for the run.
    """
    processed_scores = {}
    for k, v in scores.items():
        if hasattr(v, "cpu"):  # for torch.Tensor objects
            processed_scores[k] = v.cpu().numpy().tolist()
        else:
            processed_scores[k] = v

    # Extract the RNG state from the random.Random object as a dictionary.
    rng_state = {"state": rng.getstate()}

    state = {
        "iteration": iteration,
        "sequences": sequences,
        "scores": processed_scores,
        "rng_state": rng_state,
        "run_id": run_id,
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(checkpoint_file):
    """Load a checkpoint of the BO state from a file.

    Args:
        checkpoint_file (str): Path to the checkpoint file.

    Returns
    -------
        dict: The loaded checkpoint state.
    """
    with open(checkpoint_file, "rb") as f:
        return pickle.load(f)


def get_run_state(
    config,
    checkpoint_file,
    restore_from_checkpoint,
    vocab,
    parental_seq,
    objective_function,
    probability_matrix,
    logger,
):
    """
    Get the current state from a checkpoint if available, otherwise generate from scratch.

    Args:
        config: Configuration dictionary.
        checkpoint_file: Path to the checkpoint file.
        restore_from_checkpoint: Whether to restore from checkpoint.
        vocab: Vocabulary for sequence generation.
        parental_seq: The parental sequence.
        objective_function: Objective function object.
        logger: Logger for logging information.

    Returns: start_iter, initial_sequences, initial_scores, rng, run_id
    """
    random_seed = config.get("random_seed", 42)
    rng = set_random_state(random_seed)

    if restore_from_checkpoint and os.path.exists(checkpoint_file):
        logger.info("Checkpoint found. Loading state to resume run.")
        ckpt = load_checkpoint(checkpoint_file)
        start_iter = ckpt["iteration"]
        initial_sequences = ckpt["sequences"]
        initial_scores = ckpt["scores"]
        rng.setstate(ckpt["rng_state"]["state"])
        run_id = ckpt["run_id"]

        # If finished, return None to signal early exit
        if start_iter >= config["bayes_opt"]["n_iterations"]:
            logger.info(
                f"Experiment already finished {start_iter} iterations. To start a new run, please change "
                f"experiment name or delete checkpoint file {checkpoint_file}."
            )
            return None
        # Remove not needed args from ga config
        config["genetic_algorithm"].pop("initial_population_size", None)
        config["genetic_algorithm"].pop("initial_max_mutations", None)
    else:
        run_id = config["experiment"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        start_iter = 0
        initial_sequences = generate_random_point_mutations(
            sequence=parental_seq,
            aa_vocabulary=vocab,
            population_size=config["genetic_algorithm"].pop("initial_population_size"),
            max_point_mutations=config["genetic_algorithm"].pop("initial_max_mutations"),
            rng=rng,
            probability_matrix=probability_matrix,
        )
        assert len(initial_sequences) == len(set(initial_sequences)), "Initial sequences are not unique!"
        logger.info(f"Number unique sequences in initial population: {len(initial_sequences)}")
        initial_scores = objective_function(initial_sequences)

    return start_iter, initial_sequences, initial_scores, rng, run_id


### Reproducibility
def set_random_state(seed: int) -> random.Random:
    """
    Set the random state for reproducibility.

    Args:
        seed: random seed for initializing the random number generator

    Returns
    -------
        random.Random: The random number generator.
    """
    if not isinstance(seed, int):
        raise TypeError(f"The seed needs to be of type int, got {seed.type}")

    rng = random.Random(seed)

    # Set the global seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng
