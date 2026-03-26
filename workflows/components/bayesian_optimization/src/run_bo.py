"""Run Bayesian optimization on Boltz2 model."""

import argparse
import gc
import os
import shutil
import warnings
from pathlib import Path

import torch
import typer
from botorch.models.utils.assorted import InputDataWarning

from boat.bayesopt.loop.utils import compute_hypervolume
from boat.bayesopt.mo_loop import MOBayesOptOnSequences
from boat.data_utils import load_cdr_positions, load_probability_matrix, load_yaml_config
from workflows.components.bayesian_optimization.src.run_utils import (
    check_if_multi_objective,
    compute_metrics,
    generate_objective_function,
    get_parental_sequence,
    get_run_state,
    get_wandb_context,
    initialize_vocabulary,
    resolve_config_paths,
    save_checkpoint,
    save_results_to_csv,
    set_basic_logging_config,
    wandb_upload_artifact,
)

warnings.filterwarnings("ignore", category=InputDataWarning)
torch.serialization.add_safe_globals([argparse.Namespace])


def clear_torch_cache():
    """Clear PyTorch cache and run garbage collection."""
    gc.collect()  # Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear PyTorch's memory cache
        torch.cuda.synchronize()  # Wait for GPU operations to complete


def main(
    parental_seq: str = typer.Option(
        "",
        help="Parental sequence to optimize. Separate heavy and light chain by a linker",
    ),
    linker: str = typer.Option(
        ".", help="Linker to help separate sequences. If there is no linker, heavy chain is assumed."
    ),
    config_path: str = typer.Option(
        None,
        help="Location of the experiment config, if not provided, default config will be used.",
    ),
):
    """Run experiment."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "default_config.yaml")
    # Configs
    config = load_yaml_config(config_path)
    config = resolve_config_paths(config)

    ga_config = config["genetic_algorithm"]
    bo_config = config["bayes_opt"]
    obj_config = config["objective_functions"]

    # Set a random number generator from given seed
    random_seed = config.get("random_seed", 42)

    # Check if multi-objective acquisition function is used
    is_multi_objective = check_if_multi_objective(bo_config, obj_config)

    # Setup logger
    logger = set_basic_logging_config(
        name="boat",
        path=".",
        level=config["logging"].get("logging_level", "INFO"),
        verbose=True,
    )

    # Setup output logging and copy config file to output directory
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Logging output to {output_dir}")
    output_file = os.path.join(output_dir, f"out_{config['experiment']}.csv")
    config_dest = os.path.join(output_dir, os.path.basename(config_path))
    shutil.copy2(config_path, config_dest)
    logger.info(f"Copied config file to {config_dest}")

    # Define checkpoint file location in output_dir.
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{config['experiment']}_{random_seed}.pkl")
    restore_from_checkpoint = config.get("restore_from_checkpoint", False)

    # Load parental sequence and join as a string, initialize vocabulary
    parental_abvseq = get_parental_sequence(
        parental_seq=parental_seq,
        linker=linker,
        path_to_parental=config["input"].get("path_to_parental", ""),
        chains=config["input"].get("chains", None),
    )
    parental_seq = parental_abvseq.heavy_chain + parental_abvseq.light_chain
    vocab = initialize_vocabulary(
        mutations_yaml=config["input"].get("mutations_yaml", ""), parental_abvseq=parental_abvseq
    )
    ga_config.update({"aa_vocabulary": vocab})
    ga_config.update({"parental_sequence": parental_seq})

    # Probability matrix
    probability_matrix = load_probability_matrix(config["genetic_algorithm"].pop("path_probability_matrix", None))
    ga_config.update({"probability_matrix": probability_matrix})

    # CDR positions
    cdr_positions = load_cdr_positions(config["genetic_algorithm"].pop("path_cdr_positions", None))
    ga_config.update({"cdr_positions": cdr_positions})

    # Build objective fct
    objective_function = generate_objective_function(
        obj_config=obj_config,
        parental_abvseq=parental_abvseq,
    )

    # Get initial state from checkpoint or generate initial population
    start_iter, initial_sequences, initial_scores, rng, run_id = get_run_state(
        config,
        checkpoint_file,
        restore_from_checkpoint,
        vocab,
        parental_seq,
        objective_function,
        probability_matrix,
        logger,
    )

    log_to_wandb = config["logging"].get("log_to_wandb", True)
    with get_wandb_context(
        log_to_wandb=log_to_wandb,
        entity=config["logging"].get("wandb_entity", None),
        project=config["logging"].get("wandb_project", None),
        name=f"BO_{config['experiment']}_{random_seed}",
        config=config,
        run_id=run_id,
    ) as run:
        logger.info(f"Starting run from iteration {start_iter}/{bo_config['n_iterations']}")

        # Save initial state as checkpoint.
        save_checkpoint(
            checkpoint_file=checkpoint_file,
            iteration=start_iter,
            sequences=initial_sequences,
            scores=initial_scores,
            rng=rng,
            run_id=run_id,
        )

        wandb_upload_artifact(
            artifact_name=f"bo_checkpoint_{run_id}",
            artifact_path=checkpoint_file,
            artifact_type="checkpoint",
            run=run,
            logger=logger,
        )

        # Initialize MOBayesOptOnSequences
        bayes_opt_loop = MOBayesOptOnSequences(
            initial_sequences=initial_sequences,
            initial_scores=initial_scores,
            objective_function=objective_function,
            encoding_str=bo_config["encoding"],
            model_dict=bo_config["model"],
            acquisition_str=bo_config["acquisition"],
            batch_size=bo_config["batch_size"],
            vocab=vocab,
            ga_params=ga_config,
            rng=rng,
            n_qmc_samples=bo_config.get("n_qmc_samples", 1024),
            max_training_points=bo_config.get("max_training_points", 700),
        )
        objective_names = objective_function.objective_names

        # Log initial metrics
        if start_iter == 0:
            metrics = compute_metrics(
                step=0,
                sequences=initial_sequences,
                scores=initial_scores,
                parental_seq=parental_seq,
                objective_names=objective_names,
                pareto_front=bayes_opt_loop.pareto_front if is_multi_objective else None,
                hypervolume=compute_hypervolume(bayes_opt_loop.pareto_front, bayes_opt_loop.ref_point)
                if is_multi_objective
                else None,
            )
            run.log(metrics)

        # Write results of initial population
        save_results_to_csv(
            output_path=output_file,
            parental_abvseq=parental_abvseq,
            sequences=initial_sequences,
            scores=initial_scores,
            objective_names=objective_names,
            pareto_indices=bayes_opt_loop.pareto_indices if is_multi_objective else None,
        )

        # Running optimization
        for i in range(start_iter + 1, bo_config["n_iterations"] + 1):
            logger.info(f"Starting iteration {i}/{bo_config['n_iterations']}")

            # Take one step
            sequences, scores = bayes_opt_loop.loop_step()

            # Save checkpoint
            scores_dict = {k: scores[:, i] for i, k in enumerate(objective_function.objective_names)}
            save_checkpoint(
                checkpoint_file=checkpoint_file,
                iteration=i,
                sequences=bayes_opt_loop.sequences,
                scores=scores_dict,
                rng=rng,
                run_id=run_id,
            )

            wandb_upload_artifact(
                artifact_name=f"bo_checkpoint_{run_id}",
                artifact_path=checkpoint_file,
                artifact_type="checkpoint",
                run=run,
                logger=logger,
            )

            # Log metrics and write sequences to file
            metrics = compute_metrics(
                step=i,
                sequences=sequences,
                scores=scores,
                parental_seq=parental_seq,
                objective_names=objective_names,
                pareto_front=bayes_opt_loop.pareto_front if is_multi_objective else None,
                hypervolume=compute_hypervolume(bayes_opt_loop.pareto_front, bayes_opt_loop.ref_point)
                if is_multi_objective
                else None,
            )
            run.log(metrics)

            # Write results to file
            save_results_to_csv(
                output_path=output_file,
                parental_abvseq=parental_abvseq,
                sequences=sequences,
                scores=scores,
                objective_names=objective_names,
                pareto_indices=bayes_opt_loop.pareto_indices if is_multi_objective else None,
            )

            # Clear torch and garbage collection cache
            clear_torch_cache()

        # Remove checkpoint at the end of the run.
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"Removed checkpoint file {checkpoint_file} after successful completion of the run.")


if __name__ == "__main__":
    typer.run(main)
