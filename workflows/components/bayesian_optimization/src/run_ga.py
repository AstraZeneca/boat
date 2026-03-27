"""Run a simple genetic algorithm on selected scoring functions."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch.serialization
import typer

from boat.bayesopt.loop.utils import compute_hypervolume, compute_pareto_front
from boat.data_utils import load_cdr_positions, load_probability_matrix, load_yaml_config
from boat.genetic_algorithm.genetic_optimizers import GeneticAlgorithm
from boat.genetic_algorithm.utils import generate_random_point_mutations
from boat.scoring_function.interface import ScoringFunctionInterface
from workflows.components.bayesian_optimization.src.run_utils import (
    compute_metrics,
    generate_objective_function,
    get_parental_sequence,
    get_wandb_context,
    initialize_vocabulary,
    resolve_config_paths,
    set_basic_logging_config,
    set_random_state,
)

torch.serialization.add_safe_globals([argparse.Namespace])


class GASumScoringFunction(ScoringFunctionInterface):
    """A simple scoring function that sums the scores of multiple objectives."""

    def __init__(self, scoring_functions, weights_per_scoring_fct: dict | None = None):
        """
        Initialize the sum scoring function.

        Args:
            scoring_functions (list): A list of scoring function callables.
        """
        super().__init__("SumScoringFunction")
        self.scoring_functions = scoring_functions
        self.weights_per_scoring_fct = weights_per_scoring_fct

    def __call__(self, sequences, *args, **kwargs):
        """
        Evaluate a list of sequences by summing the scores from all scoring functions.

        Args:
            sequences (list): A list of sequences to score.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            score: The summed score corresponding to the sequences.
        """
        total_scores = None
        for func in self.scoring_functions:
            weight = 1.0
            if self.weights_per_scoring_fct is not None:
                weight = float(self.weights_per_scoring_fct.get(func.objective_names[0], 1.0))

            scores = func(sequences, *args, **kwargs)[func.objective_names[0]]
            scores = [weight * score for score in scores]
            if total_scores is None:
                total_scores = scores
            else:
                total_scores = [sum(x) for x in zip(total_scores, scores)]
        return total_scores


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

    obj_config = config["objective_functions"]
    ga_config = config["genetic_algorithm"]

    # Probability matrix
    probability_matrix = load_probability_matrix(config["genetic_algorithm"].pop("path_probability_matrix", None))
    ga_config.update({"probability_matrix": probability_matrix})

    # CDR positions
    cdr_positions = load_cdr_positions(config["genetic_algorithm"].pop("path_cdr_positions", None))
    ga_config.update({"cdr_positions": cdr_positions})

    # Set a random number generator from given seed
    random_seed = config.get("random_seed", 42)
    rng = set_random_state(random_seed)

    # Check if multi-objective acquisition function is used
    # is_multi_objective = check_if_multi_objective(bo_config, obj_config)

    # Setup logger
    logger = set_basic_logging_config(
        name="ga",
        path=".",
        level=config["logging"].get("logging_level", "INFO"),
        verbose=True,
    )

    log_to_wandb = config["logging"].get("log_to_wandb", True)
    with get_wandb_context(
        log_to_wandb=log_to_wandb,
        entity=config["logging"].get("wandb_entity", None),
        project=config["logging"].get("wandb_project", None),
        name=f"GA_{config['experiment']}_{random_seed}",
        config=config,
        run_id=config["experiment"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    ) as run:
        logger.info(f"Using random seed: {random_seed}")

        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Logging output to {output_dir}")

        # Load parental sequence and join as a string
        parental_abvseq = get_parental_sequence(
            parental_seq=parental_seq, linker=linker, path_to_parental=config["input"].get("path_to_parental", "")
        )
        parental_seq = parental_abvseq.heavy_chain + parental_abvseq.light_chain

        # Initialize vocabulary
        vocab = initialize_vocabulary(
            mutations_yaml=config["input"].get("mutations_yaml", ""), parental_abvseq=parental_abvseq
        )

        # Update genetic optimization parameters
        ga_config.update({"aa_vocabulary": vocab})
        ga_config.update({"parental_sequence": parental_seq})

        # Build scoring fct
        scoring_fct_obj = generate_objective_function(
            obj_config=obj_config,
            parental_abvseq=parental_abvseq,
        )
        scoring_fct = GASumScoringFunction(scoring_functions=[sf for sf in scoring_fct_obj.scoring_functions])

        # Generate and score initial population
        logger.info("Generating and scoring initial sequences.")
        initial_sequences = generate_random_point_mutations(
            sequence=parental_seq,
            aa_vocabulary=vocab,
            population_size=ga_config.pop("initial_population_size"),
            max_point_mutations=ga_config.pop("initial_max_mutations"),
            rng=rng,
        )
        initial_scores = scoring_fct(initial_sequences)

        introduced_mutations = [
            "-".join([str(i) for i in range(len(parental_seq)) if sequence[i] != parental_seq[i]])
            for sequence in initial_sequences
        ]
        logger.debug(f"Initial mutations: {introduced_mutations}")
        logger.debug(f"Initial scores: {initial_scores}")

        max_rounds = ga_config.pop("max_rounds", 10)

        genetic_algorithm = GeneticAlgorithm(
            initial_population=initial_sequences,
            scoring_function=scoring_fct,
            rng=rng,
            **ga_config,
            logging_level=config["logging"].get("logging_level", "INFO"),
        )

        genetic_algorithm.run(max_rounds=max_rounds)

        # Now score the generations iteratively with the original multi-objective scoring functions

        seq_up_to_gen = initial_sequences.copy()

        mo_scores = scoring_fct_obj(seq_up_to_gen)
        mo_scores = [
            [mo_scores[obj_name][i] for obj_name in scoring_fct_obj.objective_names] for i in range(len(seq_up_to_gen))
        ]
        scores_up_to_gen = torch.tensor(mo_scores, dtype=torch.float32)

        # reference point for hypervolume calculation
        min_vals = torch.min(scores_up_to_gen, dim=0)[0]
        offset = torch.abs(min_vals) * 0.1  # 10% offset
        ref_point = min_vals - offset

        for i, gen in enumerate(genetic_algorithm.generations):
            seq_up_to_gen.extend(gen.sequences)

            mo_scores = scoring_fct_obj(gen.sequences)
            mo_scores = [
                [mo_scores[obj_name][i] for obj_name in scoring_fct_obj.objective_names]
                for i in range(len(gen.sequences))
            ]
            mo_scores = torch.tensor(mo_scores, dtype=torch.float32)
            scores_up_to_gen = torch.cat((scores_up_to_gen, mo_scores), dim=0)

            # compute pareto front
            pareto_front, _ = compute_pareto_front(scores_up_to_gen)

            # compute hypervolume
            hypervolume = compute_hypervolume(pareto_front, ref_point)

            # Log final metrics
            metrics = compute_metrics(
                step=i,
                sequences=initial_sequences,
                scores=initial_scores,
                parental_seq=parental_seq,
                objective_names=scoring_fct_obj.objective_names,
                pareto_front=pareto_front,
                hypervolume=hypervolume,
            )
            run.log(metrics)


if __name__ == "__main__":
    typer.run(main)
