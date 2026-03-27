"""Bayesian optimization loop for multi objective."""

import logging
import random

import torch
import wandb
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from scipy import stats

from boat.bayesopt.acquisition.acquisition import AcquisitionFunctionOnSequences
from boat.bayesopt.acquisition.utils import get_acquisition
from boat.bayesopt.encodings.encodings import get_encoding
from boat.bayesopt.loop.utils import compute_pareto_front
from boat.bayesopt.models.utils import fit_gp, get_model, initialize_mll, initialize_model
from boat.bayesopt.utils import _plot_scatter_plot
from boat.genetic_algorithm.genetic_operators import mutate_sequence
from boat.genetic_algorithm.genetic_optimizers import (
    BatchGeneticAlgorithm,
    GeneticAlgorithm,
)
from boat.scoring_function.interface import MultiObjectiveScoringFunction


class MOBayesOptOnSequences:
    """Class for a multi objective Bayesian optimization loop on sequences."""

    def __init__(
        self,
        initial_sequences: list[str],
        initial_scores: dict[str, list[float]],
        objective_function: MultiObjectiveScoringFunction,
        encoding_str: str,
        model_dict: dict,
        acquisition_str: str,
        vocab: dict[str | int, str],
        ga_params: dict,
        rng: random.Random = random.Random(42),
        ref_point: list[float] = None,
        batch_size: int = 1,
        n_qmc_samples: int = 1024,
        max_training_points: int = 700,
        validate_surrogates: bool = True,
        validation_interval: int = 1,
        dd_dict: dict
        | None = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
        **kwargs,
    ):
        """Initialize the loop.

        Args:
            initial_sequences: initial sequences to start the loop with
            initial_scores: initial scores from the multi obj function
            model_dict: dict that maps task to specific model to use
            acquisition_str: string identifier of the acquisition function to use
            encoding_str: string identifier of the encoding to use
            objective_function: objective function to evaluate new sequences
            vocab: vocabulary to use for the sequences
            ga_params: parameters for the genetic algorithm
            rng: random number generator, defaults to random.Random(42)
            ref_point: reference point for hypervolume calculation, defaults to None
            batch_size: number of samples drawn in every iteration (requires a 'q' acquitision function)
            n_qmc_samples: number of samples to estimate the batch acquisition function
            max_training_points: maximum number of training points to use for the surrogate models
            validate_surrogates: whether to validate surrogates during training, defaults to True
            validation_interval: validate surrogates every N iterations, defaults to 1
            dd_dict: dictionary with device and dtype information, defaults to using CUDA if available
            **kwargs: additional keyword arguments for the encoding
        """
        # Set random number generator
        self.rng = rng

        # Get info about objectives
        self.sequences = initial_sequences
        self.output_dimension = objective_function.objective_dimension

        # Convert dictionary of scores to tensor with shape [n_sequences, n_objectives]
        score_lists = [initial_scores[name] for name in objective_function.objective_names]
        scores_tensor = torch.tensor(
            list(zip(*score_lists)),  # Transpose to get [n_sequences, n_objectives]
            dtype=dd_dict["dtype"],
            device=dd_dict["device"],
        )
        self.scores = scores_tensor

        # target scoring function
        self.objective_function = objective_function
        self.vocab = vocab

        self.encoding = get_encoding(encoding_str)(vocab=vocab, dd_dict=dd_dict, **kwargs)
        self.encoded_sequences = self.encoding(self.sequences).to(dd_dict["device"])
        self.max_training_points = max_training_points

        # Store model selections
        self._model_dict = model_dict
        self._acq_str = acquisition_str

        self.ga_params = ga_params

        self.validate_surrogates = validate_surrogates
        self.validation_interval = validation_interval

        self.batch_size = batch_size

        self.n_qmc_samples = n_qmc_samples

        if self.batch_size > 1 and not acquisition_str.lower().startswith("q"):
            raise ValueError(
                f"Batch size {self.batch_size} > 1 requires a batch acquisition function (starting with 'q'). "
                f"Got '{acquisition_str}'. Either use batch_size=1 or a q-acquisition function."
            )

        # Set up reference point for hypervolume
        self.ref_point = None
        if (ref_point is None) and (self._acq_str in ["EHVI", "qEHVI", "qNEHVI"]):
            # Set default reference point as min values minus 10%.
            min_vals = torch.min(self.scores, dim=0)[0]
            offset = torch.abs(min_vals) * 0.1  # 10% offset
            self.ref_point = min_vals - offset
        elif self._acq_str in ["EHVI", "qEHVI", "qNEHVI"]:
            self.ref_point = torch.tensor(ref_point, dtype=dd_dict["dtype"], device=dd_dict["device"])

        # Tracking the loop state
        self.iteration = 0

        self.pareto_front, self.pareto_indices = None, None
        if self._acq_str in ["EHVI", "qEHVI", "qNEHVI"]:
            # For multi-objective, track the Pareto front instead of a single best value
            self.pareto_front, self.pareto_indices = compute_pareto_front(self.scores)
            self.best_observed = self.pareto_front
        else:
            self.best_observed = torch.max(self.scores)

        # Set W&B
        self.wandb_run = None
        if wandb.run is not None:
            self.wandb_run = wandb.run

    def __repr__(self):
        """Return a string representation of the loop."""
        return (
            f"MOBayesOptOnSequences(model_dict={self._model_dict}, "
            f"acquisition_str={self._acq_str}, "
            f"encoding_str={self.encoding.__class__.__name__}, "
            f"iteration={self.iteration}, "
            f"num_objectives={self.output_dimension}, "
            f"pareto_front={self.pareto_front}), "
            f"best_observed={self.best_observed})"
        )

    def _set_acq_kwargs(self, model):
        acq_kwargs = {"model": model}

        if self._acq_str in ["EHVI", "qEHVI"]:
            # For EHVI, create partitioning
            # Create partitioning for EHVI
            partitioning = FastNondominatedPartitioning(
                ref_point=self.ref_point,
                Y=self.pareto_front,  # Uses current pareto front
            )
            acq_kwargs["partitioning"] = partitioning
            acq_kwargs["ref_point"] = self.ref_point.tolist()
        elif self._acq_str == "qNEHVI":
            # Use current points as baseline
            acq_kwargs["X_baseline"] = self.encoded_sequences
            acq_kwargs["prune_baseline"] = True  # Prune baseline points for efficiency - check this?
            acq_kwargs["ref_point"] = self.ref_point.tolist()
        elif self._acq_str == "qNEI":
            # Use current points as baseline
            acq_kwargs["X_baseline"] = self.encoded_sequences
            acq_kwargs["best_f"] = self.best_observed
        else:
            acq_kwargs["best_f"] = self.best_observed

        if self._acq_str.startswith("q"):
            # batch acquisition functions require solving an integral with MC, therefore we must fix the sampler
            seed = self.rng.randint(0, 2**32 - 1)
            acq_kwargs["sampler"] = SobolQMCNormalSampler(self.n_qmc_samples, seed=seed)

        return acq_kwargs

    def loop_step(self):
        """Take a single step in the loop."""
        self.iteration += 1

        model = self._update_models(self.encoded_sequences, self.scores, max_training_points=self.max_training_points)

        acq_kwargs = self._set_acq_kwargs(model)
        acq = get_acquisition(self._acq_str)(**acq_kwargs)
        acq_seq = AcquisitionFunctionOnSequences(acquisition_function=acq, encoding=self.encoding, name=self._acq_str)

        # Optimize acquisition function with genetic algorithm and score it
        new_seqs = self._optimize_acq_with_ga(acq_seq)

        if self._acq_str.startswith("q"):
            # For batch acquisition: best_results is a list containing a single batch
            batch = new_seqs[0]

            scores_dict = self.objective_function(batch)
            all_scores = [
                [scores_dict[obj_name][i] for obj_name in self.objective_function.objective_names]
                for i in range(len(batch))
            ]

            # Convert to tensor with shape [batch_size, number of scores]
            new_scores = torch.tensor(all_scores, dtype=self.scores.dtype, device=self.scores.device)

            # Update sequences and scores
            self.sequences.extend(batch)
            self.encoded_sequences = torch.cat(
                (self.encoded_sequences, self.encoding(batch).to(self.encoded_sequences.device)), dim=0
            )
            self.scores = torch.cat((self.scores, new_scores), dim=0)
        else:
            new_seqs = new_seqs[0]  # For single acquisition, just get the sequence
            scores_dict = self.objective_function([new_seqs])

            # Convert dictionary to list in consistent order
            score_list = [scores_dict[obj_name][0] for obj_name in self.objective_function.objective_names]

            # Convert to tensor
            new_score = torch.tensor(score_list, dtype=self.scores.dtype, device=self.scores.device).unsqueeze(
                0
            )  # Shape: [1, output_dimension]

            # Update sequences and scores
            self.sequences.append(new_seqs)
            self.encoded_sequences = torch.cat(
                (self.encoded_sequences, self.encoding([new_seqs]).to(self.encoded_sequences.device)), dim=0
            )
            self.scores = torch.cat((self.scores, new_score), dim=0)

        # Update Pareto front
        if self._acq_str in ["EHVI", "qEHVI", "qNEHVI"]:
            self.pareto_front, self.pareto_indices = compute_pareto_front(self.scores)
            self.best_observed = self.pareto_front
        else:
            self.best_observed = torch.max(self.scores)

        return self.sequences, self.scores

    def run(self, n_iterations: int):
        """Run the full loop.

        Args:
            n_iterations: number of iterations

        Returns
        -------
            torch.Tensor of best observed function values at the iteration given by the index
        """
        while self.iteration < n_iterations:
            self.loop_step()

        return self.sequences, self.scores

    def _validate_surrogates(self, train_x, obj_y, is_binary, model_str, obj_name):
        """Validate the surrogate model for a given objective using a train/validation split."""
        n_points = train_x.size(0)

        indices = list(range(n_points))
        self.rng.shuffle(indices)
        split_idx = int(0.8 * n_points)

        assert split_idx >= 2, "At least 2 data points are required for train / validation split."

        train_idx = torch.tensor(indices[:split_idx], device=train_x.device, dtype=torch.long)
        val_idx = torch.tensor(indices[split_idx:], device=train_x.device, dtype=torch.long)

        val_train_x, val_train_y = train_x[train_idx], obj_y[train_idx]
        val_x, val_y = train_x[val_idx], obj_y[val_idx]

        val_model = initialize_model(get_model(model_str), val_train_x, val_train_y, binary=is_binary)
        val_mll = initialize_mll(val_model, num_data=val_train_y.size(0), binary=is_binary)
        fit_gp(val_mll)
        val_model.eval()

        with torch.inference_mode():
            posterior = val_model.posterior(val_x)
            preds = posterior.mean.squeeze(-1)

            preds, val_y = preds.cpu(), val_y.cpu().squeeze(-1)

            if is_binary:
                probs = torch.sigmoid(preds)
                acc = (probs.round() == val_y).float().mean().item()
                logging.info(
                    "Validation accuracy for %s: %.4f (n=%d)",
                    obj_name,
                    acc,
                    val_y.size(0),
                )
                if self.wandb_run is not None:
                    self.wandb_run.log({f"validation_surrogate/accuracy/{obj_name}": acc}, step=self.iteration)
            else:
                mse = torch.nn.functional.mse_loss(preds, val_y).item()
                spearman = stats.spearmanr(preds.numpy(), val_y.numpy()).correlation

                logging.debug(
                    "Validation MSE for %s: %.4f (n=%d)",
                    obj_name,
                    mse,
                    val_y.size(0),
                )
                logging.debug(
                    f"First 10 scores of {obj_name}: {preds.numpy()[:10]} vs {val_y.numpy()[:10]}",
                )
                if self.wandb_run is not None:
                    self.wandb_run.log({f"validation_surrogate/mse/{obj_name}": mse}, step=self.iteration)
                    self.wandb_run.log({f"validation_surrogate/spearman/{obj_name}": spearman}, step=self.iteration)

                    scatter_plot = _plot_scatter_plot(preds.numpy(), val_y.numpy(), name=obj_name)
                    self.wandb_run.log({f"validation_surrogate/scatter/{obj_name}": scatter_plot}, step=self.iteration)

    def _update_models(self, train_x, train_y, max_training_points: int = 700) -> ModelListGP:
        """Update the models with current data.

        Args:
            train_x: Training inputs
            train_y: Training outputs of shape (n_points, n_objectives)

        Returns
        -------
            List of models, one for each objective.
        """
        models = []

        n_points = train_x.size(0)

        indices = list(range(n_points))
        self.rng.shuffle(indices)

        if n_points > max_training_points:
            selected_indices = torch.tensor(indices[:max_training_points], device=train_x.device, dtype=torch.long)
            train_x = train_x[selected_indices]
            train_y = train_y[selected_indices]

        # Determine if validation should run this iteration
        should_validate = (
            self.validate_surrogates
            and self.validation_interval > 0
            and (self.iteration % self.validation_interval == 0)
        )

        # Train a separate model for each objective
        for obj_idx, obj_name in enumerate(self.objective_function.objective_names):
            is_binary = self.objective_function.objective_types[obj_name] == "binary"
            assert (
                is_binary is not None
            ), f"Objective binary classification could not be determined from name: {obj_name}"

            logging.info(
                f"Training model for objective {self.objective_function.objective_names[obj_idx]} (binary={is_binary})"
            )

            model_str = self._model_dict["binary"] if is_binary else self._model_dict["regression"]

            # Extract the scores for this objective
            obj_y = train_y[:, obj_idx : obj_idx + 1]

            # Validate surrogates only if enabled and at appropriate interval
            if should_validate:
                self._validate_surrogates(
                    train_x, obj_y, is_binary, model_str, self.objective_function.objective_names[obj_idx]
                )

            # Initialize and train model for this objective
            model = initialize_model(get_model(model_str), train_x, obj_y, binary=is_binary)

            mll = initialize_mll(model, num_data=obj_y.size(0), binary=is_binary)
            fit_gp(mll)
            model.eval()
            models.append(model)

        model = ModelListGP(*models)
        return model

    def _optimize_acq_with_ga(self, acq_seq: AcquisitionFunctionOnSequences) -> list:
        try:
            max_rounds = self.ga_params.pop("max_rounds")
        except KeyError:
            pass

        # In _optimize_acq_with_ga, use the seeded RNG:
        initial_population = [
            mutate_sequence(
                seq,
                mutation_rate=self.ga_params["mutation_rate"],
                aa_vocabulary=self.ga_params["aa_vocabulary"],
                rng=self.rng,
            )
            for seq in self.sequences
        ]

        if self._acq_str.startswith("q"):
            ga = BatchGeneticAlgorithm(
                initial_population=initial_population,
                batch_size=self.batch_size,
                scoring_function=acq_seq,
                rng=self.rng,
                **self.ga_params,
            )
        else:
            ga = GeneticAlgorithm(
                initial_population=initial_population,
                scoring_function=acq_seq,
                rng=self.rng,
                **self.ga_params,
            )

        with torch.no_grad():
            try:
                ga.run(max_rounds=max_rounds)
                self.ga_params["max_rounds"] = max_rounds
            except NameError:
                ga.run()

        best_seqs, _ = ga.get_top_n(1)
        return best_seqs
