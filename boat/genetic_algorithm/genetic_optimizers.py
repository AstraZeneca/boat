"""Common operations to run genetic algorithms."""

import abc
import logging
import random
from collections import Counter
from typing import Callable, Dict, List, Union

from tqdm import tqdm

from boat.biologics.liabilities import filter_by_liability

from .constraints import MutationConstraintHandler, create_constraint_handler
from .genetic_operators import batch_crossover, mutate_batch, mutate_sequence, single_point_crossover
from .utils import Generation, rank_sequences

GA_TITLE = "Genetic Algorithm Generations"
LOG_SCORING_NEW_POPULATION = "Scoring the new population..."
LOG_GENERATING_NEW_POPULATION = "Generating the new population..."


class BaseGeneticAlgorithm(abc.ABC):
    """Abstract base class for genetic algorithm."""

    def __init__(
        self,
        initial_population: List[str],
        scoring_function: Union[Callable[[List[str]], List[float]], Callable[[List[List[str]]], List[float]]],
        aa_vocabulary: Dict[int, str],
        mutation_rate: float,
        crossover_rate: float,
        probability_matrix: Dict[int, Dict[str, float]] | None,
        population_size: int,
        repetitions: int,
        tournament_size: int,
        rng: random.Random,
        liability_filtering: bool,
        liability_threshold: float,
        constraint_handler: MutationConstraintHandler,
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the BaseGeneticAlgorithm with parameters for mutation, crossover, population size, and more.

        Args:
            initial_population : List[str]
                The initial population of sequences to start the genetic algorithm.
            scoring_function : Union[Callable[[List[str]], List[float]], Callable[[List[List[str]]], List[float]]],
                Function to evaluate the fitness of multiple sequences. It takes a list or list of lists of sequences
                as input and returns a corresponding list of scores as output.
            aa_vocabulary : Dict[int, str]
                Alphabet of possible amino acids for mutation. Either position-based or per-AA-based vocabulary.
            mutation_rate : float, optional
                The probability of mutation for each offspring sequence (default is 0.1).
            crossover_rate : float, optional
                The probability of performing crossover between two parents (default is 0.7).
            probability_matrix: Dict[int, Dict[str, float]], optional
                Probability matrix from a generative model that guides the single point mutation operation.
            population_size : int, optional
                The desired size of the resulting new population (default is 100).
            repetitions : int, optional
                The maximum number of times a sequence can be added to the new population
                before it is considered fully evaluated (default is 10).
            tournament_size : int
                The number of individuals to participate in the tournament selection for parent selection.
            rng : random.Random, optional
                Random number generator for reproducibility. Defaults to random.Random(42).
            liability_filtering : bool, optional
                Whether to filter sequences by liability score (default is True).
                If True, sequences with high liability scores will be filtered out.
            liability_threshold : float, optional
                The threshold for the average liability score per amino acid (default is 1.0).
                Sequences with scores higher than this threshold will be filtered out.
            constraint_handler : MutationConstraintHandler
                Handler to apply mutation constraints during the genetic algorithm.
            logging_level : int, optional
                Logging level for the genetic algorithm (default is logging.INFO).
        """
        # check that a random number generator has been given in the correct form
        if not isinstance(rng, random.Random):
            raise TypeError(f"rng must be an int or random.Random instance, got {type(rng)}")
        self.rng = rng

        # Scoring function
        self.scoring_function = scoring_function
        self.aa_vocabulary = aa_vocabulary

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.repetitions = repetitions
        self.tournament_size = tournament_size

        # Probability prior from generative model
        self.probability_matrix = probability_matrix

        # initialize the iteration counter to keep track of the number of generations
        self.iteration = 0

        # constraint handler
        self.constraint_handler = constraint_handler

        # liability filtering
        self.liability_filtering = liability_filtering
        if self.liability_filtering:
            self.liability_threshold = liability_threshold

            # filter initial population by liability
            initial_population = filter_by_liability(initial_population, aa_threshold=self.liability_threshold)

        # Make sure all sequences in the initial population meet the constraints
        initial_population = [self.constraint_handler.repair_sequence(seq) for seq in initial_population]

        if not initial_population:
            raise ValueError("Initial population is empty after applying constraints and liability filtering.")

        # Track the generations
        initial_generation = Generation(
            sequences=initial_population,
            scores=scoring_function(initial_population),
            number=self.iteration,
        )
        self.generations = [initial_generation]

        self.sequences_counter = Counter(initial_population)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    @property
    def sequences(self) -> List[str]:
        """Get the sequences of all generation."""
        return [seq for gen in self.generations for seq in gen.sequences]

    @property
    def scores(self) -> List[float]:
        """Get the scores of all generation."""
        return [score for gen in self.generations for score in gen.scores]

    @abc.abstractmethod
    def run(self) -> Dict[str, List]:
        """Run the genetic algorithm.

        Parameters
        ----------
        initial_population : List[str]
            The initial population of sequences to start the genetic algorithm.
        evaluate_sequences : Callable[[List[str]], List[float]]
            Function to evaluate the fitness of multiple sequences. It takes a list of sequences
            as input and returns a corresponding list of scores as output.

        Returns
        -------
        Dict[str, List]
            A dictionary containing:
                - "final_sequences": The final population of sequences.
                - "final_scores": The scores of the final population.
                - "all_sequences": All evaluated sequences over the generations.
                - "all_scores": The scores of all evaluated sequences.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_new_population(
        self,
        sequences: List[str],
        scores: List[float],
    ) -> List[str]:
        """Generate a new population using crossover and mutation.

        Parameters
        ----------
        sequences : List[str]
            A list of string sequences from which new sequences will be generated.
        scores : List[float]
            A list of scores corresponding to each sequence, used for selection.

        Returns
        -------
        List[str]
            A list of new sequences generated for the population.

        Notes
        -----
        The function performs tournament selection, single-point crossover, and mutation
        to generate a new population of sequences. The function ensures that no sequence
        is added to the population more than a specified number of times.
        """
        new_population_count = {}
        new_population = []

        while len(new_population) < self.population_size:
            # 1. Select parents
            parent1 = self.tournament_selection(sequences, scores, self.tournament_size)
            parent2 = self.tournament_selection(sequences, scores, self.tournament_size)
            # 2. Cross-over
            if self.rng.random() < self.crossover_rate:
                offspring1, offspring2 = single_point_crossover(parent1, parent2, rng=self.rng)
            else:
                offspring1, offspring2 = parent1, parent2

            # 3. Adjust vocabulary according to constraints and mutate
            adjusted_vocab = self.constraint_handler.adjust_vocabulary(
                offspring1,
                self.aa_vocabulary,
            )
            offspring1 = mutate_sequence(
                offspring1,
                self.mutation_rate,
                adjusted_vocab,
                rng=self.rng,
                probability_matrix=self.probability_matrix,
            )

            adjusted_vocab = self.constraint_handler.adjust_vocabulary(
                offspring2,
                self.aa_vocabulary,
            )
            offspring2 = mutate_sequence(
                offspring2,
                self.mutation_rate,
                adjusted_vocab,
                rng=self.rng,
                probability_matrix=self.probability_matrix,
            )

            # 4. Add offspring if it's not already been evaluated "repetitions" times
            for offspring in (offspring1, offspring2):
                total_count = self.sequences_counter.get(offspring, 0) + new_population_count.get(offspring, 0)
                if self.repetitions > total_count:
                    new_population.append(offspring)
                    new_population_count[offspring] = new_population_count.get(offspring, 0) + 1

        return new_population[: self.population_size]

    def tournament_selection(
        self,
        sequences: List[str],
        scores: List[float],
        tournament_size: int,
    ) -> str:
        """Select a sequence using tournament selection.

        Parameters
        ----------
        sequences : list of str
            The sequences to select from.
        scores : list of float
            The scores associated with each sequence.
        tournament_size: int
            The number of individuals to participate in the tournament.

        Returns
        -------
        str
            The winning sequence from the tournament.
        """
        selected = self.rng.sample(list(zip(sequences, scores)), tournament_size)
        return max(selected, key=lambda x: x[1])[0]

    def get_top_n(self, n: int) -> tuple[list[str], list[float]]:
        """
        Return the top n sequences and their corresponding scores from all generations.

        Parameters
        ----------
        n : int
            The number of top sequences to return.

        Returns
        -------
        tuple[list[str], list[float]]
            A tuple containing two lists:
                - The top n sequences sorted by score in descending order.
                - Their corresponding scores.
        """
        combined = list(zip(self.sequences, self.scores))
        top = sorted(combined, key=lambda x: x[1], reverse=True)[:n]
        if top:
            top_seqs, top_scores = zip(*top)
            return list(top_seqs), list(top_scores)
        return [], []

    def _apply_liability_filtering(self, sequences: List[str]) -> List[str]:
        """
        Apply liability filtering to a list of sequences if enabled.

        Parameters
        ----------
        sequences : List[str]
            The sequences to filter

        Returns
        -------
        List[str]
            Filtered sequences if liability filtering is enabled, otherwise the original sequences
        """
        if not self.liability_filtering or not sequences:
            return sequences

        filtered_sequences = filter_by_liability(sequences, aa_threshold=self.liability_threshold)
        n_elim = len(sequences) - len(filtered_sequences)

        if n_elim > 0:
            self.logger.info(f"Filtering sequences by liability eliminated {n_elim} sequences.")

        return filtered_sequences


class GeneticAlgorithm(BaseGeneticAlgorithm):
    """Genetic algorithm with a fixed number of rounds."""

    def __init__(
        self,
        initial_population: List[str],
        scoring_function: Callable[[List[str]], List[float]],
        aa_vocabulary: Dict[int, str],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        probability_matrix: Dict[int, Dict[str, float]] | None = None,
        population_size: int = 100,
        repetitions: int = 10,
        tournament_size: int = 5,
        rng: random.Random = random.Random(42),
        liability_filtering: bool = False,
        liability_threshold: float = 1.0,
        parental_sequence: str = None,
        max_mutations: int | None = None,
        max_mutations_per_cdr: Dict[str, int] = {},
        cdr_positions: Dict[str, List[int]] = {},
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the GeneticOptimizer with parameters for mutation, crossover, population size, and more.

        Args:
            initial_population : List[str]
                The initial population of sequences to start the genetic algorithm.
            scoring_function : Callable[[List[str]], List[float]]
                Function to evaluate the fitness of multiple sequences. It takes a list of sequences
                as input and returns a corresponding list of scores as output.
            aa_vocabulary : Dict[int, str]
                Alphabet of possible amino acids for mutation. Either position-based or per-AA-based vocabulary.
            mutation_rate : float, optional
                The probability of mutation for each offspring sequence (default is 0.1).
            crossover_rate : float, optional
                The probability of performing crossover between two parents (default is 0.7).
            probability_matrix: Dict[int, Dict[str, float]], optional
                Probability matrix from a generative model that guides the single point mutation operation.
            population_size : int, optional
                The desired size of the resulting new population (default is 100).
            repetitions : int, optional
                The maximum number of times a sequence can be added to the new population
                before it is considered fully evaluated (default is 10).
            tournament_size : int
                The number of individuals to participate in the tournament selection for parent selection.
            rng : random.Random, optional
                Random number generator for reproducibility. Defaults to random.Random(42).
            liability_filtering : bool, optional
                Whether to filter sequences by liability score (default is True).
                If True, sequences with high liability scores will be filtered out.
            liability_threshold : float, optional
                The threshold for the average liability score per amino acid (default is 1.0).
                Sequences with scores higher than this threshold will be filtered out.
            parental_sequence : str, optional
                The parental sequence to be used as a reference for counting mutations.
            max_mutations : int, optional
                The maximum number of mutations each scored sequence can have if limited.
            max_mutations_per_cdr : dict[str, int] | None, optional
                The maximum number of mutations allowed per CDR region. If None, no limit is applied.
            cdr_positions : dict[str, list[int]] | None, optional
                A dictionary defining the positions of CDR regions in the sequence.
                If None, no CDR-based limits are applied.
            logging_level : int, optional
                Logging level for the genetic algorithm (default is logging.INFO).
        """
        if not isinstance(rng, random.Random):
            raise TypeError(f"rng must be an int or random.Random instance, got {type(rng)}")
        self.rng = rng

        # Input checks
        if max_mutations is not None and parental_sequence is None:
            raise ValueError("If `max_mutations` is provided, `parental_sequence` must also be provided.")

        if max_mutations_per_cdr and (not cdr_positions and not parental_sequence):
            raise ValueError(
                "If `max_mutations_per_cdr` is provided, `cdr_positions` and `parental_sequence` must also be provided."
            )

        # Set up constraint handler
        constraint_handler = create_constraint_handler(
            parental_sequence=parental_sequence,
            max_mutations=max_mutations,
            max_mutations_per_cdr=max_mutations_per_cdr,
            cdr_positions=cdr_positions,
        )

        super().__init__(
            initial_population=initial_population,
            scoring_function=scoring_function,
            aa_vocabulary=aa_vocabulary,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            probability_matrix=probability_matrix,
            population_size=population_size,
            repetitions=repetitions,
            tournament_size=tournament_size,
            rng=self.rng,
            liability_filtering=liability_filtering,
            liability_threshold=liability_threshold,
            constraint_handler=constraint_handler,
            logging_level=logging_level,
        )

    def run(
        self,
        max_rounds: int = 100,
    ) -> Dict[str, List]:
        """
        Run a genetic algorithm for a fixed number of rounds.

        Parameters
        ----------
        max_rounds : int, optional
            Number of generations to run the algorithm for (default is 100).
            Stops when the iteration count reaches this number.

        Returns
        -------
        Dict[str, List]
            A dictionary containing:
                - "final_sequences": The final population of sequences.
                - "final_scores": The scores of the final population.
                - "all_sequences": All evaluated sequences over the generations.
                - "all_scores": The scores of all evaluated sequences.
        """
        self.sequences_counter = Counter(self.sequences)

        for _ in tqdm(range(self.iteration, max_rounds), desc=GA_TITLE):
            self.iteration += 1

            # get sequences and scores from all generations and rank them
            ranked_sequences, ranked_scores = rank_sequences(
                self.generations[-1].sequences,
                self.generations[-1].scores,
            )

            self.logger.debug(f"Generation {self.iteration}")
            for i in range(min(10, len(ranked_sequences))):
                self.logger.debug(f"Top {i+1} Sequence: {ranked_sequences[i]}, Score: {ranked_scores[i]}")

            self.logger.debug(LOG_GENERATING_NEW_POPULATION)
            new_sequences = self.generate_new_population(
                sequences=self.generations[-1].sequences,
                scores=self.generations[-1].scores,
            )

            # filter for liability
            new_sequences = self._apply_liability_filtering(new_sequences)

            # apply mutation constraints
            new_sequences = [self.constraint_handler.repair_sequence(seq) for seq in new_sequences]

            self.logger.debug(LOG_SCORING_NEW_POPULATION)
            new_scores = self.scoring_function(new_sequences)

            # Update the generations
            new_generation = Generation(
                sequences=new_sequences,
                scores=new_scores,
                number=self.iteration,
            )

            self.generations.append(new_generation)

        return {
            "all_sequences": self.sequences,
            "all_scores": self.scores,
        }


class BatchGeneticAlgorithm(BaseGeneticAlgorithm):
    """Genetic algorithm for batches of sequences with a fixed number of rounds."""

    def __init__(
        self,
        initial_population: List[str],
        batch_size: int,
        scoring_function: Callable[[List[List[str]]], List[float]],
        aa_vocabulary: Dict[int, str],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        probability_matrix: Dict[int, Dict[str, float]] | None = None,
        batch_crossover_rate: float = 0.2,
        single_crossover_rate: float = 0.5,
        population_size: int = 100,
        repetitions: int = 10,
        tournament_size: int = 5,
        rng: random.Random = random.Random(42),
        liability_filtering: bool = False,
        liability_threshold: float = 1.0,
        parental_sequence: str = None,
        max_mutations: int | None = None,
        max_mutations_per_cdr: Dict[str, int] | None = {},
        cdr_positions: Dict[str, list[int]] | None = {},
        logging_level: int = logging.INFO,
    ):
        """
        Initialize the GeneticOptimizer with parameters for mutation, crossover, population size, and more.

        Args:
            initial_population : List[str]
                The initial population of sequences to start the genetic algorithm.
            batch_size : int
                The number of points to select per iteration (q in batch acquisition functions).
            scoring_function : Callable[[List[List[str]]], List[float]]
                Function to evaluate the fitness of multiple lists of sequences. It takes lists of lists of
                sequences as input and returns a corresponding list of scores for each list as output.
            aa_vocabulary : Dict[int, str]
                Alphabet of possible amino acids for mutation. Either position-based or per-AA-based vocabulary.
            mutation_rate : float, optional
                The probability of mutation for each offspring sequence (default is 0.1).
            crossover_rate : float, optional
                The probability of performing crossover between two parent batches (default is 0.7).
            probability_matrix: Dict[int, Dict[str, float]], optional
                Probability matrix from a generative model that guides the single point mutation operation.
            batch_crossover_rate : float, optional
                The probability of swapping sequences between two batches in crossover (default is 0.2).
            single_crossover_rate : float, optional
                The probability of performing single-point crossover between two sequences in batch (default is 0.5).
            population_size : int, optional
                The desired size of the resulting new population (default is 100).
            repetitions : int, optional
                The maximum number of times a sequence can be added to the new population
                before it is considered fully evaluated (default is 10).
            tournament_size : int
                The number of individuals to participate in the tournament selection for parent selection.
            rng : random.Random, optional
                Random number generator for reproducibility. Defaults to random.Random(42).
            liability_filtering : bool, optional
                Whether to filter sequences by liability score (default is False).
                If True, sequences with high liability scores will be filtered out.
            liability_threshold : float, optional
                The threshold for the average liability score per amino acid (default is 1.0).
                Sequences with scores higher than this threshold will be filtered out.
            parental_sequence : str, optional
                The parental sequence to be used as a reference for counting mutations.
            max_mutations : int, optional
                The maximum number of mutations each scored sequence can have if limited.
            max_mutations_per_cdr : Dict[str, int] | None, optional
                The maximum number of mutations allowed per CDR region. If None, no limit is applied.
            cdr_positions : Dict[str, list[int]] | None, optional
                A dictionary defining the positions of CDR regions in the sequence.
                If None, no CDR-based limits are applied.
            logging_level : int, optional
                Logging level for the genetic algorithm (default is logging.INFO).
        """
        if not isinstance(rng, random.Random):
            raise TypeError(f"rng must be an int or random.Random instance, got {type(rng)}")
        self.rng = rng

        if max_mutations is not None and parental_sequence is None:
            raise ValueError("If `max_mutations` is provided, `parental_sequence` must also be provided.")

        if max_mutations_per_cdr and (not cdr_positions and not parental_sequence):
            raise ValueError(
                "If `max_mutations_per_cdr` is provided, `cdr_positions` and `parental_sequence` must also be provided."
            )

        constraint_handler = create_constraint_handler(
            parental_sequence=parental_sequence,
            max_mutations=max_mutations,
            max_mutations_per_cdr=max_mutations_per_cdr,
            cdr_positions=cdr_positions,
        )

        super().__init__(
            initial_population=initial_population,
            scoring_function=scoring_function,
            aa_vocabulary=aa_vocabulary,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            probability_matrix=probability_matrix,
            population_size=population_size,
            repetitions=repetitions,
            tournament_size=tournament_size,
            rng=rng,
            liability_filtering=liability_filtering,
            liability_threshold=liability_threshold,
            constraint_handler=constraint_handler,
            logging_level=logging_level,
        )

        self.batch_crossover_rate = batch_crossover_rate
        self.single_crossover_rate = single_crossover_rate
        self.batch_size = batch_size

        initial_batches = self._create_initial_batches(self.sequences)  # Initialise with batches

        # initialize the iteration counter to keep track of the number of generations
        self.iteration = 0

        # Track the generations
        initial_generation = Generation(
            sequences=initial_batches,
            scores=self._evaluate_batches_in_chunks(initial_batches, scoring_function),
            number=self.iteration,
        )
        self.generations = [initial_generation]

        # Convert batches to hashable tuples first
        self.sequences_counter = Counter(tuple(tuple(seq) for seq in batch) for batch in initial_batches)

    def _evaluate_batches_in_chunks(
        self, batches: List[List[str]], scoring_function, chunk_size: int = 8
    ) -> List[float]:
        """Evaluate batches in smaller chunks to avoid OOM."""
        all_scores = []

        chunk_size = min(chunk_size, len(batches))
        for i in range(0, len(batches), chunk_size):
            chunk = batches[i : i + chunk_size]
            logging.debug(
                f"Evaluating chunk {i//chunk_size + 1}/{(len(batches) + chunk_size - 1)//chunk_size} "
                f"({len(chunk)} batches, {len(chunk) * self.batch_size} total sequences)"
            )

            chunk_scores = scoring_function(chunk)
            all_scores.extend(chunk_scores)

        return all_scores

    def _create_initial_batches(self, sequences: List[str]) -> List[List[str]]:
        """Create initial batches from individual sequences."""
        # Ensure we have enough sequences
        if len(sequences) < self.batch_size:
            raise ValueError(f"Need at least {self.batch_size} sequences to create batches")

        # Create population_size batches by randomly sampling from sequences
        # Each batch has batch_size sequences
        batches = []
        for _ in range(self.population_size):
            batch = self.rng.sample(sequences, self.batch_size)
            batches.append(batch)

        return batches

    def generate_new_population(
        self,
        sequences: List[List[str]],
        scores: List[float],
    ) -> List[List[str]]:
        """Generate a new population using crossover and mutation for batches.

        Parameters
        ----------
        sequences : List[List[str]]
            A list of lists of string sequences from which new sequences will be generated.
        scores : List[float]
            A list of scores corresponding to each batch, used for selection.

        Returns
        -------
        List[List[str]]
            A list of new batches of sequences generated for the population.
        """
        new_population_count = {}
        new_population = []

        while len(new_population) < self.population_size:
            # 1. Select parents
            parent1_batch = self.tournament_selection(sequences, scores, self.tournament_size)
            parent2_batch = self.tournament_selection(sequences, scores, self.tournament_size)
            # 2. Batch cross-over
            if self.rng.random() < self.crossover_rate:
                offspring1_batch, offspring2_batch = batch_crossover(
                    parent1_batch, parent2_batch, self.single_crossover_rate, self.batch_crossover_rate, self.rng
                )
            else:
                offspring1_batch, offspring2_batch = parent1_batch.copy(), parent2_batch.copy()

            # 3. Batch mutate, taking care of constraints
            adjusted_vocabs = [
                self.constraint_handler.adjust_vocabulary(seq, self.aa_vocabulary) for seq in offspring1_batch
            ]

            offspring1_batch = mutate_batch(
                offspring1_batch,
                self.mutation_rate,
                adjusted_vocabs,
                self.rng,
                probability_matrix=self.probability_matrix,
            )

            adjusted_vocabs = [
                self.constraint_handler.adjust_vocabulary(seq, self.aa_vocabulary) for seq in offspring2_batch
            ]
            offspring2_batch = mutate_batch(
                offspring2_batch,
                self.mutation_rate,
                adjusted_vocabs,
                self.rng,
                probability_matrix=self.probability_matrix,
            )

            # 4. Add offspring batches if it's not already been evaluated "repetitions" times
            for offspring_batch in (offspring1_batch, offspring2_batch):
                batch_tuple = tuple(tuple(seq) for seq in offspring_batch)
                total_count = self.sequences_counter.get(batch_tuple, 0) + new_population_count.get(batch_tuple, 0)
                if self.repetitions > total_count:
                    new_population.append(offspring_batch)
                    new_population_count[batch_tuple] = new_population_count.get(batch_tuple, 0) + 1

        return new_population[: self.population_size]

    def run(
        self,
        max_rounds: int = 100,
    ) -> Dict[str, List]:
        """
        Run a batch genetic algorithm for a fixed number of rounds.

        Parameters
        ----------
        max_rounds : int, optional
            Number of generations to run the algorithm for (default is 100).
            Stops when the iteration count reaches this number.

        Returns
        -------
        Dict[str, List]
            A dictionary containing:
                - "final_sequences": The final population of sequences.
                - "final_scores": The scores of the final population.
                - "all_sequences": All evaluated sequences over the generations.
                - "all_scores": The scores of all evaluated sequences.
        """
        self.sequences_counter = Counter(tuple(tuple(seq) for seq in batch) for batch in self.sequences)

        for _ in tqdm(range(self.iteration, max_rounds), desc=GA_TITLE):
            self.iteration += 1

            # get sequences and scores from all generations and rank them
            ranked_sequences, ranked_scores = rank_sequences(
                self.generations[-1].sequences,
                self.generations[-1].scores,
            )

            self.logger.debug(f"Generation {self.iteration}")
            for i in range(min(10, len(ranked_sequences))):
                self.logger.debug(f"Top {i+1} Sequence: {ranked_sequences[i]}, Score: {ranked_scores[i]}")
            self.logger.debug(LOG_GENERATING_NEW_POPULATION)
            new_sequences = self.generate_new_population(
                sequences=self.generations[-1].sequences,
                scores=self.generations[-1].scores,
            )

            new_sequences = self._process_new_sequences(new_sequences)

            self.logger.debug(LOG_SCORING_NEW_POPULATION)
            new_scores = self.scoring_function(new_sequences)

            # Update the generations
            new_generation = Generation(
                sequences=new_sequences,
                scores=new_scores,
                number=self.iteration,
            )

            self.generations.append(new_generation)

        return {
            "all_batches": self.sequences,
            "all_scores": self.scores,
        }

    def tournament_selection(self, batches: List[List[str]], scores: List[float], tournament_size: int) -> List[str]:
        """Select a batch using tournament selection.

        Parameters
        ----------
        batches : List[List[str]]
            The batches of sequences to select from.
        scores : List[float]
            The scores associated with each batch.
        tournament_size : int
            Number of batches to include in the tournament.

        Returns
        -------
        List[str]
            The winning batch from the tournament.
        """
        selected = self.rng.sample(list(zip(batches, scores)), tournament_size)
        return max(selected, key=lambda x: x[1])[0].copy()

    def get_top_n(self, n: int) -> tuple[list[list[str]], list[float]]:
        """
        Return the top n batches and their corresponding scores from all generations.

        Parameters
        ----------
        n : int
            The number of top batches to return.

        Returns
        -------
        tuple[list[list[str]], list[float]]
            A tuple containing two lists:
                - The top n batches sorted by score in descending order.
                - Their corresponding scores.
        """
        # Use rank_batches utility function that's already defined
        ranked_batches, ranked_scores = rank_sequences(self.sequences, self.scores)
        return ranked_batches[:n], ranked_scores[:n]

    def _process_new_sequences(self, new_sequences: List) -> List:
        """Apply mutation limits and liability filtering to new sequences."""
        # Apply mutation limits if enabled
        new_sequences = self._apply_mutation_constraints(new_sequences)

        # Apply liability filtering if enabled
        if self.liability_filtering:
            new_sequences = self._apply_liability_filtering(new_sequences)

        return new_sequences

    def _apply_mutation_constraints(self, new_sequences: List) -> List:
        """Apply mutation limits to new sequences."""
        return [self._adjust_batch(batch) for batch in new_sequences]

    def _adjust_batch(self, batch: list[str]) -> list[str]:
        """
        Adjust all sequences in a batch so that each has at most the desired number of mutations.

        Parameters
        ----------
        batch : list of str
            The batch of sequences to adjust.

        Returns
        -------
        list of str
            The adjusted batch of sequences.
        """
        return [self.constraint_handler.repair_sequence(seq) for seq in batch]

    def _apply_liability_filtering(self, new_sequences: List) -> List:
        """Apply liability filtering to batches of sequences."""
        filtered_batches = []
        total_eliminated = 0

        for batch in new_sequences:
            # Apply liability filtering to each batch
            filtered_batch = filter_by_liability(batch, aa_threshold=self.liability_threshold)
            # Only keep batches that still have at least one sequence

            if len(filtered_batch) == self.batch_size:
                filtered_batches.append(filtered_batch)
                total_eliminated += len(batch) - len(filtered_batch)
            else:
                # The entire batch was filtered out
                total_eliminated += len(batch)

        logging.info(f"Filtering sequences by liability eliminated {total_eliminated} sequences across all batches.")
        logging.info(f"Remaining batches: {len(filtered_batches)} out of {self.population_size}")

        return filtered_batches
