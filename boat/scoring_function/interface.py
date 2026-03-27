"""Defining an interface for scoring functions used in the genetic algorithms."""

from abc import ABC, abstractmethod
from typing import List

from boat.biologics.sequence import CDR, AbVSeq, cdr_to_seq


class ScoringFunctionInterface(ABC):
    """Base class for scoring functions used in genetic algorithms."""

    def __init__(self, name):
        """
        Initialize the scoring function interface.

        Args:
            name (str): The name of the scoring function.
        """
        self.name = name

    def __repr__(self):
        """
        Return a string representation of the scoring function.

        Returns
        -------
            str: The name of the scoring function.
        """
        return self.name

    @abstractmethod
    def __call__(self, sequences, *args, **kwargs):
        """
        Evaluate a list of sequences using the scoring function.

        Args:
            sequences (list): A list of sequences to score.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            score: The score or a list of scores corresponding to the sequences.
        """
        pass


class SingleObjectiveScoringFunction(ScoringFunctionInterface):
    """A single-objective scoring function for evaluating sequences.

    A single-objective scoring function evaluates a single objective.
    This objective can be single or multi-output, the latter occuring when a single function call returns
    multiple metrics.
    """

    def __init__(
        self,
        name,
        parental: AbVSeq | None = None,
        output_type: str = "continuous",
    ):
        """Initialize the single-objective scoring function.

        Args:
            name (str): The name of the scoring function.
            parental (AbVSeq, optional): The parental sequence as an AbVSeq object.
            output_type (str, optional): The type of output, either "continuous" or "binary".
        """
        super().__init__(name)
        self.parental = parental
        self.objective_types = self._set_output_types(output_type)
        self.objective_names = list(self.objective_types.keys())
        self.objective_dimension = len(self.objective_types)

    def _set_output_types(self, output_type: str) -> dict:
        """
        Set the output type and metrics to score.

        Args:
            output_type (str): The type of output, either "continuous" or "binary".
        """
        if output_type not in ("continuous", "binary"):
            raise ValueError(f"Unknown output type '{output_type}' for scoring function '{self.name}'")

        name = "_".join([self.name, output_type[0]])
        return {name: output_type}

    def _reconstruct_sequences_from_cdr(self, sequences: list[str]) -> list[AbVSeq]:
        """
        Reconstruct the full sequences from the CDR string representations.

        Args:
            sequences (list[str]): A list of CDR sequences.

        Returns
        -------
            list[AbVSeq]: A list of AbVSeq objects representing the full sequences.
        """
        # perform some checks
        self._cdr_checks(sequences)

        # transform sequences to full sequences
        sequences = [CDR(id=self.current_cdr.id, pos=self.current_cdr.pos, sequence=seq) for seq in sequences]
        sequences = [cdr_to_seq(cdr, self.parental) for cdr in sequences]

        return sequences

    def _reconstruct_full_sequences(self, sequences: list[str]) -> list[AbVSeq]:
        """
        Reconstruct a list of strings to a list of AbVSeq objects.

        The full sequences come in concatenated, i.e. the chains have to be separated using info from the parent.

        Args:
            sequences (list[str]): A list of sequences to score.

        Returns
        -------
            list[AbVSeq]: A list of AbVSeq objects representing the full sequences.
        """
        # perform some checks
        self._full_sequence_checks(sequences)

        # reconstruct chains from pure sequences
        split_position = len(self.parental.heavy_chain)
        sequences = [AbVSeq(heavy_chain=seq[:split_position], light_chain=seq[split_position:]) for seq in sequences]

        return sequences

    def _cdr_checks(self, sequences: list[str]):
        """
        Perform checks on the CDR sequences.

        Args:
            sequences (list[str]): A list of CDR sequences to check.
        """
        if self.current_cdr is None:
            raise ValueError("CDR is not set. Use set_cdr() to set the CDR before scoring.")

        seq_lengths = set([len(seq) for seq in sequences])

        if seq_lengths != {len(self.current_cdr.sequence)}:
            raise ValueError(
                f"All sequences must have the same length as the CDR sequence ({len(self.current_cdr.sequence)})."
            )

    def _full_sequence_checks(self, sequences: list[str]):
        """
        Perform checks on the full sequences.

        Args:
            sequences (list[str]): A list of sequences to check.
        """
        seq_lengths = set([len(seq) for seq in sequences])

        if seq_lengths != {len("".join([self.parental.heavy_chain, self.parental.light_chain]))}:
            raise ValueError("All sequences must have the same length as the parental sequence.")

    def set_cdr(self, cdr: CDR):
        """
        Set the CDR to be optimized.

        Args:
            cdr (CDR): The CDR to be optimized.
        """
        self.current_cdr = cdr


class MultiObjectiveScoringFunction(ScoringFunctionInterface):
    """Combine multiple single-objective scoring functions into a multi-objective scoring function."""

    def __init__(
        self,
        scoring_functions: List[SingleObjectiveScoringFunction],
        args_list: List[tuple] = None,
        kwargs_list: List[dict] = None,
        name: str = "MultiObjectiveScoringFunction",
    ):
        """
        Initialize the multi-objective scoring function.

        Args:
            scoring_functions (list): A list of single-objective scoring functions.
            args_list (list): Optional list of tuples containing additional positional arguments
                for each scoring function.
            kwargs_list (list): Optional list of dictionaries containing additional keyword arguments
                for each scoring function.
            name (str): Optional name for this multi-objective scoring function (defaults to "MultiObjective")
        """
        super().__init__(name)
        self.scoring_functions = scoring_functions

        # Set up function-specific arguments
        self.args_list = args_list or [() for _ in range(len(scoring_functions))]
        self.kwargs_list = kwargs_list or [{} for _ in range(len(scoring_functions))]

        # Validate argument lists
        if len(self.args_list) != len(scoring_functions):
            raise ValueError(f"Expected {len(scoring_functions)} args tuples, got {len(self.args_list)}")
        if len(self.kwargs_list) != len(scoring_functions):
            raise ValueError(f"Expected {len(scoring_functions)} kwargs dicts, got {len(self.kwargs_list)}")

        # Output names, types, and total dimension of scores, as some objectives might be multi-output
        self.objective_types = {}
        self.objective_names = []
        for func in scoring_functions:
            self.objective_types.update(func.objective_types)
            self.objective_names.extend(func.objective_names)
        self.objective_dimension = len(self.objective_names)

    def __call__(self, sequences, *args, **kwargs):
        """
        Evaluate a list of sequences using the multi-objective scoring function.

        Args:
            sequences (list): A list of sequences to score.
            *args: Additional positional arguments (passed to all scoring functions).
            **kwargs: Additional keyword arguments (passed to all scoring functions).

        Returns
        -------
            dict: A dictionary mapping objective names to lists of scores for sequences.
        """
        n_sequences = len(sequences)
        results = {}

        # Iterate over each scoring function
        for i, func in enumerate(self.scoring_functions):
            # Merge global kwargs with function-specific kwargs
            merged_kwargs = {**kwargs, **self.kwargs_list[i]}
            merged_args = args + self.args_list[i]

            # Call the scoring function
            scores = func(sequences, *merged_args, **merged_kwargs)
            assert isinstance(scores, dict)

            for obj_name, obj_scores in scores.items():
                if not isinstance(obj_scores, list):
                    obj_scores = [obj_scores] * n_sequences
                if len(obj_scores) != n_sequences:
                    raise ValueError(
                        f"Function {func.name} objective {obj_name} returned {len(obj_scores)} \
                        scores for {n_sequences} sequences"
                    )
                results[obj_name] = obj_scores

        return results
