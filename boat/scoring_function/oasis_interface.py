"""Defining an interface for the OASis scoring of humanness."""

from typing import Optional

from promb import init_db

from ..biologics.sequence import AbVSeq, abvseq_to_str
from .interface import SingleObjectiveScoringFunction


class OASisScoringFunction(SingleObjectiveScoringFunction):
    """Interface for OASis to be used as a scoring function."""

    def __init__(
        self, parental: AbVSeq, threshold: Optional[float] = None, reference_db: str = "human-oas", name: str = "OASis"
    ):
        """
        Initialize the scoring function interface.

        Args:
            parental (AbVSeq): The parental sequence as an AbVSeq object.
            threshold (Optional[float]): Threshold for binary classification. If None, regression is used.
            reference_db (str): Reference database to use ("human-oas" or "human-swissprot").
            name (str): Name of the scoring function (default: "OASis").
        """
        output_type = "binary" if threshold is not None else "continuous"
        super().__init__(name=name, parental=parental, output_type=output_type)

        # Initialize for the use of optimization on one CDR only
        self.current_cdr = None

        if reference_db == "human-oas":
            self.db = init_db(reference_db)
        elif reference_db == "human-swissprot":
            self.db = init_db(reference_db, 9)
        self.threshold = threshold

    def __call__(self, sequences: list[str], *args, **kwargs) -> dict[str, list[float]]:
        """
        Evaluate a list of sequences using the scoring function.

        Args:
            sequences (list[str]): A list of sequences to score.
                These sequences are either concatenated heavy and light chain or CDR regions,
                in which case they have to be transformed back to the full sequence before scoring.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            score (dict): A dict with the name of the scoring and a list of scores.
        """
        if self.current_cdr is not None:
            # Run in CDR optimization mode
            sequences = self._reconstruct_sequences_from_cdr(sequences)
        else:
            sequences = self._reconstruct_full_sequences(sequences)

        scores = []

        for seq in sequences:
            seq_str = abvseq_to_str(seq)
            score = self.db.compute_peptide_content(seq_str[0])

            if self.threshold:
                score = float(score > self.threshold)
            scores.append(score)

        return {self.objective_names[0]: scores}
