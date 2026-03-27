"""Collection of fake scoring functions for testing purposes."""

from collections import Counter

from .interface import SingleObjectiveScoringFunction


class FakeMultimodalScoringFunction(SingleObjectiveScoringFunction):
    """Fake multimodal scoring function for testing purposes.

    This scoring function assigns non-zero weight to specified amino acids and returns the maximum score of
    recurring amino acids in the sequence, e.g. if the weights are {'A': 1.2, 'K': 1.0, 'Q': 0.8} and the sequence is
    "AKQAKQ", the score will be 2.4, as "A" appears twice and has the highest weight. The score of "AKKAKQ" will be 3.0,
    as three "K"s have a higher score than two "A"s or one "Q".
    """

    def __init__(self, weights: dict[str, float] = None):
        """Initialize the fake multimodal scoring function.

        Args:
            weights (dict[str, float], optional): A dictionary mapping amino acids to their weights.
                If None, default weights are used.
            output_type (str): Type of output, either "continuous" or "binary".
        """
        name = "_".join([str(f) for f in weights.values()]) if weights else ""
        name = f"fake_{name}"
        super().__init__(name=name, parental=None, output_type="continuous")
        if not weights:
            weights = {"A": 1.2, "K": 1.0, "Q": 0.8}
        self.weights = weights

    def __call__(self, sequences: list[str]) -> dict[str, list[float]]:
        """Evaluate a list of sequences using the fake scoring function.

        Args:
            sequences (list[str]): A list of sequences to score.

        Returns
        -------
            score (dict): A dict with the name of the scoring and a list of scores.
            The name needs to include binary or regression.
        """
        scores = []
        for sequence in sequences:
            # Collect the generated scores and select the maximum value.
            score = max(self._score_aas(sequence, self.weights), default=0.0)
            scores.append(score)
        return {self.objective_names[0]: scores}

    def _score_aas(self, sequence: str, weights: dict):
        """Generate scores for each amino acid in the sequence."""
        aa_counter = Counter(sequence)
        for aa in aa_counter.keys():
            yield weights.get(aa, 0) * aa_counter[aa]
