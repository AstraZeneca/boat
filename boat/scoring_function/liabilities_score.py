"""Liability scoring function implementation."""

from boat.biologics.liabilities import score_sequence
from boat.scoring_function.interface import SingleObjectiveScoringFunction


class LiabilityScoringFunction(SingleObjectiveScoringFunction):
    """Scoring function that computes liability scores for antibody sequences.

    Lower scores are better (fewer liabilities)
    """

    def __init__(self, name: str = "Liability", invert: bool = True):
        """
        Initialize the liability scoring function.

        Args:
            name: Name of the scoring function.
            output_type: Type of output, either "continuous" or "binary".
            invert: If True, invert scores so higher is better.
        """
        super().__init__(name=name, parental=None, output_type="continuous")
        self.invert = invert

    def __call__(self, sequences: list[str]) -> dict[str, list[float]]:
        """
        Evaluate a list of sequences using the liability scoring function.

        Args:
            sequences: A list of antibody sequences to score.

        Returns
        -------
            A list of scores corresponding to the sequences.
        """
        scores = [score_sequence(seq) for seq in sequences]
        if self.invert:
            scores = [-score for score in scores]
        return {self.objective_names[0]: scores}


if __name__ == "__main__":
    # Example usage
    scoring_function = LiabilityScoringFunction()
    sequences = ["CAVDGVV", "AKGDKGD", "AAAAKM"]
    scores = scoring_function(sequences)
    print(scores)  # Should print the liability scores for the sequences
