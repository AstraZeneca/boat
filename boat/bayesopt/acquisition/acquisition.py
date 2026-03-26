"""Implement acquisition functions on sequences for use with the genetic algorithm."""

import torch
from botorch.acquisition import AcquisitionFunction

from boat.bayesopt.encodings.encodings import Encoding
from boat.scoring_function.interface import ScoringFunctionInterface


class AcquisitionFunctionOnSequences(ScoringFunctionInterface):
    """Interface for using acquisition functions for evaluation with sequences."""

    def __init__(self, acquisition_function: AcquisitionFunction, encoding: Encoding, name: str):
        """
        Initialize the acquisition function interface.

        Args:
            name (str): The name of the acquisition function.
        """
        super().__init__(name)
        self.acquisition_function = acquisition_function
        self.encoding = encoding

    def __call__(self, sequences: list[str] | list[list[str]], *args, **kwargs):
        """
        Evaluate a list of sequences using the acquisition function.

        Args:
            sequences: Either:
                - List[str]: Individual sequences (traditional or q=1 batch)
                - List[List[str]]: Batches of sequences (q>1 batch)
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            score: The score or a list of scores corresponding to the sequences.
        """
        # Convert sequences to the appropriate format for the acquisition function
        # Handle both q=1 and q>1 cases
        if isinstance(sequences[0], str):
            # q=1 case: each sequence is evaluated individually
            encoded_sequences = self.encoding(sequences)
            # Add q dimension of size 1: [b, d] -> [b, 1, d]
            encoded_batches = encoded_sequences.unsqueeze(1)
        else:
            # q>1 case: batch evaluation
            all_batches = []
            for batch in sequences:
                encoded_batch = self.encoding(batch)
                all_batches.append(encoded_batch)
            # Stack along batch dimension to get [b, q, d]
            encoded_batches = torch.stack(all_batches)

        # Evaluate all batches
        return self.acquisition_function.forward(encoded_batches, *args, **kwargs).detach().cpu()
