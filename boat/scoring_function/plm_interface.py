"""Interface for PLM log-likelihoods."""
import logging

import numpy as np
from esm.pretrained import load_model_and_alphabet
from torch.nn.functional import softmax

from ..biologics.sequence import AbVSeq
from .interface import SingleObjectiveScoringFunction


class ESMInterface(SingleObjectiveScoringFunction):
    """ESM interface."""

    def __init__(self, parental: AbVSeq, checkpoint_path: str, device: str = "cpu", name: str = "esm"):
        """ESM interface.

        Args:
            parental (AbVSeq): The parental sequence as an AbVSeq object.
            checkpoint_path (str): Path to the ESM model checkpoint.
            device (str): Device to load the model on (default: "cpu").
            name (str): Name of the scoring function (default: "esm").
        """
        super().__init__(name=name, parental=parental, output_type="continuous")
        self.device = device

        logging.info(f"Loading ESM model from {checkpoint_path}")
        self.model, self.alphabet = load_model_and_alphabet(checkpoint_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.aa2id = self.alphabet.tok_to_idx

        self.model.eval()
        self.model.to(self.device)

        self.current_cdr = None

    def esm_logits(self, batch):
        """Query esm model to receive logits."""
        _, _, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(self.device)

        hidden_states = self.model(batch_tokens)
        return hidden_states["logits"]

    def mean_log_likelihood(self, batch):
        """Calculate mean log likelihoods."""
        lengths = [len(seq[1]) for seq in batch]  # or light_chains
        assert (
            all(len_seq == lengths[0] for len_seq in lengths) and lengths[0] > 0
        ), "Not all sequences are the same length."

        batch_size = len(batch)
        logits = self.esm_logits(batch)

        mean_ll = [None for _ in range(batch_size)]
        for i in range(batch_size):
            seq = batch[i][1]
            seq_indices = [self.aa2id[x] for x in seq]

            # Select positions corresponding to amino acids and drop any special tokens
            aa_logits = logits[i, 1:-1, :].detach().cpu()

            # Apply the softmax function to convert logits to probabilities
            likelihood = softmax(aa_logits, dim=-1)
            log_probs = np.log10([likelihood[x, seq_indices[x]] for x in range(len(seq))])
            mean_ll[i] = np.mean(log_probs)
        return mean_ll

    def __call__(self, sequences) -> dict[str, list[float]]:
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
        heavy_chains, light_chains = self._get_input_sequences(sequences)

        scores = []
        if heavy_chains and light_chains:
            mean_ll_heavy = self.mean_log_likelihood(heavy_chains)
            mean_ll_light = self.mean_log_likelihood(light_chains)
            scores = [(heavy + light) / 2 for heavy, light in zip(mean_ll_heavy, mean_ll_light)]
        elif heavy_chains:
            scores = self.mean_log_likelihood(heavy_chains)
        elif light_chains:
            scores = self.mean_log_likelihood(light_chains)
        else:
            raise ValueError("No heavy or light chains found in input sequences.")

        return {self.objective_names[0]: scores}

    def _get_input_sequences(self, abvseqs: list[AbVSeq]) -> tuple[list, list]:
        """
        Extract heavy and light chain sequences from a list of AbVSeq objects.

        Args:
            abvseqs (list[AbVSeq]): A list of AbVSeq objects.

        Returns
        -------
            tuple: A tuple containing the heavy and light chain sequences.
        """
        has_heavy = bool(self.parental.heavy_chain)
        has_light = bool(self.parental.light_chain)

        if has_heavy and has_light:  # both chains are present
            heavy_chains = [(f"heavy chain {i}", seq.heavy_chain) for i, seq in enumerate(abvseqs)]
            light_chains = [(f"light chain {i}", seq.light_chain) for i, seq in enumerate(abvseqs)]
        elif has_heavy and not has_light:  # only heavy chain is present
            heavy_chains = [(f"heavy chain {i}", seq.heavy_chain) for i, seq in enumerate(abvseqs)]
            light_chains = []
        elif has_light and not has_heavy:  # only light chain is present
            heavy_chains = []
            light_chains = [(f"light chain {i}", seq.light_chain) for i, seq in enumerate(abvseqs)]

        return heavy_chains, light_chains
