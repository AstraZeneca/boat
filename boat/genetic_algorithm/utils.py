"""Contains the helper functions for the genetic algorithm."""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from boat.genetic_algorithm.genetic_operators import _get_mutable_positions, mutate_sequence


@dataclass
class Generation:
    """Class to represent a generation in the genetic algorithm."""

    sequences: List[str]
    scores: List[float]
    number: int


def rank_sequences(sequences: list[str] | list[list[str]], scores: List[float]) -> Tuple[List[str], List[float]]:
    """Rank sequences or batches of sequences based on their scores.

    Parameters
    ----------
    sequences : The sequences to be ranked. Either:
        - List[str]: Individual sequences.
        - List[List[str]]: Batches of sequences.
    scores : list of float
        The scores associated with each sequence.

    Returns
    -------
    tuple of (list of str, list of float)
        Ranked sequences and their corresponding scores.
    """
    combined = list(zip(sequences, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in combined], [score for _, score in combined]


def count_mutations(seq1: str, seq2: str) -> int:
    """Count the number of differing positions between two sequences."""
    return sum(1 for a, b in zip(seq1, seq2) if a != b)


def generate_random_point_mutations(
    sequence: str,
    aa_vocabulary: Dict[int | str, str],
    population_size: int = 100,
    max_point_mutations: int = 1,
    rng: random.Random = random.Random(42),
    probability_matrix: dict[int, dict[str, float]] | None = None,
) -> List[str]:
    """
    Generate a population of sequences from a given parental sequence by introducing up to n point mutations.

    Parameters
    ----------
    sequence : str
        The original sequence to mutate.
    aa_vocabulary : Dict[int | str, str], optional
        Alphabet of possible amino acids, either positional or per AA.
    population_size : int
        The number of mutated sequences to generate.
    max_point_mutations : int
        The maximum number of mutations to introduce in each generated sequence.
    rng : random.Random, optional
        Random number generator for reproducibility. Defaults to random.Random(42).
    probability_matrix: dict[int, dict[str, float]], optional
        Probability matrix from a generative prior to guide point mutations.

    Returns
    -------
    List[str]
        A list of mutated sequences.
    """
    population = []  # Use a list to preserve order.
    attempts = 0

    # Generate mutations
    max_mutable_positions = len(_get_mutable_positions(list(sequence), aa_vocabulary))
    max_point_mutations = min(max_point_mutations, max_mutable_positions)
    mutation_rate = max_point_mutations / max_mutable_positions

    max_attempts = population_size * max_point_mutations * 20
    attempts = 0

    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        new_seq = mutate_sequence(
            sequence,
            mutation_rate=mutation_rate,
            rng=rng,
            aa_vocabulary=aa_vocabulary,
            probability_matrix=probability_matrix,
        )
        if new_seq not in population and new_seq != sequence:
            population.append(new_seq)

    return population
