"""Contains the basic genetic operators for the genetic algorithm."""

import random

import numpy as np

from boat.genetic_algorithm.vocabularies import aa_vocabulary_complete


def single_point_crossover(seq1: str, seq2: str, rng: random.Random = random.Random(42)) -> tuple[str, str]:
    """Perform single-point crossover on two sequences.

    Parameters
    ----------
    seq1 : str
        First sequence.
    seq2 : str
        Second sequence.
    rng : random.Random, optional
        Random number generator for reproducibility. Defaults to random.Random(42).

    Returns
    -------
    tuple of str
        Two new sequences after crossover.

    Raises
    ------
    ValueError
        If sequences are not of the same length.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")

    if isinstance(rng, int):
        rng = random.Random(rng)

    crossover_point = rng.randint(1, len(seq1) - 1)
    new_seq1 = seq1[:crossover_point] + seq2[crossover_point:]
    new_seq2 = seq2[:crossover_point] + seq1[crossover_point:]

    return new_seq1, new_seq2


def batch_crossover(
    batch1: list[str],
    batch2: list[str],
    single_crossover_rate: float = 0.7,
    batch_crossover_rate: float = 0.2,
    rng: random.Random = random.Random(42),
) -> tuple[list[str], list[str]]:
    """Perform crossover between two batches.

    This function performs a batch crossover by swapping sequences from two batches.
    Then it applies single point crossover between the sequences in the new batches.

    Args:
        batch1 : list[str]
            The first batch of sequences.
        batch2 : list[str]
            The second batch of sequences.
        single_crossover_rate : float, optional
            Probability of performing crossover between sequences (default is 0.7).
        batch_crossover_rate : float, optional
            Probability of performing crossover between batches (default is 0.2).
        rng : random.Random, optional
            Random number generator for reproducibility (default is random.Random(42)).

    Returns
    -------
        tuple[list[str], list[str]]
            Two new batches of sequences after crossover.
    """
    if len(batch1) == len(batch2) == 1:
        new1, new2 = single_point_crossover(batch1[0], batch2[0], rng=rng)
        return [new1], [new2]

    # shuffle batches
    batch1 = rng.sample(batch1, k=len(batch1))
    batch2 = rng.sample(batch2, k=len(batch2))

    new_batch1 = []
    new_batch2 = []

    for seq1, seq2 in zip(batch1, batch2):
        if rng.random() < single_crossover_rate:
            # perform single sequence crossover between sequences in batches
            # with certain probability, to also retain some sequences from the batches
            seq1, seq2 = single_point_crossover(seq1, seq2, rng=rng)

        # Batch crossover
        # Note that for batch_crossover_rate = 1, the batches are swapped
        if rng.random() < batch_crossover_rate:
            new_batch1.append(seq2)
            new_batch2.append(seq1)
        else:
            new_batch1.append(seq1)
            new_batch2.append(seq2)

    return new_batch1, new_batch2


def mutate_sequence(
    sequence: str,
    mutation_rate: float = 0.05,
    aa_vocabulary: dict[int | str, str] = aa_vocabulary_complete(),
    rng: random.Random = random.Random(42),
    probability_matrix: dict[int, dict[str, float]] | None = None,
) -> str:
    """
    Perform mutation on a sequence.

    Parameters
    ----------
    sequence : str
        The sequence to mutate.
    mutation_rate : float, optional
        Probability of each amino acid being mutated.
    aa_vocabulary : dict[int | str, str], optional
        Alphabet of possible amino acids for mutation, either position or AA-based.
        If not provided, defaults to aa_vocabulary_complete().
    rng : random.Random, optional
        Random number generator for reproducibility. Defaults to random.Random(42).
    probability_matrix: Dict[int, Dict[str, float]], optional
        Probability matrix from a generative model.

    Returns
    -------
    str
        The mutated sequence.
    """
    sequence_list = list(sequence)

    mutable_positions = _get_mutable_positions(sequence_list, aa_vocabulary)

    if not mutable_positions:
        raise ValueError("No mutable positions found in the sequence.")

    # Determine how many positions to mutate
    num_mutations = int(mutation_rate * len(mutable_positions))

    if num_mutations > 0:
        positions_to_mutate = rng.sample(mutable_positions, min(num_mutations, len(mutable_positions)))

        # Apply mutations to selected positions
        for i in positions_to_mutate:
            key = i if i in aa_vocabulary else sequence_list[i]
            current_aa = sequence_list[i]
            # Find amino acids different from the current one
            alternative_aas = [aa for aa in aa_vocabulary[key] if aa != current_aa]
            if alternative_aas:  # Only mutate if alternatives exist
                prob_matrix_row = probability_matrix[i] if probability_matrix is not None else None
                if prob_matrix_row is not None:
                    # Filter probabilities for alternative amino acids
                    # Note: One could consider to more gracefully handle missing AAs in prob. matrix.
                    # Here we just raise an error.
                    prob_matrix_row = [prob_matrix_row[aa] for aa in alternative_aas]

                    # Re-normalize via softmax
                    prob_matrix_row = np.array(prob_matrix_row) / sum(prob_matrix_row)
                    assert np.isclose(
                        sum(prob_matrix_row), 1.0, atol=2e-2
                    ), "Sum of prob. matrix row must be 1 after normalization."
                sequence_list[i] = rng.choices(alternative_aas, weights=prob_matrix_row, k=1)[0]

    return "".join(sequence_list)


def mutate_batch(
    batch: list[str],
    mutation_rate: float,
    aa_vocabularies: list[dict[int, str]],
    rng: random.Random,
    probability_matrix: dict[int, dict[str, float]] | None = None,
) -> list[str]:
    """
    Perform mutation on a batch of sequences.

    Parameters
    ----------
    batch : list[str]
        The batch of sequences to mutate.
    mutation_rate : float
        Probability of each amino acid being mutated.
    aa_vocabularies : list[dict[int , str]]
        Alphabet of possible amino acids for mutation for each sequence in the batch.
    rng : random.Random
        Random number generator for reproducibility.
    probability_matrix: Dict[int, Dict[str, float]], optional
        Probability matrix from a generative model.

    Returns
    -------
    list[str]
        The mutated batch of sequences.
    """
    return [
        mutate_sequence(seq, mutation_rate, vocab, rng, probability_matrix)
        for seq, vocab in zip(batch, aa_vocabularies)
    ]


def _get_mutable_positions(sequence_list: list[str], aa_vocabulary: dict[str | int, str]) -> list[int]:
    """
    Determine which positions can be mutated.

    This is just for efficiency, since the vocabulary contains positions with only the parental AA,
    which cannot be mutated. Sampling these would be inefficient, so we first determine the mutable positions
    where more than one AA is allowed.

    Args:
        sequence_list : list of str
            The sequence to mutate as list of AAs.
        aa_vocabulary : dict[int | str, str]
            Alphabet of possible amino acids for mutation, either position or AA-based.

    Returns
    -------
    list of int
        List of mutable positions.
    """
    mutable_positions = []
    for i in range(len(sequence_list)):
        key = i if i in aa_vocabulary else sequence_list[i]  # position or amino acid
        if key in aa_vocabulary and len(aa_vocabulary[key]) > 1:
            mutable_positions.append(i)

    return mutable_positions
