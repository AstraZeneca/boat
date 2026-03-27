"""
Functions to construct vocabularies for the genetic algorithm.

We distinguish between different types of vocabularies:

- AA-based vocabularies: These are dictionaries that map each amino acid to a string of possible AAs for mutation.
  AA-based vocabularies available are:
    - `aa_vocabulary_complete`: Contains all amino acids for each AA.
    - `aa_vocabulary_reduced`: Contains all amino acids except those specified in `exclude_aas`,
      e.g. to exclude cysteines.
    - `aa_vocabulary_blosum`: Contains amino acids with positive similarity scores according to the BLOSUM matrix.

- position-based vocabularies: These are dictionaries that map each position in a reference sequence to a string
  of possible mutations. This is useful if we are given position-specific mutation data that we want the GA to
  constrain to.

All methods require position-based vocabularies, so if an AA-based vocabulary is given, use
`aa_to_positional_vocabulary` to convert it to a position-based vocabulary for the reference sequence of interest.
"""

import itertools

import blosum
import numpy as np

AA_VOCABULARY = "ACDEFGHIKLMNPQRSTVWY"


def aa_vocabulary_complete() -> dict[str, str]:
    """
    Create a vocabulary of all amino acids.

    The vocabulary maps each amino acid to a dictionary that maps each amino acid
    to all amino acids in the AA_VOCABULARY.

    Returns
    -------
        dict: A dictionary with the amino acids as keys and amino acids as values.
    """
    return {aa: AA_VOCABULARY for aa in AA_VOCABULARY}


def aa_vocabulary_reduced(exclude_aas: str = "CM") -> dict[str, str]:
    """
    Create a vocabulary of amino acids excluding specified ones.

    The vocabulary maps each amino acid to a list containing all amino acids,
    except those specified in the `exclude_aas` parameter.
    The excluded ones map only to themselves.

    Args:
        exclude_aas (str): A string of amino acids to exclude from the vocabulary.
            Default is "CM" to exclude cysteine and methionine from being mutated.

    Returns
    -------
        dict: A dictionary with the amino acids as keys and remaining amino acids as values.
    """
    reduced_vocab = "".join([a for a in AA_VOCABULARY if a not in exclude_aas])
    vocab = {aa: reduced_vocab if aa not in exclude_aas else aa for aa in AA_VOCABULARY}
    return vocab


def aa_vocabulary_blosum(similarity=62, level=0, exclude_aas: str = "BJZX*") -> dict[str, str]:
    """
    Create a vocabulary based on BLOSUM.

    For each amino acid in the alphabet, the vocabulary contains a list of amino acids
    that have a positive similarity score with respect to the BLOSUM matrix.

    Args:
        similarity (int): The BLOSUM similarity score to use (default is 62).
        level (int): The BLOSUM level of similarity to use (defaults to 0).
        exclude_aas (str): String of amino acids to exclude from the vocabulary (default is "BJZX*").

    Returns
    -------
        dict: A dictionary where keys are amino acids and values are strings of amino acids with positive similarity.
    """
    blosum_matrix = blosum.BLOSUM(similarity, default=0)

    aa_vocab = [aa for aa in blosum_matrix.keys() if aa not in exclude_aas]

    blosum_vocab = {}
    for aa1 in aa_vocab:
        blosum_vocab[aa1] = "".join([aa2 for aa2 in aa_vocab if blosum_matrix[aa1][aa2] > level])

    return blosum_vocab


def positional_vocabulary(sequence: str, mutations: dict[int, str]) -> dict[int, str]:
    """
    Create a positional vocabulary with respect to a full sequence and a mutations dictionary.

    The vocabulary consists of listed mutations and the AA in the parental.
    0-based indexing is used for both mutations and vocabulary.

    Args:
        sequence (str): A complete chain to create the vocabulary for.
        mutations (dict): Dictionary containing mutations without the parental AAs

    Returns
    -------
        dict: A dictionary with positions as keys and possible mutations as values.
    """
    vocab = {}
    for pos in range(len(sequence)):
        unique_aas = set(mutations.get(pos, "") + sequence[pos])
        vocab[pos] = "".join(sorted(unique_aas))
    return vocab


def aa_to_positional_vocabulary(sequence: str, aa_vocab: dict[str, list[str]]) -> dict[int, list[str]]:
    """
    Transform a per-AA-vocabulary to a positional vocabulary for a specific sequence.

    The input vocabulary lists possible mutations for each amino acid, the output vocabulary lists possible mutations
    for each position in the sequence. 0-based indexing is used.

    Args:
        sequence (str): the reference sequence for the vocabulary
        aa_vocab (Dict[str, list[str]]): AA-based vocabulary.

    Returns
    -------
        dict[int, str]: A dictionary with positions as keys and possible mutations as values.
    """
    vocab = {}
    for pos in range(len(sequence)):
        aa = sequence[pos]
        vocab[pos] = aa_vocab.get(aa, aa)
    return vocab


def count_mutation_permutations(sequence: str, aa_vocab: dict[int | str, str], n_muts: int) -> int:
    """
    Count the number of possible mutations in a sequence given a vocabulary.

    Args:
        sequence (str): The reference sequence to count mutations.
        vocab (dict): Dictionary containing the vocabulary.
        n_muts (list[int]): Number of mutations to permit.

    Returns
    -------
        dict: A dictionary with positions as keys and number of possible mutations as values.
    """
    if all([isinstance(key, str) for key in aa_vocab.keys()]):
        # If the vocabulary is AA-based, convert it to positional vocabulary
        aa_vocab = aa_to_positional_vocabulary(sequence, aa_vocab)
    combs = itertools.combinations(range(len(sequence)), n_muts)  # Combinations of positions to mutate
    return int(sum([np.prod(np.asarray([len(aa_vocab[i]) - 1 for i in comb])) for comb in combs]))
