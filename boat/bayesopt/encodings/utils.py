"""Utility functions for encodings.

The methods build_one_hot, encode_string and pad are used to create one-hot encodings and taken from
https://github.com/leojklarner/gauche/blob/main/gauche/kernels/string_kernels/sskkernel.py
"""

from itertools import chain
from typing import Dict, Optional, Tuple

import torch


def build_one_hot(alphabet: list) -> Tuple[torch.Tensor, dict]:
    """
    Build one-hot encodings for a given alphabet.

    Args:
        alphabet: unique alphabet/characters possible in the string

    Returns
    -------
        embs: one-hot embeddings of the alphabet
        index: integer index for the alphabet
    """
    dim = len(alphabet)
    embs = torch.zeros((dim + 1, dim), dtype=torch.double)
    index = {}
    for i, symbol in enumerate(alphabet):
        embs[i + 1, i] = 1.0
        index[symbol] = i + 1
    return embs, index


def encode_string(s: str, index: dict[str, int]) -> list:
    """
    Transform a string in a list of integers.

    The ints correspond to indices in an
    embeddings matrix.

    Args:
        s: input string
        index: integer index for the alphabet

    Returns: integer encoding for the string s
    """
    return [index[symbol] for symbol in s]


def pad(s, length):
    """
    Pad out input strings to a maximum length.

    (required to pass same length tensors to gpytorch modules)

    Args:
        s: input string
        length: max length including zero-padding

    Returns: padded string with zeros
    """
    new_s = torch.zeros(length, dtype=torch.double)
    new_s[: len(s)] = torch.tensor(s)
    return new_s


def vocab_to_list(vocab: Dict[str | int, str], linker: Optional[str] = "") -> str:
    """
    Convert a vocabulary to a list representation.

    Args:
        vocab: list of characters

    Returns: list representation of the vocabulary
    """
    if isinstance(vocab, dict):
        vocab = sorted(set(chain.from_iterable(vocab.values())))
    elif isinstance(vocab, str):
        vocab = list(vocab)

    return vocab + sorted(set(linker)) if linker else vocab
