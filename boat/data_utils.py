"""Utilities for reading and writing data that is generated."""

import os
import sys
from typing import Dict, List

import pandas as pd
import yaml

from boat.biologics.sequence import CDR, AbVSeq
from boat.genetic_algorithm.vocabularies import AA_VOCABULARY


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file and resolve any relative paths."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_mutations_from_yaml(file_path: str) -> Dict[int, str]:
    """
    Load mutations from a YAML file.

    The file is supposed to list the amino acid single letter for each of
    the positions mutated. Uses 0-based counting.

    Args:
        file_path (str): Path to the YAML file containing mutations.

    Returns
    -------
        dict: A dictionary containing the mutations.
    """
    with open(file_path, "r") as file:
        mutations = yaml.safe_load(file)

    # Convert list of AAs to a single string for each position
    for pos, muts in mutations.items():
        mutations[pos] = "".join(muts)

    return mutations


def extract_cdr_positions(mutations: Dict):
    """
    Extract CDR positions from mutations dictionary.

    Assumes that only the CDRs have been mutated and clusters the positions into three blocks,
    separating by the largest distance between the positions. Assumes there is one chain with
    three CDRs (H1, H2, H3 or L1, L2, L3).
    Uses the same counting as the mutations dictionary, which is 0-based.

    Args:
        mutations (dict): Dictionary containing mutations.

    Returns
    -------
        list: List of CDR positions.
    """
    positions = [pos for pos in list(mutations.keys())]  # 0-based index
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    largest_indices = sorted(range(1, len(gaps) + 1), key=lambda i: gaps[i - 1], reverse=False)[-2:]

    cdr_positions = []
    start = 0

    for idx in largest_indices:
        cdr_positions.append(positions[start:idx])
        start = idx
    cdr_positions.append(positions[start:])

    return cdr_positions


def load_cdr_positions(path: str | None) -> Dict[str, List[int]]:
    """Load CDR positions from a YAML file."""
    cdr_positions = {}
    if path:
        cdr_positions = load_yaml_config(path)

    return cdr_positions


def cdrs_from_positions(parental: AbVSeq, positions: List[List[int]], is_heavy=True) -> Dict[str, CDR]:
    """
    Create CDR objects from positions.

    Args:
        parental (AbVSeq): Parent antibody variable region sequence.
        positions (list): List of CDR positions on one chain. Assumes there are three of them.
        is_heavy (bool): If True, create CDRs for the heavy chain, otherwise for the light chain.

    Returns
    -------
        list: List of CDR objects.
    """
    cdrs = {}
    for i, pos in enumerate(positions):
        cdr_id = f"H{i+1}" if is_heavy else f"L{i+1}"
        cdr_seq = parental.heavy_chain[pos[0] : pos[-1] + 1] if is_heavy else parental.light_chain[pos[0] : pos[-1] + 1]
        cdrs[cdr_id] = CDR(id=cdr_id, pos=(pos[0], pos[-1]), sequence=cdr_seq)
    return cdrs


def read_in_fasta(file_path: str) -> AbVSeq:
    """Read and parse FASTA file.

    reads in fasta file and returns dictionary.

    Args:
        file: fasta file

    Returns
    -------
        AbVSeq: An AbVSeq object containing the heavy and light chain sequences.
    """
    seq = {}

    if os.path.exists(file_path):
        with open(file_path, "r") as fasta:
            for line in fasta.readlines():
                line = line.rstrip("\n")
                if line.startswith(">"):
                    chain_id = line[1:]
                else:
                    seq[chain_id] = line
    else:
        sys.exit("cannot find fasta files: " + file_path)

    # transform to AbVSeq object, assuming the last letter of the chain ID is 'H' for heavy and 'L' for light
    heavy_chain = ""
    light_chain = ""
    for chain_id, sequence in seq.items():
        if chain_id.endswith("H"):
            heavy_chain = sequence
        elif chain_id.endswith("L"):
            light_chain = sequence
    if not heavy_chain or not light_chain:
        sys.exit(
            f"FASTA file '{file_path}' must contain both heavy ('*H') and light ('*L') chains."
        )

    return AbVSeq(heavy_chain=heavy_chain, light_chain=light_chain)


def library_to_mutations_yaml(
    library_file: str, parental_file: str, out_file: str, chain: str = "heavy"
) -> Dict[int, List[str]]:
    """
    Extract the mutations per position and chain w.r.t. a parental from a library csv file and saves them as a YAML.

    Saves a dictionary  of mutations where keys are positions (0-based) and values are lists of amino acids
    mutated at that position.

    Args:
        library_file (str): Path to the csv input file containing sequences.
        parental_file (str): Path to the parental sequence, assuming a CSV similar to the library.
        out_file (str): Path to the output YAML file where mutations will be saved.
        chain (str): The chain to read, either "heavy" or "light". Defaults to "heavy".
    columns (list): List of columns to read from the CSV file. Defaults to ["sequence", "chain"].
        Assumes the CSV has "heavy_chain" or "light_chain" as the column names for sequences.
    """
    if not os.path.exists(library_file):
        raise FileNotFoundError(f"CSV file not found: {library_file}")

    col_key = f"{chain}_chain"

    try:
        df_lib = pd.read_csv(library_file, usecols=[col_key])
        df_prnt = pd.read_csv(parental_file, usecols=[col_key])
    except ValueError:
        raise ValueError(f"CSV files must contain a column named '{chain}_chain'.")

    parental_sequence = df_prnt[col_key].iloc[0]

    mutations = {i: [] for i in range(len(parental_sequence))}  # Initialize with all positions

    for _, row in df_lib.iterrows():
        sequence = row[col_key]
        for pos, aa in enumerate(sequence):
            if aa not in mutations[pos] and aa != parental_sequence[pos]:
                mutations[pos].append(aa)

    # Delete the positions that have remained unchanged
    mutations = {pos: muts for pos, muts in mutations.items() if len(muts) > 0}

    # Save the mutations to a YAML file
    with open(out_file, "w") as file:
        yaml.dump(mutations, file)

    return mutations


def get_mutations_dict(sequences: list, parental: str) -> dict:
    """Get the mutations from a list of sequences, 0-based counting.

    Args:
        sequences (list): List of sequences to compare against the parental sequence.
        parental (str): The parental sequence to compare against.

    Returns
    -------
        dict: A dictionary where keys are positions (0-based) and values are lists of mutated
        amino acids (single-character strings) found at those positions.
    """
    mutations_dict = {}
    parent_seq = parental
    for seq in sequences:
        for i, (p_aa, s_aa) in enumerate(zip(parent_seq, seq)):
            if p_aa != s_aa:
                if i not in mutations_dict:
                    mutations_dict[i] = []
                mutations_dict[i].append(s_aa)
    mutations_dict = {i: set(v) for i, v in mutations_dict.items()}  # Convert lists to sets to remove duplicates
    # Ensure the dictionary is ordered by keys
    mutations_dict = dict(sorted(mutations_dict.items()))
    return mutations_dict


def load_probability_matrix(path: str | None):
    """Load probability matrix from generative AI prior."""
    probability_matrix = None
    if path:
        probability_matrix = load_yaml_config(path)

        # Sanity checks
        assert len(probability_matrix) > 0, f"Probability matrix is empty from location {path}."
        for pos, vocab in probability_matrix.items():
            assert len(vocab) >= len(AA_VOCABULARY), f"Not full vocab at position {pos}."

    return probability_matrix
