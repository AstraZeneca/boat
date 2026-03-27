"""Antibody sequence utilities."""

import warnings
from dataclasses import dataclass


@dataclass
class AbVSeq:
    """
    Amino acid sequence of an antibody variable region.

    Attributes
    ----------
        heavy_chain (str): the heavy chain sequence
        light_chain (str): the light chain sequence

    The sequences use the one-letter amino acid codes.
    Empty string ("") is used if the chain is missing.
    """

    heavy_chain: str = ""
    light_chain: str = ""


@dataclass
class CDR:
    """
    Complementarity-determining region (CDR) of an antibody variable region.

    Attributes
    ----------
        id (str): the label of the CDR (e.g., "H1", "L2", etc.)
        pos (tuple): start and end position of the CDR in the chain
        sequence (str): the amino acid sequence of the CDR

    The sequences use the one-letter amino acid codes.
    Empty string ("") is used if the chain is missing.
    """

    id: str
    pos: tuple
    sequence: str


def cdr_to_seq(cdrs: CDR | list[CDR], parent: AbVSeq) -> AbVSeq:
    """
    Convert a CDR or list of CDRs to a full sequence by reconstructing it from the parent sequence.

    This is useful for reconstructing full sequences from mutated CDRs

    Args:
        cdrs (CDR | list[CDR]): The CDRs to consider.
        parent (AbVSeq): The antibody variable region sequence.

    Returns
    -------
        mut_seq (AbVSeq): The full new sequence of the CDR.
    """
    if isinstance(cdrs, CDR):
        cdrs = [cdrs]

    antibody = AbVSeq(heavy_chain=parent.heavy_chain, light_chain=parent.light_chain)
    for cdr in cdrs:
        if cdr.id[0] == "H":
            antibody.heavy_chain = (
                antibody.heavy_chain[0 : cdr.pos[0]] + cdr.sequence + antibody.heavy_chain[cdr.pos[1] + 1 :]
            )
        elif cdr.id[0] == "L":
            antibody.light_chain = (
                antibody.light_chain[0 : cdr.pos[0]] + cdr.sequence + antibody.light_chain[cdr.pos[1] + 1 :]
            )
        else:
            raise ValueError("Invalid CDR ID")
    return antibody


def abvseq_to_str(abvseqs: AbVSeq | list[AbVSeq], linker="") -> list[str]:
    """
    Convert an AbVSeq object to a string representation.

    Args:
        abvseqs: AbVSeq | list[AbVSeq] The AbVSeq object(s) to convert.
        linker: Optional; a string to insert between the heavy and light chain sequences.

    Returns
    -------
        list[str]: The string representation of the AbVSeq object(s).
    """
    if isinstance(abvseqs, AbVSeq):
        abvseqs = [abvseqs]

    seqs = []
    for abvseq in abvseqs:
        if linker and (not abvseq.light_chain or not abvseq.heavy_chain):
            linker = ""
        seqs.append(abvseq.heavy_chain + linker + abvseq.light_chain)

    return seqs


def str_to_abvseq(sequences: list[str], ref_abvseq: AbVSeq | None = None, linker: str = "") -> list[AbVSeq]:
    """
    Convert a list of string sequences to AbVSeq objects.

    For reconstruction, either a reference AbVSeq object must be provided to determine chain lengths,
    or the sequences must contain the linker to split heavy and light chains.

    Args:
        sequences (list[str]): The list of sequences to convert.
        ref_abvseq: Optional; Reference AbVSeq object to determine chain lengths.
        linker: Optional; a string to insert between the heavy and light chain sequences.

    Returns
    -------
        list[AbVSeq]: The list of AbVSeq objects.
    """
    abvseqs = []
    for seq in sequences:
        if linker:
            if linker not in seq:
                raise ValueError("Linker not found in sequence.")
            heavy_chain, light_chain = seq.split(linker)

            if ref_abvseq is not None:
                if len(heavy_chain) != len(ref_abvseq.heavy_chain):
                    raise ValueError("Heavy chain length does not match reference.")
                if len(light_chain) != len(ref_abvseq.light_chain):
                    raise ValueError("Light chain length does not match reference.")

        else:
            if ref_abvseq is None:
                raise ValueError("Either ref_abvseq or linker must be provided to identify chains.")

            heavy_len = len(ref_abvseq.heavy_chain)
            light_len = len(ref_abvseq.light_chain)

            linker_len = len(seq) - (heavy_len + light_len)
            if linker_len != 0:
                warnings.warn(
                    "Linker not provided for splitting chains, reconstruction from reference AbVSeq assumes "
                    f"chains to be separated by a linker of length {linker_len}."
                )

            heavy_chain = seq[0:heavy_len]
            light_chain = seq[-light_len:]

        abvseqs.append(AbVSeq(heavy_chain=heavy_chain, light_chain=light_chain))

    return abvseqs


def extract_sequence_from_abvseq_or_cdr(abvseq_or_cdr: AbVSeq | CDR, linker: str = "") -> str:
    """
    Extract the sequence string from an AbVSeq or CDR object.

    Args:
        sequences (list[AbVSeq | CDR]): The object to extract the sequence from.
        linker: Optional; a string to insert between the heavy and light chain sequences.

    Returns
    -------
        str: The sequence string.
    """
    if isinstance(abvseq_or_cdr, AbVSeq):
        if (not abvseq_or_cdr.light_chain or not abvseq_or_cdr.heavy_chain) and len(linker) > 0:
            # If there is no light chain or heavy chain, there should not be a linker
            raise ValueError("Linker cannot be added if one of the chains is missing.")
        return abvseq_or_cdr.heavy_chain + linker + abvseq_or_cdr.light_chain
    elif isinstance(abvseq_or_cdr, CDR):
        return abvseq_or_cdr.sequence
    else:
        raise ValueError("Invalid input type")
