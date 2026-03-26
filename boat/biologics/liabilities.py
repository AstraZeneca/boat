"""
Score sequence liabilities.

Enables liability filtering or scoring sequences according to liability.
"""

import re

MPROC_SLEEP_TIME = 3

# single residue liabilites
LIABILITIES_SINGLE = {"C": 15, "M": 5.4, "W": 5.4}

# multi residue liabilites
LIABILITIES_NGRAM = {
    "NG": 15,
    "DS": 9,
    "DG": 9,
    "NS": 9,
    "NN": 9,
    "DK": 1.8,
    "DT": 1.8,
    "RGD": 0.6,
    "KGD": 0.6,
    "RYD": 0.6,
    "NGR": 0.6,
    "LDV": 0.6,
    "DGE": 0.6,
    "GPR": 0.6,
    "DP": 0.6,
}

# glycolisation liabilities
# avoid pattern "N*[S|T]" where * can be any AAs except PRO
LIABILITIES_GLYCO = ["N" + t for t in "ARNDCQEGHILKMFSTWYV"]

GLYCOLISATION_PAYLOAD = 15.0


def single_residue_liabilities(sequence: str) -> float:
    """Return the liability score associated with the frequency of risky tokens.

    Parameters
    ----------
    sequence : str
        sequence representing heavy or light chain (or complete sequence)

    Returns
    -------
    float
        liability score for the sequence
    """
    return sum([w * sequence.count(t) for t, w in LIABILITIES_SINGLE.items()])


def ngram_liabilities(sequence: str) -> float:
    """Liability score associated with n-grams that can occur anywhere in the sequence.

    Parameters
    ----------
    sequence : str
        sequence representing heavy or light chain (or complete sequence)

    Returns
    -------
    float
        liability score for the sequence
    """
    return sum([w * sequence.count(t) for t, w in LIABILITIES_NGRAM.items()])


def glycolisation_liabilities(sequence: str) -> float:
    """Liability score associated with glycolisation.

    i.e., existence of patterns "N*[S|T]" where * can be any AAs except PRO.

    Parameters
    ----------
    sequence : str
        sequence representing heavy or light chain (or complete sequence)

    Returns
    -------
    float
        liability score for the sequence
    """
    return sum(GLYCOLISATION_PAYLOAD * sequence.count(f"{glyc}{t}") for glyc in LIABILITIES_GLYCO for t in ["S", "T"])


def score_sequence(sequence: str) -> float:
    """Compute liability score for the sequence using rules supplied by CSB.

    Parameters
    ----------
    sequence : str
            sequence representing heavy or light chain (or complete sequence)

    Returns
    -------
    float
            liability score for the sequence
    """
    sc1 = single_residue_liabilities(sequence)
    sc2 = ngram_liabilities(sequence)
    sc3 = glycolisation_liabilities(sequence)
    return sc1 + sc2 + sc3


def score_sequence_list(sequence_list: list[str]) -> list[float]:
    """
    Score a list of sequences.

    Args:
        sequence_list: list of sequences

    Returns
    -------
    list[float]
        Liability scores for the sequences
    """
    return [score_sequence(seq) for seq in sequence_list]


def filter_by_liability(sequence_list: list[str], aa_threshold: float = 1.0) -> list[str]:
    """
    Filter sequences by liability below a threshold.

    Sequences with higher values than the threshold are considered risky and are removed.

    Args:
        sequence_list: list of sequences to filter
        aa_threshold: Threshold for liability score per amino acid, default is 1.0

    Returns
    -------
    list[str]
        Filtered list of sequences below the threshold
    """
    return [seq for seq in sequence_list if score_sequence(seq) < aa_threshold * len(seq)]


def count_glycosylation_sites_inside(sequence):
    """
    Return a number of potential glycosylation sites in query sequence.

    Such sites are defined as NX(S/T).
    Method taken from InSiDe.
    """
    matches = re.findall("N[^P][ST]", sequence)
    return len(matches)
