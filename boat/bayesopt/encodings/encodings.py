"""Encodings for use with proteins."""

import abc
from typing import Dict, List, Optional, Type

import ablang2
import blosum
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from boat.biologics.sequence import CDR, AbVSeq, extract_sequence_from_abvseq_or_cdr

from .utils import build_one_hot, encode_string, pad, vocab_to_list


class Encoding(abc.ABC):
    """Abstract base class for encodings."""

    def __init__(self, dd_dict):
        """Initialize encoding.

        Args:
            dd_dict: device and dtype in dict
        """
        self.device = dd_dict["device"]
        self.dtype = dd_dict["dtype"]

    @abc.abstractmethod
    def __call__(self, sequences: List[AbVSeq | CDR | str]) -> torch.Tensor:
        """Encode sequences.

        Args:
            sequences: list of AbVSeq, CDR or str sequences
        Returns
        -------
            encoded sequences as torch Tensor
        """
        raise NotImplementedError

    def _extract_sequences(self, sequences: List[AbVSeq | CDR | str], linker: str) -> np.ndarray:
        """Extract sequences from a list of AbVSeq or CDR objects.

        Args:
            sequences: list of AbVSeq, CDR or str sequences

        Returns
        -------
            numpy array of sequences
        """
        if all([isinstance(seq, str) for seq in sequences]):
            return np.array(sequences)
        return np.array([extract_sequence_from_abvseq_or_cdr(seq, linker) for seq in sequences])


class OneHotEncoding(Encoding):
    """One-hot encoding of sequences."""

    def __init__(
        self,
        vocab: Dict[str | int, str],
        linker: Optional[str] = "",
        dd_dict: Optional[Dict] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
        **kwargs
    ):
        """Construct one-hot encoding from vocabulary.

        Args:
            vocab: AA vocabulary as a dictionary w.r.t a position in the sequence or to an amino acid
            dd_dict: device and dtype in dict
            kwargs: keyword arguments, just to make the method work when getting unexpected keyword arguments
        """
        super().__init__(dd_dict)
        self.vocab = vocab_to_list(vocab, linker)
        self.linker = linker

        # construct embedding from vocabulary
        self.embds, self.index_dict = build_one_hot(self.vocab)
        self.embds = self.embds.to(dtype=self.dtype, device=self.device)

    def __str__(self):
        """Return string identifier of encoding."""
        return "one-hot"

    def __call__(self, sequences: List[AbVSeq | CDR | str]) -> torch.Tensor:
        """One-hot encoding of sequences.

        Encodes sequences with a one-hot encoding based on the vocabulary.

        Args:
            sequences: list of AbVSeq, CDR or str sequences

        Returns
        -------
            one-hot encoded sequences as torch.Tensor
        """
        sequences = self._extract_sequences(sequences, self.linker)
        maxlen = np.max([len(seq) for seq in sequences])
        classes = (
            torch.cat(
                [pad(encode_string(seq, self.index_dict), maxlen).unsqueeze(0) for seq in sequences],
                dim=0,
            )
            - 1
        )
        x_oh = torch.nn.functional.one_hot(classes.to(torch.int64), num_classes=len(self.vocab))
        return x_oh.to(dtype=self.dtype, device=self.device).reshape(len(x_oh), -1)


class BagOfAAsEncoding(Encoding):
    """Count encoding of sequences using max_ngram-size bag of words."""

    def __init__(
        self,
        max_ngram: Optional[int] = 5,
        vocab: Optional[Dict[str | int, str]] = None,
        linker: Optional[str] = "",
        dd_dict: Optional[Dict] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
        **kwargs
    ):
        """Initialize the bag of AAs encoding.

        Args:
            max_ngram: the max size of a word in the bag-of-words encoding
            encoding_features: optional features to use when encoding
            dd_dict: device and dtype in dict
            kwargs: keyword arguments, just to make the method work when getting unexpected keyword arguments
        """
        super().__init__(dd_dict)

        self.vocab = vocab_to_list(vocab, linker)
        self.linker = linker
        print("Using vocabulary", self.vocab)

        self.max_ngram = max_ngram
        self.count_vectorizer = CountVectorizer(
            ngram_range=(1, self.max_ngram), analyzer="char", lowercase=False, vocabulary=self.vocab
        )

    def __str__(self):
        """Return string identifier of encoding."""
        return "bag_of_aas"

    def __call__(self, sequences: List[AbVSeq | CDR | str]) -> torch.Tensor:
        """Encode sequences with a bag of AAs encoding.

        Args:
            sequences: list of AbVSeq, CDR or str sequences
        Returns
        -------
            encoded sequences as torch.Tensor
        """
        sequences = self._extract_sequences(sequences, self.linker)
        return torch.tensor(self.count_vectorizer.fit_transform(sequences).toarray()).to(
            dtype=self.dtype, device=self.device
        )

    def get_features(self) -> np.ndarray:
        """Extract the features used for the bag of AAs encoding."""
        return self.count_vectorizer.get_feature_names_out()


class BlosumEncoding(Encoding):
    """Use BLOSUM to obtain sequence embedding.

    This encoding does an SVD of the BLOSUM matrix and uses the decomposiion for encoding sequences.
    Credits to Dino Oglic and Eugen Buehler.
    """

    def __init__(
        self,
        vocab: Dict[str | int, str],
        linker: Optional[str] = "",
        n_blosum: Optional[int] = 45,
        dd_dict: Optional[Dict] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
        **kwargs
    ):
        """Initialize BLOSUM encoding.

        Args:
            vocab: vocabulary
            n_blosum: integer from {45,50,62,80,90}, defaults to 45
            dd_dict: device and dtype in dict
            kwargs: keyword arguments, just to make the method work when getting unexpected keyword arguments
        """
        super().__init__(dd_dict)
        self.n_blosum = n_blosum

        # append linker to the vocabulary
        self.linker = linker

        vocab = vocab_to_list(vocab, linker)
        blosum_vals = blosum.BLOSUM(self.n_blosum, default=0)
        vocab_length = len(vocab)
        blosum_matrix = np.zeros((vocab_length, vocab_length))

        # Populate BLOSUM matrix
        for i, v1 in enumerate(vocab):
            for j, v2 in enumerate(vocab):
                blosum_matrix[i, j] = blosum_vals[v1][v2]
                blosum_matrix[j, i] = blosum_vals[v2][v1]

        # Our vocab has a separator symbol. Unknown characters result in zeros (as default=0 above).
        # Set the diagonal entries to 1.
        idx = np.diag(blosum_matrix) == 0
        blosum_matrix[idx, idx] = 1

        # Do singular value decomposition to get encoding
        u, s, _ = np.linalg.svd(blosum_matrix, full_matrices=True)
        aa_emb = u * s**0.5

        # initialize embedding dictionary
        self.aa_emb_dict = {}

        for i, v in enumerate(vocab):
            self.aa_emb_dict[v] = aa_emb[i, :]

    def __str__(self):
        """Return string identifier of encoding."""
        return "blosum"

    def __call__(self, sequences: List[AbVSeq | CDR | str]) -> torch.Tensor:
        """Encode sequences with BLOSUM.

        Args:
            sequences: list of AbVSeq, CDR or str sequences

        Returns
        -------
            BLOSUM encoded sequences as torch.Tensor
        """
        sequences = self._extract_sequences(sequences, self.linker)
        blosum_emb = np.vstack([np.asarray([self.aa_emb_dict[v] for v in seq]).flatten() for seq in sequences])
        return torch.tensor(blosum_emb).to(dtype=self.dtype, device=self.device)


class Ablang2Encoding(Encoding):
    """Class for encoding antibody sequences using the antibody language model AbLang-2.

    Ablang-2 is a language model specifically for antibodies. Details here:
        https://github.com/oxpig/AbLang2
    """

    def __init__(
        self,
        linker: Optional[str] = "",
        dd_dict: Optional[Dict] = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        },
        **kwargs
    ):
        """Initialize class by downloading the model.

        Args:
            linker: the separator symbol between VL and VH in the string representation
            dd_dict: device and dtype in dict
            kwargs: keyword arguments, just to make the method work when getting unexpected keyword arguments
        """
        super().__init__(dd_dict)
        self.linker = linker
        self.ablang2 = ablang2.pretrained(device=self.device)  # load ablang-2 model

    def __str__(self):
        """Return string identifier of encoding."""
        return "ablang2"

    @torch.inference_mode()
    def __call__(self, sequences: List[AbVSeq | CDR | str]) -> torch.Tensor:
        """Encode antibody sequences using Ablang-2.

        Args:
            sequences: list of AbVSeq, CDR or str sequences

        Returns
        -------
            Sequence encodings as torch.Tensor of size [number of sequences, length of encoding]
        """
        # Get sequences in the format required by ablang-2
        sequence_list = self._format_sequences(sequences)
        with torch.no_grad():
            seqs_enc = self.ablang2(sequence_list, mode="seqcoding")
        return torch.tensor(seqs_enc).to(dtype=self.dtype, device=self.device)

    def _format_sequences(self, sequences: List[AbVSeq | CDR | str]) -> List[List[str]]:
        """Format the input sequences to be compatible with Ablang-2.

        Ablang-2 accepts sequences as a list of strings as [VH(str), VL(str)].
        This method takes a list of AbVSeq or CDR, and arranges them in the required list.

        Args:
            sequences: list of AbVSeq, CDR or str sequences

        Returns
        -------
            List of lists of sequences arranged as [VH, VL]
        """
        if all(isinstance(seq, str) for seq in sequences):
            if self.linker:
                return [[seq.split(self.linker)[0], seq.split(self.linker)[1]] for seq in sequences]
            else:
                return [[seq, ""] for seq in sequences]

        # Handle AbVSeq or CDR objects
        try:
            sequences = [[seq.heavy_chain, seq.light_chain] for seq in sequences]
        except AttributeError:
            # If sequences are CDRs, we assume they are already in the correct format
            sequences = [[seq.sequence, ""] if seq.id[0] == "H" else ["", seq.sequence] for seq in sequences]

        return sequences


def get_encoding(encoding_str: str) -> Type[Encoding]:
    """Set up encoding from the string identifier.

    Args:
        encoding_str: String identifier of the encoding
        dd_dict: device and dtype in dict
        kwargs: keyword arguments relevant for some encodings, e.g. vocab, fingerprint_method

    Returns
    -------
        an encoding class
    """
    encoding_dict = {
        "one-hot": OneHotEncoding,
        "bag_of_aas": BagOfAAsEncoding,
        "blosum": BlosumEncoding,
        "ablang2": Ablang2Encoding,
    }
    return encoding_dict[encoding_str]
