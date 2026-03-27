"""Test encodings in the bayesopt module."""
import pytest
import torch

from boat.bayesopt.encodings.encodings import (
    Ablang2Encoding,
    BagOfAAsEncoding,
    BlosumEncoding,
    OneHotEncoding,
    get_encoding,
)
from boat.bayesopt.encodings.utils import build_one_hot, encode_string, pad
from boat.biologics.sequence import AbVSeq

# --- Utilities Tests ---


def test_build_one_hot():
    """Test the build_one_hot function to ensure it creates a one-hot encoding matrix."""
    alphabet = ["A", "D", "E"]
    embs, index = build_one_hot(alphabet)
    # Expect shape: (len(alphabet)+1, len(alphabet))
    assert embs.shape == (len(alphabet) + 1, len(alphabet))
    # Check each letter has index i+1 and the corresponding row is one-hot
    for i, letter in enumerate(alphabet):
        assert index[letter] == i + 1
        expected = torch.zeros(len(alphabet), dtype=torch.double)
        expected[i] = 1.0
        assert torch.allclose(embs[i + 1], expected)


def test_encode_string():
    """Test the encode_string function to ensure it encodes a string using the index from build_one_hot."""
    alphabet = ["A", "D", "E", "G"]
    _, index = build_one_hot(alphabet)
    s = "ADG"
    encoding = encode_string(s, index)
    expected = [index["A"], index["D"], index["G"]]
    assert encoding == expected


def test_pad():
    """Test the pad function to ensure it pads a sequence to a specified length."""
    s = [1, 2, 3]
    length = 5
    padded = pad(s, length)
    expected = torch.tensor([1, 2, 3, 0, 0], dtype=torch.double)
    assert torch.allclose(padded, expected)


# Fixture for sample sequences.
@pytest.fixture
def sample_abvseq():
    """Fixture to provide a sample list of AbVSeq objects for testing."""
    return [AbVSeq(heavy_chain="ADE"), AbVSeq(heavy_chain="ADG")]


# --- OneHotEncoding Tests ---


def test_one_hot_encoding(sample_abvseq):
    """Test the OneHotEncoding class to ensure it encodes sequences correctly."""
    vocab = {0: "ADE", 1: "G", 2: "ADEG", 3: "AG"}
    device = torch.device("cpu")
    dd_dict = {"device": device, "dtype": torch.float64}
    one_hot_enc = OneHotEncoding(vocab, dd_dict=dd_dict)
    encoded = one_hot_enc(sample_abvseq)
    # Each sequence is padded to max length and reshaped to a flat vector.
    maxlen = max(len(x.heavy_chain) for x in sample_abvseq)
    expected_dim = maxlen * len(one_hot_enc.vocab)
    assert encoded.shape[0] == len(sample_abvseq)
    assert encoded.shape[1] == expected_dim


# --- BagOfAAsEncoding Tests ---


def test_bag_of_aas_encoding(sample_abvseq):
    """Test the BagOfAAsEncoding class to ensure it encodes sequences correctly."""
    vocab = {0: "ADE", 1: "G", 2: "ADEG", 3: "AG"}
    device = torch.device("cpu")
    dd_dict = {"device": device, "dtype": torch.float64}
    bag_enc = BagOfAAsEncoding(max_ngram=2, vocab=vocab, dd_dict=dd_dict)
    encoded = bag_enc(sample_abvseq)
    n_features = len(bag_enc.get_features())
    assert encoded.shape[0] == len(sample_abvseq)
    assert encoded.shape[1] == n_features


# --- BlosumEncoding Tests ---


def test_blosum_encoding(sample_abvseq):
    """Test the BlosumEncoding class to ensure it encodes sequences correctly."""
    vocab = "ADEG"
    device = torch.device("cpu")
    dd_dict = {"device": device, "dtype": torch.float64}
    blosum_enc = BlosumEncoding(vocab, n_blosum=45, dd_dict=dd_dict)
    encoded = blosum_enc(sample_abvseq)
    n_samples = len(sample_abvseq)
    embedding_size = list(blosum_enc.aa_emb_dict.values())[0].shape[0]
    maxlen = max(len(seq.heavy_chain) for seq in sample_abvseq)
    expected_dim = maxlen * embedding_size
    assert encoded.shape[0] == n_samples
    assert encoded.shape[1] == expected_dim


# --- Ablang2Encoding Tests ---


def test_ablang2_encoding(sample_abvseq):
    """Test the Ablang2Encoding class to ensure it encodes sequences correctly."""
    pytest.importorskip("ablang2")
    device = torch.device("cpu")
    dd_dict = {"device": device, "dtype": torch.float64}
    ablang_enc = Ablang2Encoding(dd_dict=dd_dict)
    encoded = ablang_enc(sample_abvseq)
    assert encoded.shape[0] == len(sample_abvseq)


# --- Factory Method Tests ---


def test_get_encoding():
    """Test the get_encoding factory method to ensure it returns the correct encoding class."""
    device = torch.device("cpu")
    dd_dict = {"device": device, "dtype": torch.float64}
    one_hot = get_encoding("one-hot")(dd_dict=dd_dict, vocab={0: "ABC", 1: "ABC", 2: "ABC"})
    assert isinstance(one_hot, OneHotEncoding)
    bag = get_encoding("bag_of_aas")(dd_dict=dd_dict, max_ngram=2, vocab={0: "ABC", 1: "ABC", 2: "ABC"})
    assert isinstance(bag, BagOfAAsEncoding)
    blo = get_encoding("blosum")(dd_dict=dd_dict, vocab={0: "ABC", 1: "ABC", 2: "ABC"}, n_blosum=45)
    assert isinstance(blo, BlosumEncoding)
    pytest.importorskip("ablang2")
    ablang = get_encoding("ablang2")(dd_dict=dd_dict)
    assert isinstance(ablang, Ablang2Encoding)
