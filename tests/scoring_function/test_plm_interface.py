"""Tests for PLM scoring functions."""

import numpy as np
import pytest
import torch

from boat.biologics.sequence import AbVSeq

pytest.importorskip("esm")

from boat.scoring_function.plm_interface import ESMInterface


class DummyAlphabet:
    """Small fake ESM alphabet for deterministic tests."""

    def __init__(self):
        self.tok_to_idx = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        for i, token in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=3):
            self.tok_to_idx[token] = i

    def get_batch_converter(self):
        """Return a fake batch converter matching ESM's signature."""
        pad_idx = self.tok_to_idx["<pad>"]
        bos_idx = self.tok_to_idx["<bos>"]
        eos_idx = self.tok_to_idx["<eos>"]

        def _batch_converter(batch):
            labels = [label for label, _ in batch]
            sequences = [sequence for _, sequence in batch]
            max_len = max(len(seq) for seq in sequences)
            batch_tokens = torch.full((len(batch), max_len + 2), pad_idx, dtype=torch.long)

            for i, sequence in enumerate(sequences):
                batch_tokens[i, 0] = bos_idx
                for j, aa in enumerate(sequence):
                    batch_tokens[i, j + 1] = self.tok_to_idx[aa]
                batch_tokens[i, len(sequence) + 1] = eos_idx

            return labels, sequences, batch_tokens

        return _batch_converter


class DummyModel(torch.nn.Module):
    """Fake ESM model that returns deterministic logits."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, batch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits with high score at the observed token index."""
        batch_size, seq_len = batch_tokens.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -5.0)
        logits.scatter_(2, batch_tokens.unsqueeze(-1), 5.0)
        return {"logits": logits}


@pytest.fixture
def checkpoint_path():
    """Dummy checkpoint path used for mocked ESM loading."""
    return "dummy_esm_checkpoint.pt"


@pytest.fixture
def mock_esm_loader(monkeypatch):
    """Mock the ESM loader so tests do not require real checkpoint files."""

    def _load_model_and_alphabet(_checkpoint_path):
        alphabet = DummyAlphabet()
        model = DummyModel(vocab_size=max(alphabet.tok_to_idx.values()) + 1)
        return model, alphabet

    monkeypatch.setattr("boat.scoring_function.plm_interface.load_model_and_alphabet", _load_model_and_alphabet)


@pytest.fixture
def parental_heavy_str():
    """Parental heavy chain."""
    return "EVQLVESGGGLVQPGGSLRLSCAASG"


@pytest.fixture
def parental_light_str():
    """Parental light chain."""
    return "DIVMTQSPDSLAVSLGERAT"


def test_esm_get_input_sequences_both_chains(checkpoint_path, mock_esm_loader):
    """Test esm input processing."""
    abvseqs = [AbVSeq(heavy_chain="AAA", light_chain="BBB")]
    parental = AbVSeq(heavy_chain="AAV", light_chain="BDB")
    esminterface = ESMInterface(parental, checkpoint_path=checkpoint_path, device="cpu")

    heavy, light = esminterface._get_input_sequences(abvseqs)
    assert len(heavy) == 1
    assert len(light) == 1
    assert heavy[0][1] == "AAA"
    assert light[0][1] == "BBB"


def test_esm_call(parental_heavy_str: str, parental_light_str: str, checkpoint_path: str, mock_esm_loader):
    """Test call for multiple scenarios."""
    heavy_chains = ["EVQLVESGGGLVQPGGSLLLSCAASG", "EVQLVESGGGLVQPGGSLRLSCAAAG", "EVQLVESGGGLVQPGGSLRLSCAAAG"]
    light_chains = ["DIVMTQSPDSLAVSLGERAT", "DIVMTQSPDSLAVSLEERAT", "DIVMTQSPDSLAVSLEERAT"]
    scfv = [heavy + light for heavy, light in zip(heavy_chains, light_chains)]

    # scFv scores
    parental_scfv = AbVSeq(heavy_chain=parental_heavy_str, light_chain=parental_light_str)
    esm_interface_scfv = ESMInterface(parental=parental_scfv, checkpoint_path=checkpoint_path, device="cpu")
    scores_scfv = esm_interface_scfv(scfv)
    scores_scfv = list(scores_scfv.values())[0]

    # Heavy only
    parental_heavy = AbVSeq(heavy_chain=parental_heavy_str, light_chain="")
    esm_interface_heavy = ESMInterface(parental=parental_heavy, checkpoint_path=checkpoint_path, device="cpu")
    scores_heavy = esm_interface_heavy(heavy_chains)
    scores_heavy = list(scores_heavy.values())[0]

    # Light only
    parental_light = AbVSeq(heavy_chain="", light_chain=parental_light_str)
    esm_interface_light = ESMInterface(parental=parental_light, checkpoint_path=checkpoint_path, device="cpu")
    scores_light = esm_interface_light(light_chains)
    scores_light = list(scores_light.values())[0]

    assert len(scores_scfv) == len(scores_heavy) == len(scores_light) == 3
    assert np.all([isinstance(score, float) for score in scores_scfv])
    assert scores_scfv[-1] == scores_scfv[-2]
    assert np.isclose(
        [(heavy + light) / 2 for heavy, light in zip(scores_heavy, scores_light)], scores_scfv, atol=1e-5
    ).all()
