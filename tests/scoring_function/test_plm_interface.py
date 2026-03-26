"""Tests for plm scoring functions."""
import os

import numpy as np
import pytest

from boat.biologics.sequence import AbVSeq
from boat.scoring_function.plm_interface import ESMInterface


@pytest.fixture
def checkpoint_path():
    """Path to esm checkpoint."""
    al_shared_vol_path = "/home/jovyan/shared-mlab-active-learning/esm_checkpoints/esm2_t6_8M_UR50D.pt"
    dev_shared_vol_path = "/home/jovyan/iron-mlab-developability/llm_models/ESM/esm2_t6_8M_UR50D.pt"
    if os.path.isfile(al_shared_vol_path):
        return al_shared_vol_path
    elif os.path.isfile(dev_shared_vol_path):
        return dev_shared_vol_path
    else:
        raise ValueError("No model checkpoint available.")


@pytest.fixture
def parental_heavy_str():
    """Parental heavy chain."""
    return "EVQLVESGGGLVQPGGSLRLSCAASG"


@pytest.fixture
def parental_light_str():
    """Parental light chain."""
    return "DIVMTQSPDSLAVSLGERAT"


def test_esm_get_input_sequences_both_chains(checkpoint_path):
    """Test esm input processing."""
    abvseqs = [AbVSeq(heavy_chain="AAA", light_chain="BBB")]
    parental = AbVSeq(heavy_chain="AAV", light_chain="BDB")
    esminterface = ESMInterface(parental, checkpoint_path=checkpoint_path, device="cpu")

    heavy, light = esminterface._get_input_sequences(abvseqs)
    assert len(heavy) == 1
    assert len(light) == 1
    assert heavy[0][1] == "AAA"
    assert light[0][1] == "BBB"


def test_esm_call(parental_heavy_str: str, parental_light_str: str, checkpoint_path: str):
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
