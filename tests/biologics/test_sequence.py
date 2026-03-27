"""Tests for the sequence module in the boat.biologics package."""
import pytest

from boat.biologics.sequence import (
    CDR,
    AbVSeq,
    cdr_to_seq,
)


def test_cdr_to_seq_with_light_chain():
    """Test the cdr_to_seq function with a light chain."""
    # Test input
    parent = AbVSeq(
        heavy_chain="QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCARYYDDHYCLDYWGQGTTLTVSSAKTTAPSVYPLA",
        light_chain="DIQMTQSPSSLSASVGDRVTITCQQG",
    )
    cdr = CDR(id="L1", pos=(5, 9), sequence="YYYYY")

    # Call function
    result = cdr_to_seq(cdr, parent)

    # Assertions
    assert result.light_chain[: cdr.pos[0]] == parent.light_chain[:5]
    assert result.light_chain[cdr.pos[0] : cdr.pos[1] + 1] == "YYYYY"
    assert result.light_chain[cdr.pos[1] + 1 :] == parent.light_chain[10:]
    assert len(result.light_chain) == len(parent.light_chain)
    assert result.heavy_chain == parent.heavy_chain


def test_cdr_list_to_seq():
    """Test the cdr_to_seq function when input is a list of CDRs."""
    # Test input
    parent = AbVSeq(
        heavy_chain="QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCARYYDDHYCLDYWGQGTTLTVSSAKTTAPSVYPLA",
        light_chain="DIQMTQSPSSLSASVGDRVTITCQQG",
    )
    cdr1 = CDR(id="H1", pos=(11, 15), sequence="YYYYY")
    cdr2 = CDR(id="L1", pos=(5, 9), sequence="ZZZZZ")

    # Call function
    result = cdr_to_seq([cdr1, cdr2], parent)

    # Assertions
    assert result.heavy_chain[: cdr1.pos[0]] == parent.heavy_chain[: cdr1.pos[0]]
    assert result.heavy_chain[cdr1.pos[0] : cdr1.pos[1] + 1] == "YYYYY"
    assert result.heavy_chain[cdr1.pos[1] + 1 :] == parent.heavy_chain[cdr1.pos[1] + 1 :]
    assert len(result.heavy_chain) == len(parent.heavy_chain)

    assert result.light_chain[: cdr2.pos[0]] == parent.light_chain[: cdr2.pos[0]]
    assert result.light_chain[cdr2.pos[0] : cdr2.pos[1] + 1] == "ZZZZZ"
    assert result.light_chain[cdr2.pos[1] + 1 :] == parent.light_chain[cdr2.pos[1] + 1 :]
    assert len(result.light_chain) == len(parent.light_chain)


def test_cdr_to_seq_invalid_cdr():
    """Test the cdr_to_seq function with an invalid CDR ID."""
    # Test input
    parent = AbVSeq(
        heavy_chain="QVQLQQSGAELARPGASVKMSCKASGYTFTRYTMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCARYYDDHYCLDYWGQGTTLTVSSAKTTAPSVYPLA",
        light_chain="DIQMTQSPSSLSASVGDRVTITCQQG",
    )
    cdr = CDR(id="X1", pos=(5, 9), sequence="YYYYY")

    # Call function and expect ValueError
    with pytest.raises(ValueError):
        cdr_to_seq(cdr, parent)
