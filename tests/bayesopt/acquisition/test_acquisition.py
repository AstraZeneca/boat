"""Test cases for the AcquisitionFunctionOnSequences class in the bayesopt module."""


import pytest
import torch

from boat.bayesopt.acquisition.acquisition import AcquisitionFunctionOnSequences
from boat.bayesopt.acquisition.utils import get_acquisition
from boat.bayesopt.encodings.encodings import OneHotEncoding


class DummyAcquisitionFunction:
    """A dummy acquisition function class for testing purposes."""

    def __init__(self, return_tensor):
        """Initialize a dummy acquisition function that returns a predefined tensor."""
        self.return_tensor = return_tensor

    def forward(self, x, *args, **kwargs):
        """Forward a tensor."""
        # Simply return the predefined tensor
        return self.return_tensor


@pytest.fixture
def encoding():
    """Fixture to provide a OneHotEncoding instance for testing."""
    vocab = {0: "AD", 1: "BE", 2: "CF"}
    return OneHotEncoding(vocab=vocab)


def test_acquisition_function_on_sequences_returns_expected_value(encoding):
    """Test that the AcquisitionFunctionOnSequences returns the expected tensor."""
    sequences = ["ABC", "DEF"]
    # Define what the acquisition function should return (simulate computation).
    expected_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    dummy_acquisition = DummyAcquisitionFunction(expected_tensor)
    af = AcquisitionFunctionOnSequences(dummy_acquisition, encoding, "dummy_acq")

    output = af(sequences)
    torch.testing.assert_close(output, expected_tensor)


# Test acquisition utils
@pytest.mark.parametrize(
    "acq_str, expected_class_name",
    [
        ("EI", "LogExpectedImprovement"),
        ("EHVI", "ExpectedHypervolumeImprovement"),
    ],
)
def test_get_acquisition_valid(acq_str, expected_class_name):
    """Test that get_acquisition returns the correct acquisition class."""
    acq_class = get_acquisition(acq_str)
    assert (
        expected_class_name in acq_class.__name__
    ), f"Expected acquisition class to contain '{expected_class_name}', got '{acq_class.__name__}'"


def test_get_acquisition_invalid():
    """Test that get_acquisition raises an error for an invalid acquisition string."""
    invalid_acq_str = "INVALID"
    with pytest.raises(NotImplementedError) as exc_info:
        get_acquisition(invalid_acq_str)
    assert f"The acquisition function {invalid_acq_str} is not implemented" in str(exc_info.value)
