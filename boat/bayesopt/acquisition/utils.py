"""Utility functions for acquisition functions in Bayesian optimization."""

from typing import Type

from botorch.acquisition import AcquisitionFunction, LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)


def get_acquisition(
    acq_str: str,
) -> Type[AcquisitionFunction]:
    """
    Set up acquition function based on the string identifier.

    Args:
        acq_str: String identifier of the acquisition function

    Returns
    -------
        A class of the acquisition function
    """
    acq_dict = {
        "EI": LogExpectedImprovement,
        "EHVI": ExpectedHypervolumeImprovement,
        "qEI": qLogExpectedImprovement,
        "qNEI": qLogNoisyExpectedImprovement,
        "qEHVI": qExpectedHypervolumeImprovement,
        "qNEHVI": qNoisyExpectedHypervolumeImprovement,
    }
    if acq_str not in acq_dict:
        raise NotImplementedError(
            f"The acquisition function {acq_str} is not implemented. Available functions: {list(acq_dict.keys())}"
        )

    return acq_dict[acq_str]
