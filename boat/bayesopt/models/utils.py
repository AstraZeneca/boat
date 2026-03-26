"""Contains a class to manage models and training."""

from typing import Type

import gpytorch
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood, VariationalELBO

from .gps import TanimotoGP, TanimotoGPBinary


def get_model(model_str: str) -> Type[Model]:
    """
    Get the raw model class from its string identifier.

    Args:
        model_str: String identifier of the encoding

    Returns
    -------
        a raw model class
    """
    model_dict = {
        "TanimotoGP": TanimotoGP,
        "TanimotoGPBinary": TanimotoGPBinary,
    }

    # checks
    if model_str not in model_dict:
        raise NotImplementedError(
            f"The model {model_str} is not implemented. Available models: {list(model_dict.keys())}"
        )

    return model_dict[model_str]


def initialize_model(
    model: Type[Model], x_train: torch.Tensor, y_train: torch.Tensor, binary: bool = False, **kwargs
) -> Model:
    """
    Set up the model from its string identifier.

    Args:
        model: A raw botorch model class
        x_train: training inputs (encoded)
        y_train: training outputs
        kwargs: other parameters that might be passed to a GP.

    Returns
    -------
        an instance of a GP model.
    """
    if binary:
        # Ensure float targets in {0,1} and 1D
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.view(-1)
        elif y_train.ndim != 1:
            raise ValueError(f"Binary y_train must be shape [N] or [N,1], got {y_train.shape}")
        model = model(x_train=x_train, y_train=y_train, **kwargs)
    else:
        # Regression: keep 2D shape [N,1]
        if y_train.ndim == 1:
            y_train = y_train.view(-1, 1)
        model = model(x_train=x_train, y_train=y_train, outcome_transform=Standardize(m=1), **kwargs)

    # checks
    if x_train.device != y_train.device:
        raise ValueError("x_train and y_train must be on the same device.")

    device = x_train.device
    model = model.to(device)
    return model


def initialize_mll(
    model: Model, fix_noise: bool = True, binary: bool = False, num_data: int = None
) -> MarginalLogLikelihood:
    """Initialize the log marginal likelihood."""
    if fix_noise:
        model.likelihood.noise = 1e-4
        model.likelihood.requires_grad_(False)

    if binary:
        mll = VariationalELBO(model.likelihood, model, num_data=num_data)
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return mll


def fit_gp(mll: MarginalLogLikelihood):
    """
    Fit the gp model.

    Args:
        mll: Marginal log likelihood to fit
    """
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        try:
            fit_gpytorch_mll(mll)
        except ModelFittingError as e:
            print(f"Model fitting failed: {e}.")
