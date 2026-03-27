"""Test Gaussian Process model utilities."""
import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from boat.bayesopt.models.gps import TanimotoGP
from boat.bayesopt.models.utils import fit_gp, get_model, initialize_mll, initialize_model


def test_get_model():
    """Test retrieving the model class by its string identifier."""
    model_class = get_model("TanimotoGP")
    assert model_class == TanimotoGP

    with pytest.raises(NotImplementedError):
        get_model("NonExistentModel")


def test_initialize_model():
    """Test the model initialization."""
    x_train = torch.randn(10, 3)
    y_train = torch.randn(10, 1)
    model = initialize_model(TanimotoGP, x_train, y_train)
    assert isinstance(model, TanimotoGP)
    print(model.train_inputs[0].shape)
    assert model.train_inputs[0].shape == (10, 3)
    assert model.train_targets.shape == (10,)


def test_initialize_mll():
    """Test the marginal log likelihood initialization."""
    x_train = torch.randn(10, 3)
    y_train = torch.randn(10, 1)
    model = initialize_model(TanimotoGP, x_train, y_train)
    mll = initialize_mll(model)
    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.likelihood.noise_covar.noise.shape == (1,)


def test_fit_gp():
    """Test fitting the Gaussian Process model."""
    x_train = torch.randn(10, 3)
    y_train = torch.randn(10, 1)
    model = initialize_model(TanimotoGP, x_train, y_train)
    mll = initialize_mll(model)

    try:
        fit_gp(mll)
    except Exception as e:
        pytest.fail(f"Model fitting failed with error: {e}")

    assert model.train_inputs[0].shape == (10, 3)
    assert model.train_targets.shape == (10,)
