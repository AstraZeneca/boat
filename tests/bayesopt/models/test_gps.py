"""Test Gaussian Process models."""
import torch
from gpytorch.distributions import MultivariateNormal

from boat.bayesopt.models.gps import TanimotoGP
from boat.bayesopt.models.utils import fit_gp, initialize_mll


def test_repr():
    """Test the string representation of the TanimotoGP model."""
    x_train = torch.randn(10, 3)
    y_train = torch.randn(10, 1)
    model = TanimotoGP(x_train, y_train)
    assert repr(model) == "TanimotoGP"


def test_forward_returns_multivariate_normal():
    """Test that the forward method returns a MultivariateNormal distribution."""
    x_train = torch.randn(10, 3)
    y_train = torch.randn(10, 1)
    model = TanimotoGP(x_train, y_train)
    mll = initialize_mll(model)
    fit_gp(mll)
    x_test = torch.randn(5, 3)
    output = model(x_test)
    assert isinstance(output, MultivariateNormal)


def test_forward_output_shape():
    """Test that the output shape of the forward method is correct."""
    x_train = torch.randn(20, 5)
    y_train = torch.randn(20, 1)
    model = TanimotoGP(x_train, y_train)
    mll = initialize_mll(model)
    fit_gp(mll)

    model.eval()
    x_test = torch.randn(8, 5)
    output = model(x_test)
    # The output mean should have one entry per test point.
    assert output.mean.shape[0] == 8
