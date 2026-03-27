"""BoTorch-ready models for the various kernels, including an initializer."""

from typing import Any

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from boat.bayesopt.models.kernels import TanimotoKernel


class TanimotoGPBinary(ApproximateGP, GPyTorchModel):
    """Binary gaussian Process model using a Tanimoto kernel."""

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, **model_kwargs: Any) -> None:
        """Initialize a binary GP model.

        Args:
            x_train: training inputs (encoded)
            y_train: training outputs
            model_kwargs: Other parameters that might be passed to a SingleTaskGP
        """
        self.train_inputs = (x_train,)
        self.train_targets = y_train

        variational_distribution = CholeskyVariationalDistribution(x_train.size(0))
        variational_strategy = VariationalStrategy(self, x_train, variational_distribution)
        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.likelihood = BernoulliLikelihood()

        self.to(x_train)
        self.to(y_train)

    def __repr__(self) -> str:
        """Return string representation of the GP model."""
        return "TanimotoGPBinary"

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Evaluate GP model at input locations x.

        Args:
            x: Tensor with inputs for prediction

        Returns
        -------
            Multivariate normal distribution of model predictions
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TanimotoGP(SingleTaskGP):
    """Gaussian Process model using a Tanimoto kernel.

    Refer to https://botorch.org/api/models.html#botorch.models.gp_regression.SingleTaskGP for detailed reference.
    """

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor, **model_kwargs: Any) -> None:
        """Initialize the GP model.

        Args:
            x_train: training inputs (encoded)
            y_train: training outputs
            model_kwargs: Other parameters that might be passed to a SingleTaskGP
        """
        super().__init__(x_train, y_train, likelihood=GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(x_train)

    def __repr__(self) -> str:
        """Return string representation of the GP model."""
        return "TanimotoGP"

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Evaluate GP model at input locations x.

        Args:
            x: Tensor with inputs for prediction

        Returns
        -------
            Multivariate normal distribution of model predictions
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
