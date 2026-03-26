"""Test cases for the Tanimoto kernel implementation in the bayesopt.models.kernels module."""
import pytest
import torch

from boat.bayesopt.models.kernels import TanimotoKernel


def manual_tanimoto_kernel(x1, x2, eps=1e-6):
    """Compute the Tanimoto kernel manually for two sets of binary vectors."""
    dot_prod = torch.matmul(x1, x2.t())
    x1_norm = torch.sum(x1**2, dim=-1, keepdim=True)
    x2_norm = torch.sum(x2**2, dim=-1, keepdim=True)
    sim = (dot_prod + eps) / (x1_norm + x2_norm.t() - dot_prod + eps)
    return sim.clamp_min(0)


def test_diag_forward():
    """Test the forward method with diag=True."""
    # Create a simple tensor and verify that diag=True returns a tensor of ones.
    x = torch.rand(5, 3)
    kernel = TanimotoKernel()
    result = kernel.forward(x, x, diag=True)
    expected = torch.ones(5, 5, dtype=x.dtype, device=x.device)
    assert torch.allclose(result, expected), "Expected ones when diag=True"


def test_covariance_single():
    """Test the covariance computation on single inputs."""
    # Test the covariance computation on non-batch (single) inputs.
    x1 = torch.randint(0, 2, (4, 6)).float()
    x2 = torch.randint(0, 2, (5, 6)).float()
    kernel = TanimotoKernel()
    result = kernel.forward(x1, x2, diag=False)
    expected = manual_tanimoto_kernel(x1, x2)
    assert result.shape == (4, 5)
    assert torch.allclose(result, expected, atol=1e-5)


def test_covariance_symmetric():
    """Test that the covariance matrix is symmetric for identical inputs."""
    # For identical inputs, the computed covariance matrix should be symmetric.
    x = torch.randint(0, 2, (7, 10)).float()
    kernel = TanimotoKernel()
    result = kernel.forward(x, x, diag=False)
    assert torch.allclose(result, result.t(), atol=1e-5), "Covariance matrix should be symmetric"


def test_covariance_batch():
    """Test the covariance computation with batched inputs."""
    # Test the kernel with batched inputs.
    batch_x = torch.randint(0, 2, (3, 8, 5)).float()
    kernel = TanimotoKernel()
    result = kernel.forward(batch_x, batch_x, diag=False)
    # Expected shape: [batch, n, n]
    assert result.shape == (3, 8, 8)
    # Check symmetry for each batch element.
    for i in range(3):
        assert torch.allclose(result[i], result[i].t(), atol=1e-5), f"Batch {i} covariance not symmetric"


def test_invalid_input_dimension():
    """Test that an error is raised for inputs with less than 2 dimensions."""
    # Test that _batch_sim raises an error when inputs don't have at least 2 dimensions.
    kernel = TanimotoKernel()
    with pytest.raises(ValueError):
        x1 = torch.tensor([1.0, 2.0])
        x2 = torch.tensor([1.0, 2.0])
        kernel._batch_sim(x1, x2)
