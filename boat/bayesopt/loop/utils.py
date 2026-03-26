"""Utilities for multi-objective optimization."""

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume


def compute_hypervolume(pareto_front: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Compute the hypervolume indicator for the current Pareto front.

    Returns
    -------
        float: The hypervolume value
    """
    # Make sure we have a non-empty Pareto front
    if len(pareto_front) == 0:
        return 0.0

    hv = Hypervolume(ref_point=ref_point)
    return hv.compute(pareto_front)


def compute_pareto_front(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Pareto front from a set of scores.

    Args:
        scores: Tensor of shape (n_points, n_objectives)

    Returns
    -------
        Tuple of (pareto_front, pareto_indices) where:
            pareto_front is a Tensor of shape (n_pareto, n_objectives)
            pareto_indices is a Tensor of indices of the Pareto-optimal points
    """
    # Use maximisation criteria
    is_efficient = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)

    for i in range(scores.shape[0]):
        if is_efficient[i]:
            # Compare current point to all others
            # A point dominates another if it's >= in all objectives and > in at least one
            dominates = torch.all(scores[i] >= scores, dim=1) & torch.any(scores[i] > scores, dim=1)

            # The current point doesn't dominate itself, so reset that
            dominates[i] = False

            # Mark dominated points as not efficient
            is_efficient = is_efficient & ~dominates

    # Get the Pareto front and corresponding indices
    pareto_indices = torch.where(is_efficient)[0]
    pareto_front = scores[pareto_indices]

    return pareto_front, pareto_indices
