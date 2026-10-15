"""Detached scalar statistics for training diagnostics."""

import math

import torch


@torch.no_grad()
def diagnostic_quantile(values: torch.Tensor, q: float) -> float:
    """Linear quantile over all elements, with zero for an empty diagnostic.

    torch.quantile rejects inputs larger than 2**24 elements. For those
    inputs select the two neighboring order statistics directly, preserving
    the quantile definition without subsampling or averaging chunk quantiles.
    """
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be finite and in [0, 1]")
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    if flat.numel() <= 2**24:
        return torch.quantile(flat, q).item()
    # Match quantile's NaN propagation; kthvalue alone does not guarantee it.
    if torch.isnan(flat).any().item():
        return float("nan")
    rank = q * (flat.numel() - 1)
    lower = math.floor(rank)
    fraction = rank - lower
    lo = flat.kthvalue(lower + 1).values
    if fraction == 0:
        return lo.item()
    hi = flat.kthvalue(lower + 2).values
    return torch.lerp(lo, hi, fraction).item()
