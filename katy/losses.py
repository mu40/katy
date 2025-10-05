"""Differentiable losses."""

import katy as kt
import torch


def dice(x, y, /):
    """Compute a soft-dice loss (https://arxiv.org/abs/1606.04797).

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        Tensor of probabilities.
    y : (B, C, *size) torch.Tensor
        Tensor of probabilities.

    Returns
    -------
    torch.Tensor
        Mean Dice loss.

    """
    # Broadcasting would work but will likely be a bug.
    if x.shape != y.shape:
        raise ValueError(f'shapes {x.shape} and {y.shape} differ')

    # Flatten spatial dimensions.
    x = x.flatten(start_dim=2)
    y = y.flatten(start_dim=2)

    # Nominator, denominator.
    top = 2 * (x * y).sum(-1)
    bot = (x * x).sum(-1) + (y * y).sum(-1)

    # Avoid division by zero for all-zero inputs.
    dice = top / bot.clamp(min=1e-6)
    return 1 - dice.mean()


def ncc(*args, **kwargs):
    """Compute a (local) normalized cross correlation loss.

    See `katy.metrics.ncc` for details.

    Returns
    -------
    torch.Tensor
        Negated mean NCC.

    """
    return kt.metrics.ncc(*args, **kwargs).mean().mul(-1)


def axial(f, /, norm=1):
    """Compute an axial diffusion regularization loss.

    Identical to VoxelMorph's `Grad` loss, which does not actually compute the
    gradient. It only considers the diagonal of the Jacobian, penalizing
    contraction and expansion along each spatial axis independently.

    Parameters
    ----------
    f : (B, N, *size) torch.Tensor
        Displacement vector field of spatial N-element `size`.
    norm : {1, 2}, optional
        Order of the norm, `1` for L1 and `2` for L2 regularization.

    Returns
    -------
    torch.Tensor
        Mean partial diffusion loss.

    """
    f = torch.as_tensor(f, dtype=torch.get_default_dtype())
    ndim, *size = f.shape[1:]
    if f.ndim != ndim + 2:
        raise ValueError(f'field does not have {ndim} spatial axes')
    if norm not in (1, 2):
        raise ValueError(f'order of the norm {norm} is not 1 or 2')

    f = [torch.gradient(f[:, i], dim=i + 1)[0] for i in range(ndim)]
    return torch.stack(f).abs().pow(norm).mean()
