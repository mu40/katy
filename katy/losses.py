"""Differentiable losses."""

import katy as kt


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
