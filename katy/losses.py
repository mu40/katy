"""Differentiable losses."""


import katy as kt


def dice(true, pred):
    """Compute a soft-dice loss (https://arxiv.org/abs/1606.04797).

    Parameters
    ----------
    true : (B, C, *size) torch.Tensor
        Tensor of probabilities.
    pred : (B, C, *size) torch.Tensor
        Tensor of probabilities.

    Returns
    -------
    torch.Tensor
        Mean Dice loss.

    """
    # Broadcasting would work but will likely be a bug.
    if true.shape != pred.shape:
        raise ValueError(f'shapes {true.shape} and {pred.shape} differ')

    # Flatten spatial dimensions.
    true = true.flatten(start_dim=2)
    pred = pred.flatten(start_dim=2)

    # Nominator, denominator.
    top = 2 * (true * pred).sum(-1)
    bot = (true * true).sum(-1) + (pred * pred).sum(-1)

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
