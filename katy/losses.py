"""Differentiable losses."""


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
    # Flatten spatial dimensions.
    true = true.flatten(start_dim=2)
    pred = pred.flatten(start_dim=2)

    # Nominator, denominator.
    top = 2 * (true * pred).sum(-1)
    bot = (true * true).sum(-1) + (pred * pred).sum(-1)

    # Avoid dividion by zero for all-zero inputs.
    dice = top / bot.clamp(min=1e-6)
    return 1 - dice.mean()
