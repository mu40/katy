"""Module of metrics, which may not be differentiable."""


import torch
import katy as kt


def dice(true, pred, labels=None):
    """Compute hard Dice scores (https://www.jstor.org/stable/1932409).

    Parameters
    ----------
    true : (B, C, *size) torch.Tensor
        One-hot or single-channel index labels.
    pred : (B, C, *size) torch.Tensor
        The same as `true`.
    labels : int, optional
        Number of labels. Required for index labels. None for one-hot maps.

    Returns
    -------
    (B, C) torch.Tensor
        Dice scores.

    """
    true = torch.as_tensor(true)
    pred = torch.as_tensor(pred)
    if labels is None and true.size(1) == 1:
        raise ValueError('single-channel one-hot maps are likely a bug')
    if true.shape != pred.shape:
        raise ValueError(f'sizes {true.shape} and {pred.shape} differ')

    # Convert probabilities to index labels. Destroys gradients.
    if labels is None:
        labels = true.size(1)
        true = true.argmax(dim=1, keepdim=True)
        pred = pred.argmax(dim=1, keepdim=True)

    # One-hot, creating a new channel dimension.
    true = kt.labels.one_hot(true, labels).flatten(start_dim=2)
    pred = kt.labels.one_hot(pred, labels).flatten(start_dim=2)

    # Nominator, denominator.
    top = 2 * (true * pred).sum(-1)
    bot = true.sum(-1) + pred.sum(-1)

    # Avoid dividion by zero for all-zero inputs.
    return top / bot.clamp(min=1e-6)
