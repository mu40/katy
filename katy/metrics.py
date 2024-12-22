"""Module of metrics, which may not be differentiable."""


import torch
import katy as kt


def dice(true, pred, labels=None):
    """Compute hard Dice scores (https://www.jstor.org/stable/1932409).

    Parameters
    ----------
    true : (B, C, *size) torch.Tensor
        One-hot or discrete-valued label map.
    pred : (B, C, *size) torch.Tensor
        One-hot or discrete-valued label map.
    labels : sequence of int, optional
        Label values or one-hot indices. Not required for one-hot maps.

    Returns
    -------
    (B, C) torch.Tensor
        Dice scores.

    """
    true = torch.as_tensor(true)
    pred = torch.as_tensor(pred)
    if labels is None and true.size(1) == 1:
        raise ValueError('labels required except for one-hot maps')
    if true.shape != pred.shape:
        raise ValueError(f'sizes {true.shape} and {pred.shape} differ')

    # Convert probabilities to index labels. Destroys gradients.
    if true.size(1) > 1:
        if labels is None:
            labels = range(true.size(1))

        true = true.argmax(dim=1, keepdim=True)
        pred = pred.argmax(dim=1, keepdim=True)

    # Label selection.
    true = true.to(torch.int64)
    pred = pred.to(torch.int64)
    labels = torch.as_tensor(labels, device=true.device)
    highest = max(true.max(), pred.max())
    highest = max(highest, labels.max())

    lut = torch.full(size=[highest + 1], fill_value=-1, device=true.device)
    depth = labels.numel()
    lut[labels] = torch.arange(depth, device=true.device)

    # One-hot, creating a new channel dimension.
    true = lut[true]
    pred = lut[pred]
    true = kt.labels.one_hot(true, depth).flatten(start_dim=2)
    pred = kt.labels.one_hot(pred, depth).flatten(start_dim=2)

    # Nominator, denominator.
    top = 2 * (true * pred).sum(-1)
    bot = true.sum(-1) + pred.sum(-1)

    # Avoid dividion by zero for all-zero inputs.
    return top / bot.clamp(min=1e-6)
