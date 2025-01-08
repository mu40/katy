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


def ncc(x, y, width=3, eps=1e-6):
    """Compute (local) normalized cross correlation.

    Actually computes squared NCC, for differentiability.

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        Input.
    y : (B, C, *size) torch.Tensor
        Input.
    width : int, optional
        Window size.
    eps : float, optional
        Epsilon, to avoid dividing by zero.

    Returns
    -------
    (B, C) torch.Tensor
        Mean NCC.

    """
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    if x.shape != y.shape:
        raise ValueError(f'shapes {x.shape} and {y.shape} differ')

    # Average over patches via convolution.
    ndim = x.ndim - 2
    size = (x.size(1), 1, *[width] * ndim)
    mean = torch.ones(size, device=x.device)
    mean = mean / torch.tensor(size[2:]).prod()

    # Separate convolutions, with as many groups as channels.
    conv = getattr(torch.nn.functional, f'conv{ndim}d')
    prop = dict(weight=mean, groups=x.size(1), padding='same')

    # Remove mean.
    x = x - conv(x, **prop)
    y = y - conv(y, **prop)

    # Variance.
    var_x = conv(x * x, **prop).clamp(min=eps)
    var_y = conv(y * y, **prop).clamp(min=eps)

    # Cross correlation.
    cc = conv(x * y, **prop).clamp(min=eps)
    cc = cc * cc / var_x / var_y
    return cc.flatten(2).mean(-1)
