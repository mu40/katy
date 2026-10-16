"""Module of metrics, which may not be differentiable."""

import katy as kt
import os
import torch


def dice(x, y, /, labels):
    """Compute hard Dice scores (https://www.jstor.org/stable/1932409).

    Parameters
    ----------
    x : (B, 1, *size) torch.Tensor
        Discrete label map.
    y : (B, 1, *size) torch.Tensor
        Discrete label map.
    labels : os.PathLike or sequence of int
        Label values to include.

    Returns
    -------
    (B, C) torch.Tensor
        Dice scores, where `C` is the number of labels.

    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    if x.shape != y.shape:
        raise ValueError(f'sizes {x.shape} and {y.shape} differ')

    if isinstance(labels, (str, os.PathLike)):
        labels = kt.io.load(labels)
    if isinstance(labels, dict):
        labels = list(map(int, labels))
    labels = torch.as_tensor(labels).ravel()

    # One-hot, one channel per label.
    x = torch.cat([x.eq(i) for i in labels], dim=1).flatten(start_dim=2)
    y = torch.cat([y.eq(i) for i in labels], dim=1).flatten(start_dim=2)

    # Numerator, denominator.
    top = 2 * (x & y).sum(-1)
    bot = x.sum(-1) + y.sum(-1)

    # Avoid division by zero for all-zero inputs.
    return top / bot.clamp(min=1)


def ncc(x, y, /, width=3, eps=1e-6):
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
