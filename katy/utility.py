"""Utility module."""


import torch


def chance(prob, size=1, device=None, generator=None):
    """Return True with given probability.

    Parameters
    ----------
    prob : float
        Probability of returning True, in the range [0, 1].
    size : torch.Size
        Output shape.
    device : torch.device, optional
        Device of the returned tensor.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    torch.Tensor
        True with probability `prob`.

    """
    if prob < 0 or prob > 1:
        raise ValueError(f'probability {prob} is not in range [0, 1]')

    size = torch.as_tensor(size).ravel()
    return torch.rand(*size, device=device, generator=generator) < prob


def resize(x, size, fill=0):
    """Symmetrically crop or pad an N-dimensional tensor to a new size.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Tensor of `N`-element spatial shape.
    size : int or sequence of int
        New spatial shape. Pass 1 or N values.
    fill : float, optional
        Fill value for padding.

    Returns
    -------
    (B, C, *size) torch.Tensor
        Resized tensor.

    """
    # Dimensions. Keep batch and channel dimensions.
    ndim = x.ndim - 2
    size_old = torch.as_tensor(x.shape)
    size_new = torch.as_tensor(size).ravel().expand(ndim)
    size_new = torch.cat((size_old[:2], size_new))

    # Indexing. Non-zero starting indices into the input are equivalent to
    # cropping, and those into the output are equivalent to padding.
    overlap = torch.minimum(size_old, size_new)
    a_old = size_old.sub(size_new).clamp(min=0) // 2
    a_new = size_new.sub(size_old).clamp(min=0) // 2
    ind_old = tuple(map(slice, a_old, a_old + overlap))
    ind_new = tuple(map(slice, a_new, a_new + overlap))

    out = torch.full(size_new.tolist(), fill, device=x.device)
    out[ind_new] = x[ind_old]
    return out
