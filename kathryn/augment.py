"""Augmentation module."""


import torch
import kathryn as kt


def gamma(x, gamma=0.5, prob=1, shared=False, gen=None):
    """Apply a random gamma transform to the intensities of a tensor.

    See https://en.wikipedia.org/wiki/Gamma_correction.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    gamma : float, optional
        Value in (0, 1), leading to exponents in [1 - gamma, 1 + gamma].
    prob : float, optional
        Probability of the transform.
    shared : bool, optional
        Transform all channels the same way.
    gen : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Transformed tensor, normalized into [0, 1].

    """
    if not 0 < gamma < 1:
        raise ValueError(f'gamma {gamma} is not in range (0, 1)')

    # Need intensities in [0, 1] range.
    x = torch.as_tensor(x)
    dim = tuple(range(1 if shared else 2, x.ndim))
    x = x - x.amin(dim, keepdim=True)
    x = x / x.amax(dim, keepdim=True)

    # Randomize across batches, or batches and channels.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1

    # Exponents.
    a = 1 - gamma
    b = 1 + gamma
    exp = torch.rand(*size, device=x.device, generator=gen) * (b - a) + a

    # Randomize application.
    bit = kt.utility.chance(prob, size, device=x.device, gen=gen)
    exp = bit * exp + ~bit * 1

    return x.pow(exp)
