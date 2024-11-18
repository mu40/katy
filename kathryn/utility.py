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
