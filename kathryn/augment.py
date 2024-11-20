"""Augmentation module."""


import torch
import kathryn as kt


def gamma(x, gamma=0.5, prob=1, shared=False, generator=None):
    """Apply a random gamma transform to the intensities of a tensor.

    See https://en.wikipedia.org/wiki/Gamma_correction.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    gamma : float, optional
        Value in (0, 1), leading to exponents in [1 - gamma, 1 + gamma].
    prob : float, optional
        Probability of transforming a channel.
    shared : bool, optional
        Transform all channels the same way.
    generator : torch.Generator, optional
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

    # Exponents. Randomize across batches, or batches and channels.
    prop = dict(device=x.device, generator=generator)
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    a = 1 - gamma
    b = 1 + gamma
    exp = torch.rand(*size, **prop) * (b - a) + a

    # Randomize application.
    bit = kt.utility.chance(prob, size, **prop)
    exp = bit * exp + ~bit * 1

    return x.pow(exp)


def noise(x, sd=0.1, prob=1, shared=False, generator=None):
    """Add Gaussian noise to a tensor.

    Uniformly samples the standard deviation (SD) of the noise.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    sd : float or sequence of float, optional
        SD range. Pass 1 value to define the upper bound, setting the lower
        bound to 0. Pass 2 values to define lower and upper bounds.
    prob : float, optional
        Probability of adding noise to a channel.
    shared : bool, optional
        Use the same SD for all channels.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Tensor with added noise.

    """
    # Inputs.
    x = torch.as_tensor(x)
    sd = torch.as_tensor(sd, device=x.device).ravel()
    prop = dict(device=x.device, generator=generator)

    # Standard deviation.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    a, b = (0, sd) if len(sd) == 1 else sd
    sd = torch.rand(*size, **prop) * (b - a) + a
    sd = sd * kt.utility.chance(prob, size, **prop)

    return x + sd * torch.randn(x.shape, **prop)


def blur(x, fwhm=1, prob=1, generator=None):
    """Blur a tensor by convolving its N spatial axes with Gaussian kernels.

    Uniformly samples anisotropic blurring full widths at half maximum (FWHM)
    between bounds `a` and `b`, blurring all channels of a batch the same way.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    fwhm : float or sequence of float, optional
        FWHM range. Pass 1 value to set the upper bound `b`, keeping the lower
        bound `a` at 0. Pass 2 values to set `(a, b)`. Pass `2 * N` values to
        set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes.
    prob : float, optional
        Probability of blurring a batch entry.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Blurred tensor.

    """
    # Inputs.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    fwhm = torch.as_tensor(fwhm, device=x.device).ravel()
    if len(fwhm) not in (1, 2, 2 * ndim):
        raise ValueError(f'FWHM {fwhm} is not of length 1, 2, or {2 * ndim}')

    # Conform FWHM bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    if len(fwhm) == 1:
        fwhm = torch.cat((torch.zeros_like(fwhm), fwhm))
    if len(fwhm) == 2:
        fwhm = fwhm.repeat(ndim)

    # FWHM sampling.
    prop = dict(device=x.device, generator=generator)
    a, b = fwhm[0::2], fwhm[1::2]
    batch = x.size(0)
    size = (batch, ndim)
    fwhm = torch.rand(size, **prop) * (b - a) + a

    # Smoothing at per-batch probability.
    bit = kt.utility.chance(prob, size=batch, **prop)
    dim = 1 + torch.arange(ndim, device=x.device)
    out = torch.empty_like(x)
    for i, batch in enumerate(x):
        out[i] = kt.filter.blur(batch, fwhm[i], dim) if bit[i] else batch

    return out


def bias(x, floor=(0, 0.5), points=4, prob=1, shared=False, generator=None):
    """Modulate tensor intensities by applying a smooth bias field.

    Generates and multiplies the input by a Perlin-noise field. We uniformly
    sample the number of control points and the field-intensity minimum. The
    maximum of the field is always 1, to avoid the need for renormalization.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    floor : float, optional
        Field-minimum sampling range, in [0, 1]. Pass 1 value `a` to sample
        from [a, 1). Pass 2 values `a` and `b` to sample from [a, b).
    points : int or sequence of int, optional
        Control-point range. Pass 1 value to set the upper bound `b`, keeping
        the lower bound `a` at 2. Pass 2 values to set `(a, b)`. Pass `2 * N`
        values to set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes.
    prob : float, optional
        Probability of modulating a channel.
    shared : bool, optional
        Modulate all channels of a batch with the same field.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Intensity-modulated tensor.

    """
    # Input.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2

    # Randomize across batches, or batches and channels.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    prop = dict(device=x.device, generator=generator)
    bit = kt.utility.chance(prob, size, **prop)

    # Bias-field minimum.
    floor = torch.as_tensor(floor, device=x.device).ravel()
    if floor.lt(0).any() or floor.gt(1).any():
        raise ValueError(f'bias floor {floor} is not in range [0, 1]')
    if len(floor) == 1:
        floor = torch.cat((floor, torch.ones_like(floor)))
    a, b = floor
    floor = torch.rand(*size, **prop) * (b - a) + a
    floor = bit * floor + ~bit

    # Conform control-point bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    points = torch.as_tensor(points, device=x.device).ravel()
    if len(points) not in (1, 2, 2 * ndim):
        raise ValueError(f'points {points} not of length 1, 2, or {2 * ndim}')
    if len(points) == 1:
        points = torch.cat((torch.tensor([2]), points))
    if len(points) == 2:
        points = points.repeat(ndim)

    # Control-point sampling.
    a, b = points[0::2], points[1::2] + 1
    if a.lt(2).any() or torch.tensor(x.shape[2:]).lt(b).any():
        raise ValueError(f'controls points {points} is not all in [2, size)')
    points = torch.rand(size[0], ndim, **prop) * (b - a) + a
    points = points.type(torch.int32)

    # Field.
    field = torch.empty_like(x)
    for i, p in enumerate(points):
        field[i] = kt.noise.perlin(x.shape[2:], p, batch=size[1], **prop)

    # Channel-wise normalization.
    dim = tuple(range(2, x.ndim))
    field -= field.amin(dim, keepdim=True)
    field /= field.amax(dim, keepdim=True)
    field = field * (1 - floor) + floor

    return x.mul(field)
