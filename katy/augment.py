"""Augmentation module."""


import torch
import katy as kt


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
    bit = kt.random.chance(prob, size, **prop)
    exp = bit * exp + ~bit * 1

    return x.pow(exp)


def noise(x, sd=(0.01, 0.1), prob=1, shared=False, generator=None):
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
    sd = sd * kt.random.chance(prob, size, **prop)

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
        raise ValueError(f'FWHM {fwhm} is not of length 1, 2, or 2N')

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
    bit = kt.random.chance(prob, size=batch, **prop)
    dim = 1 + torch.arange(ndim)
    out = torch.empty_like(x)
    for i, batch in enumerate(x):
        out[i] = kt.filter.blur(batch, fwhm[i], dim) if bit[i] else batch

    return out


def bias(
    x,
    floor=(0, 0.5),
    points=4,
    prob=1,
    shared=False,
    generator=None,
    return_bias=False,
):
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
    return_bias : bool, optional
        Return bias field.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Intensity-modulated tensor.
    (B, D, ...) torch.Tensor
        Bias field, if `return_bias` is True. `D` is 1 if `shared` is True.

    """
    # Input.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2

    # Randomize across batches, or batches and channels.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    dev = dict(device=x.device)
    prop = dict(**dev, generator=generator)
    bit = kt.random.chance(prob, size, **prop)

    # Bias-field minimum.
    floor = torch.as_tensor(floor, **dev).ravel()
    if floor.lt(0).any() or floor.gt(1).any():
        raise ValueError(f'bias floor {floor} is not in range [0, 1]')
    if len(floor) == 1:
        floor = torch.cat((floor, torch.ones_like(floor)))
    a, b = floor
    floor = torch.rand(*size, **prop) * (b - a) + a
    floor = bit * floor + ~bit

    # Conform control-point bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    points = torch.as_tensor(points, **dev).ravel()
    if len(points) not in (1, 2, 2 * ndim):
        raise ValueError(f'points {points} is not of length 1, 2, or 2N')
    if len(points) == 1:
        points = torch.cat((torch.tensor([2], **dev), points))
    if len(points) == 2:
        points = points.repeat(ndim)

    # Control-point sampling.
    a, b = points[0::2], points[1::2] + 1
    if a.lt(2).any() or torch.tensor(x.shape[2:], **dev).lt(b).any():
        raise ValueError(f'controls points {points} is not all in [2, size)')
    points = torch.rand(size[0], ndim, **prop) * (b - a) + a
    points = points.to(torch.int64)

    # Field.
    field = torch.empty(*size[:2], *x.shape[2:], **dev)
    for i, p in enumerate(points):
        field[i] = kt.noise.perlin(x.shape[2:], p, batch=size[1], **prop)

    # Channel-wise normalization.
    dim = tuple(range(2, x.ndim))
    field -= field.amin(dim, keepdim=True)
    field /= field.amax(dim, keepdim=True)
    field = field * (1 - floor) + floor

    x = x.mul(field)
    return (x, field) if return_bias else x


def downsample(x, factor=4, method='linear', prob=1, generator=None):
    """Reduce the resolution of an N-dimensional tensor.

    Downsamples a tensor and upsamples it again, to simulate upsampled lower
    resolution data. As the function assumes prior random blurring for
    augmentation, we do not apply anti-aliasing. For simplicity, downsampling
    always uses nearest neighbors. All channels will see the same effect.

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        Input tensor of spatial N-element size.
    factor : float, optional
        Subsampling range, greater or equal to 1. Pass 1 value `b` to sample
        from [1, b]. Pass 2 values `a` and `b` to sample from [a, b]. Pass
        `2 * N` values to set `(a_1, b_1, ..., a_N, b_N)` for N spatial axes.
    method : {'nearest', 'linear'}, optional
        Upsampling method. Use nearest for discrete-valued label maps.
    prob : float, optional
        Downsampling probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, *size) torch.Tensor
        Downsampled tensor.

    """
    # Inputs.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    size = x.shape[2:]

    # Conform factor bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    factor = torch.as_tensor(factor, device=x.device).ravel()
    if len(factor) not in (1, 2, 2 * ndim):
        raise ValueError(f'factor {factor} is not of length 1, 2, or 2N')
    if factor.lt(1).any():
        raise ValueError(f'factor {factor} includes values less than 1')
    if len(factor) == 1:
        factor = torch.cat((torch.ones_like(factor), factor))
    if len(factor) == 2:
        factor = factor.repeat(ndim)
    if torch.tensor(size, device=x.device).div(factor[1::2]).le(1).any():
        raise ValueError(f'factors {factor} not all less than size')

    # Factor sampling.
    prop = dict(device=x.device, generator=generator)
    batch = x.size(0)
    a, b = factor[0::2], factor[1::2]
    factor = torch.rand(batch, ndim, **prop) * (b - a) + a
    bit = kt.random.chance(prob, size=(batch, 1), **prop)
    factor = factor * bit + ~bit
    factor = 1 / factor

    if method == 'nearest':
        method = 'nearest-exact'
    if method == 'linear' and ndim == 2:
        method = 'bilinear'
    if method == 'linear' and ndim == 3:
        method = 'trilinear'

    # The built-in function is way faster than applying scaling matrices.
    f = torch.nn.functional.interpolate
    out = torch.empty_like(x)
    for i, s in enumerate(factor):
        batch = f(x[i:i + 1], scale_factor=s.tolist(), mode='nearest-exact')
        out[i] = f(batch, size=size, mode=method)

    return out


def remap(x, points=8, bins=256, prob=1, shared=False, generator=None):
    """Remap image intensities using smooth lookup tables.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input image.
    points : int or sequence of int, optional
        Control-point range. Pass 1 value to set the upper bound `b`, keeping
        the lower bound `a` at 2. Pass 2 values to set `(a, b)`.
    bins : int, optional
        Number of grayscale levels remap.
    prob : float, optional
        Remapping probability.
    shared : bool, optional
        Remap all channels the same way.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, C, ...) torch.Tensor
        Tensor with remapped intensities.

    """
    # Input.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    bins = torch.as_tensor(bins)

    # Randomize across batches, or batches and channels.
    size = torch.as_tensor(x.shape[:2])
    size[1 if shared else 2:] = 1
    prop = dict(device=x.device, generator=generator)

    # Control-point bounds.
    points = torch.as_tensor(points).ravel()
    if len(points) not in (1, 2):
        raise ValueError(f'points {points} is not of length 1 or 2')
    if len(points) == 1:
        points = torch.cat((torch.tensor([2]), points))

    # Control-point sampling.
    a, b = points[0], points[1] + 1
    if a.lt(2).any() or bins.lt(b).any():
        raise ValueError(f'controls points {points} is not all in [2, {bins})')
    points = torch.rand(size[0], **prop) * (b - a) + a
    points = points.to(torch.int64)

    # Discretization.
    dim = tuple(range(1 if shared else 2, x.ndim))
    x = x - x.amin(dim, keepdim=True)
    x = x / x.amax(dim, keepdim=True)
    x = x * (bins - 1)
    x = x.to(torch.int64)

    # Lookup tables. Oversample, as edges are zero.
    lut = torch.empty(*size, bins)
    for i, p in enumerate(points):
        n = kt.noise.perlin(size=bins * 2, points=p, batch=size[1], **prop)
        lut[i] = n[:, bins // 2:-bins // 2]

    # Normalize to full range. If shared, there will be only one channel.
    lut -= lut.amin(dim=-1, keepdim=True)
    lut /= lut.amax(dim=-1, keepdim=True)

    # Randomization.
    bit = kt.random.chance(prob, size, **prop).view(-1)
    lut.view(-1, bins)[~bit] = torch.linspace(0, 1, bins, device=x.device)

    # Indices into LUT.
    off_c = torch.arange(size[1], device=x.device) * bins
    off_b = torch.arange(size[0], device=x.device) * size[1] * bins
    ind = x + off_b.view(-1, 1, *[1] * ndim) + off_c.view(1, -1, *[1] * ndim)

    return lut.view(-1)[ind]


def crop(x, mask=None, crop=0.33, prob=1, generator=None, return_mask=False):
    """Crop an input image or label map along a random spatial axis.

    Has the same effect on all channels.

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        Input image or label map.
    mask : (B, C, *size) torch.Tensor, optional
        Boolean mask, defining a region to crop. None means the whole input.
    crop : float, optional
        Cropping range, in [0, 1]. Pass 1 value `b` to sample from [0, b].
        Pass 2 values `a` and `b` to sample from [a, b].
    prob : float, optional
        Cropping probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.
    return_mask : bool, optional
        Return cropping mask.

    Returns
    -------
    (B, C, *size) torch.Tensor
        Cropped input.
    (B, 1, *size) torch.Tensor
        Cropping mask, if `return_mask` is True.

    """
    # Inputs.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    size = x.shape[2:]

    # Conform bounds to (a, b).
    crop = torch.as_tensor(crop).ravel()
    if len(crop) not in (1, 2):
        raise ValueError(f'crop {crop} not of length 1, or 2')
    if crop.lt(0).any() or crop.gt(1).any():
        raise ValueError(f'crop {crop} includes values outside range [0, 1]')
    if len(crop) == 1:
        crop = torch.cat((torch.zeros_like(crop), crop))

    # Draw cropping amount, proportion to apply to lower end.
    prop = dict(device=x.device, generator=generator)
    batch = x.shape[:1]
    bit = kt.random.chance(prob, size=batch, **prop)
    a, b = crop.to(x.device)
    crop = bit * torch.rand(batch, **prop) * (b - a) + a
    dist = torch.rand(batch, **prop)

    # Treat channels as one.
    if mask is None:
        mask = torch.ones_like(x)
    mask = torch.as_tensor(mask).any(dim=1)

    out = torch.zeros_like(mask)
    for i, batch in enumerate(mask):
        # Mask extent along random axis.
        dim = torch.randint(ndim, size=())
        batch = batch.any(dim=[n for n in range(ndim) if n != dim])
        low, upp = batch.nonzero().aminmax()

        # Distribute cropping proportion between lower and upper end.
        cut = upp.sub(low).add(1) * crop[i]
        add = cut.mul(dist[i]).add(0.5).to(torch.int64)
        sub = cut.add(0.5).to(torch.int64) - add

        # Everything True except along axis.
        ind = [slice(0, s) for s in size]
        ind[dim] = slice(low + add, upp - sub + 1)
        out[i, *ind] = True

    # Restore channel dimension.
    mask = out.unsqueeze(1)
    x = x * mask
    return (x, mask) if return_mask else x
