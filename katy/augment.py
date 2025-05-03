"""Augmentation module."""

import katy as kt
import os
import torch

from . import utility


def gamma(x, gamma=0.5, *, prob=1, shared=False, generator=None):
    """Apply a random gamma transform to the intensities of a tensor.

    See https://en.wikipedia.org/wiki/Gamma_correction.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor.
    gamma : float or sequence of float, optional
        Exponent range. Pass 1 value `a` to sample from [1 - a, 1 + a]. Pass 2
        values `(a, b)` to sample from [a, b]. The range must be greater zero.
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
    # Conform gamma.
    gamma = torch.as_tensor(gamma).ravel()
    if len(gamma) not in (1, 2):
        raise ValueError(f'gamma {gamma} is not of length 1 or 2')
    if len(gamma) == 1:
        gamma = torch.tensor((1 - gamma, 1 + gamma))
    if gamma.le(0).any():
        raise ValueError(f'gamma range {gamma} must be greater 0')

    # Need intensities in [0, 1] range.
    dim = tuple(range(1 if shared else 2, x.ndim))
    x = kt.utility.normalize_minmax(x, dim)

    # Exponents. Randomize across batches, or batches and channels.
    prop = dict(device=x.device, generator=generator)
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    a, b = gamma
    exp = torch.rand(*size, **prop) * (b - a) + a

    # Randomize application.
    bit = kt.random.chance(prob, size, **prop)
    exp = bit * exp + ~bit * 1

    return x.pow(exp)


def noise(x, sd=(0.01, 0.1), *, prob=1, shared=False, generator=None):
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


@utility.batch(batch=True)
def blur(x, fwhm=1, *, prob=1, generator=None):
    """Blur a tensor by convolving its N spatial axes with Gaussian kernels.

    Uniformly samples anisotropic blurring full widths at half maximum (FWHM)
    between bounds `a` and `b`, blurring all channels of a batch the same way.

    Parameters
    ----------
    x : (..., C, ...) torch.Tensor
        Input tensor with or without batch dimension, depending on `batch`.
    fwhm : float or sequence of float, optional
        FWHM range. Pass 1 value to set the upper bound `b`, keeping the lower
        bound `a` at 0. Pass 2 values to set `(a, b)`. Pass `2 * N` values to
        set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Probability of blurring a batch entry.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (..., C, ...) torch.Tensor
        Blurred tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    fwhm = torch.as_tensor(fwhm, device=x.device).ravel()
    if len(fwhm) not in (1, 2, 2 * ndim):
        raise ValueError(f'FWHM {fwhm} is not of length 1, 2, or 2N')

    # Conform FWHM bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    if len(fwhm) == 1:
        fwhm = torch.cat((torch.zeros_like(fwhm), fwhm))
    if len(fwhm) == 2:
        fwhm = fwhm.repeat(ndim)

    # Smoothing at per-batch probability.
    prop = dict(device=x.device, generator=generator)
    if not kt.random.chance(prob, **prop):
        return x

    # Blur.
    a, b = fwhm[0::2], fwhm[1::2]
    fwhm = torch.rand(ndim, **prop) * (b - a) + a
    dim = torch.arange(1, 1 + ndim)
    return kt.filter.blur(x, fwhm, dim)


@utility.batch(batch=True)
def bias(
    x,
    floor=(0, 0.5),
    points=4,
    *,
    prob=1,
    shared=False,
    generator=None,
    return_bias=False,
):
    """Modulate intensities by applying a smooth bias field.

    Generates and multiplies the input by a Perlin-noise field. We uniformly
    sample the number of control points and the field-intensity minimum. The
    maximum of the field is always 1, to avoid the need for renormalization.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    floor : float, optional
        Field-minimum sampling range, in [0, 1]. Pass 1 value `a` to sample
        from [a, 1). Pass 2 values `a` and `b` to sample from [a, b).
    points : int or sequence of int, optional
        Control-point range. Pass 1 value to set the upper bound `b`, keeping
        the lower bound `a` at 2. Pass 2 values to set `(a, b)`. Pass `2 * N`
        values to set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes.
    batch : bool, optional
        Expect batched inputs.
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
    (..., C, ...) torch.Tensor
        Intensity-modulated tensor.
    (..., D, ...) torch.Tensor
        Bias field, if `return_bias` is True. `D` is 1 if `shared` is True.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    size = x.shape[1:]
    ndim = x.ndim - 1
    channels = (1 if shared else x.size(0), *[1] * ndim)

    # Bias-field minimum.
    floor = torch.as_tensor(floor).ravel()
    if floor.lt(0).any() or floor.gt(1).any():
        raise ValueError(f'bias floor {floor} is not in range [0, 1]')
    if len(floor) == 1:
        floor = torch.cat((floor, torch.ones_like(floor)))

    # Sampling.
    prop = dict(device=x.device, generator=generator)
    bit = kt.random.chance(prob, size=channels, **prop)
    a, b = floor
    floor = torch.rand(channels, **prop) * (b - a) + a
    floor = bit * floor + ~bit

    # Conform control-point bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    points = torch.as_tensor(points).ravel()
    if len(points) not in (1, 2, 2 * ndim):
        raise ValueError(f'points {points} is not of length 1, 2, or 2N')
    if len(points) == 1:
        points = torch.cat((torch.tensor([2]), points))
    if len(points) == 2:
        points = points.repeat(ndim)

    # Control-point sampling.
    a, b = points[0::2], points[1::2] + 1
    if a.lt(2).any() or torch.tensor(size).lt(b).any():
        raise ValueError(f'controls points {points} is not all in [2, size)')
    a = a.to(x.device)
    b = b.to(x.device)
    points = torch.rand(1, ndim, **prop) * (b - a) + a
    points = points.to(torch.int64)

    # Field.
    field = kt.noise.perlin(size, points, batch=channels[0], **prop)

    # Channel-wise normalization.
    field = kt.utility.normalize_minmax(field, dim=tuple(range(1, x.ndim)))
    field = field * (1 - floor) + floor
    x = x.mul(field)

    return (x, field) if return_bias else x


@utility.batch(batch=True)
def downsample(x, factor=4, *, method='linear', prob=1, generator=None):
    """Reduce the resolution of a tensor.

    Downsamples a tensor and upsamples it again, to simulate upsampled lower
    resolution data. As the function assumes prior random blurring for
    augmentation, we do not apply anti-aliasing. For simplicity, downsampling
    always uses nearest neighbors. All channels will see the same effect.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    factor : float, optional
        Subsampling range, greater or equal to 1. Pass 1 value `b` to sample
        from [1, b]. Pass 2 values `a` and `b` to sample from [a, b]. Pass
        `2 * N` values to set `(a_1, b_1, ..., a_N, b_N)` for N spatial axes.
    method : {'nearest', 'linear'}, optional
        Upsampling method. Use nearest for discrete-valued label maps.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Downsampling probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Downsampled tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    size = x.shape[1:]

    # Conform factor bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    factor = torch.as_tensor(factor).ravel()
    if len(factor) not in (1, 2, 2 * ndim):
        raise ValueError(f'factor {factor} is not of length 1, 2, or 2N')
    if factor.lt(1).any():
        raise ValueError(f'factor {factor} includes values less than 1')
    if len(factor) == 1:
        factor = torch.cat((torch.ones_like(factor), factor))
    if len(factor) == 2:
        factor = factor.repeat(ndim)
    if torch.tensor(size).div(factor[1::2]).le(1).any():
        raise ValueError(f'factors {factor} not all less than size')

    # Factor sampling.
    prop = dict(device=x.device, generator=generator)
    factor = factor.to(x.device)
    a, b = factor[0::2], factor[1::2]
    factor = torch.rand(ndim, **prop) * (b - a) + a
    bit = kt.random.chance(prob, **prop)
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
    x = f(x.unsqueeze(0), scale_factor=factor.tolist(), mode='nearest-exact')
    return f(x, size=size, mode=method).squeeze(0)


@utility.batch(batch=True)
def remap(x, points=8, bins=256, *, prob=1, shared=False, generator=None):
    """Remap image intensities using smooth lookup tables.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    points : int or sequence of int, optional
        Control-point range. Pass 1 value to set the upper bound `b`, keeping
        the lower bound `a` at 2. Pass 2 values to set `(a, b)`.
    bins : int, optional
        Number of grayscale levels remap.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Remapping probability.
    shared : bool, optional
        Remap all channels the same way.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (..., C, ...) torch.Tensor
        Remapped intensities.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    bins = torch.as_tensor(bins)

    # Shared channels.
    channels = 1 if shared else x.size(0)
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
    points = torch.rand(1, **prop) * (b - a) + a
    points = points.to(torch.int64)

    # Discretization.
    dim = tuple(range(0 if shared else 1, x.ndim))
    x = kt.utility.normalize_minmax(x, dim) * (bins - 1e-3)
    x = x.to(torch.int64)

    # Lookup tables. Oversample, as edges are zero.
    lut = kt.noise.perlin(bins * 2, points, batch=channels, **prop)
    lut = lut[:, bins // 2:-bins // 2]

    # Normalize to full range. If shared, there will be only one channel.
    lut = kt.utility.normalize_minmax(lut, dim=-1)

    # Randomization.
    bit = kt.random.chance(prob, size=channels, **prop).view(-1)
    lut[~bit] = torch.linspace(0, 1, bins, device=x.device)

    # Indices into LUT.
    offset = torch.arange(channels, device=x.device) * bins
    ind = x + offset.view(-1, *[1] * ndim)
    return lut.reshape(-1)[ind]


@utility.batch(batch=True)
def crop(
    x,
    crop=0.33,
    *,
    prob=1,
    generator=None,
    return_mask=False,
):
    """Crop a tensor along a random spatial axis.

    Has the same effect on all channels.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    crop : float, optional
        Cropping range, in [0, 1]. Pass 1 value `b` to sample from [0, b].
        Pass 2 values `a` and `b` to sample from [a, b].
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Cropping probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.
    return_mask : bool, optional
        Return cropping mask.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Cropped input.
    (..., 1, *size) torch.Tensor
        Cropping mask, if `return_mask` is True.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    size = x.shape[1:]

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
    a, b = crop
    crop = torch.rand(1, **prop) * (b - a) + a
    crop = crop * kt.random.chance(prob, **prop)

    # Distribute cropping between lower and upper end of random axis.
    dim = torch.randint(ndim, size=())
    crop = crop * size[dim]
    low = torch.rand(1, **prop)
    low = (crop * low).to(torch.int64)
    upp = (crop - low).to(torch.int64)

    # Everything True except along axis.
    ind = [slice(0, s) for s in size]
    ind[dim] = slice(low, size[dim] - upp)

    # Singleton channel dimension.
    mask = torch.zeros_like(x[:1])
    mask[:, *ind] = 1
    x = x * mask

    return (x, mask) if return_mask else x


@utility.batch(batch=True)
def lines(x, lines=3, *, prob=1, generator=None):
    """Fill lines along a spatial axis with a random value between 0 and 1.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input tensor with or without batch dimension, depending on `batch`.
    lines : float, optional
        Line sampling range, greater than 0. Pass 1 value `b` to sample from
        [1, b]. Pass 2 values `a` and `b` to sample from [a, b].
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Line-corruption probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Corrupted tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    size = x.shape[1:]

    # Conform bounds to (a, b).
    lines = torch.as_tensor(lines).ravel()
    if len(lines) not in (1, 2):
        raise ValueError(f'lines {lines} is not of length 1 or 2')
    if lines.lt(1).any():
        raise ValueError(f'lines {lines} includes values less than 1')
    if len(lines) == 1:
        lines = torch.cat((torch.ones_like(lines), lines))

    # Axis.
    prop = dict(device=x.device, generator=generator)
    dim = torch.randint(ndim, size=(), **prop)

    # Number of lines.
    a, b = lines
    lines = torch.randint(a, b + 1, size=(), **prop)
    lines = lines * kt.random.chance(prob, **prop)

    # Line selection.
    ind = torch.rand(lines, **prop)
    ind = ind.mul(size[dim] - 1).add(0.5).to(torch.int64)

    # Fill value.
    val = torch.rand((), **prop)
    return x.clone().index_fill(dim + 1, ind, val)


@utility.batch(batch=True)
def roll(x, shift=0.1, *, prob=1, generator=None):
    """Roll a tensor along a random spatial axis.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input image or label map.
    shift : float or sequence of float, optional
        Shift range relative to `size`, in [0, 1]. Pass 1 value `b` to
        sample from [0, b]. Pass 2 values `(a, b)` to sample from [a, b].
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Rolling probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Rolled input.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1

    # Conform bounds to (a, b).
    shift = torch.as_tensor(shift).ravel()
    if len(shift) not in (1, 2):
        raise ValueError(f'shift {shift} is not of length 1, or 2')
    if shift.lt(0).any() or shift.gt(1).any():
        raise ValueError(f'shift {shift} has values outside range [0, 1]')
    if len(shift) == 1:
        shift = torch.cat((torch.zeros_like(shift), shift))

    prop = dict(device=x.device, generator=generator)
    if not kt.random.chance(prob, **prop):
        return x

    # Draw shifts.
    a, b = shift
    shift = torch.rand(1, **prop) * (b - a) + a
    shift = shift * (-1) ** kt.random.chance(0.5, **prop)

    dim = torch.randint(ndim, size=()) + 1
    return x.roll(shifts=int(shift * x.shape[dim]), dims=int(dim))


@utility.batch(batch=True)
def flip(x, dim=0, labels=None, *, generator=None):
    """Flip an N-dimensional tensor along a random axis.

    x : torch.Tensor
        Input tensor with or without batch dimension, depending on `batch`.
    dim : int or sequence of int
        Spatial dimensions to draw from, in [-N, N). None means all.
    labels : os.PathLike or dict, optional
        Label-name mapping for left-right remapping.
    batch : bool, optional
        Expect batched inputs.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    torch.Tensor
        Potentially flipped tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1

    if dim is None:
        dim = range(ndim)
    dim = torch.as_tensor(dim).ravel()
    if dim.lt(-ndim).any() or dim.ge(ndim).any():
        raise ValueError(f'dimensions {dim} not in [-{ndim}, {ndim - 1}]')

    # No flip as probable as any flip.
    ind = torch.randint(low=-1, high=len(dim), size=(), generator=generator)
    if ind < 0:
        return x
    dim = dim[ind]

    # Mapping from labels to names. Make all keys Python integers, because
    # JSON stores keys as strings, PyTorch tensors are not hashable, and
    # torch.uint8 scalars are interpreted as boolean indices.
    if isinstance(labels, (str, os.PathLike)):
        labels = kt.io.load(labels)

    if labels:
        labels = {int(k): v.lower() for k, v in labels.items()}
        retour = {v: k for k, v in labels.items()}
        for k, v in labels.items():
            if 'left' in v:
                labels[k] = retour[v.replace('left', 'right')]
            elif 'right' in v:
                labels[k] = retour[v.replace('right', 'left')]
            else:
                labels[k] = k

        # Restore type.
        x = kt.labels.remap(x, mapping=labels).to(x.dtype)

    # Account for channel dimension.
    return x.flip(dim if dim < 0 else dim + 1)
