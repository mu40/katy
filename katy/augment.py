"""Augmentation module."""

import katy as kt
import os
import torch

from . import utility


def gamma(x, /, gamma=0.5, *, prob=1, shared=False):
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
    x = kt.utility.normalize(x, dim=range(1 if shared else 2, x.ndim))

    # Exponents. Randomize across batches, or batches and channels.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    a, b = gamma
    exp = torch.rand(*size, device=x.device) * (b - a) + a

    # Randomize application.
    bit = kt.random.chance(prob, size, device=x.device)
    exp = bit * exp + ~bit * 1

    return x.pow(exp)


def noise(x, /, sd=(0.01, 0.1), *, prob=1, shared=False):
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

    Returns
    -------
    (B, C, ...) torch.Tensor
        Tensor with added noise.

    """
    # Inputs.
    x = torch.as_tensor(x)
    sd = torch.as_tensor(sd).ravel()

    # Standard deviation.
    size = torch.as_tensor(x.shape)
    size[1 if shared else 2:] = 1
    a, b = (0, *sd) if len(sd) == 1 else sd
    sd = torch.rand(*size, device=x.device) * (b - a) + a
    sd = sd * kt.random.chance(prob, size, device=x.device)

    return x + sd * torch.randn(x.shape, device=x.device)


@utility.batch(batch=True)
def blur(x, /, fwhm=1, *, prob=1):
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

    Returns
    -------
    (..., C, ...) torch.Tensor
        Blurred tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    fwhm = torch.as_tensor(fwhm).ravel()
    if len(fwhm) not in (1, 2, 2 * ndim):
        raise ValueError(f'FWHM {fwhm} is not of length 1, 2, or 2N')

    # Conform FWHM bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    if len(fwhm) == 1:
        fwhm = torch.cat((torch.zeros_like(fwhm), fwhm))
    if len(fwhm) == 2:
        fwhm = fwhm.repeat(ndim)

    # Smoothing at per-batch probability.
    if not kt.random.chance(prob):
        return x

    # Blur.
    a, b = fwhm[0::2], fwhm[1::2]
    fwhm = torch.rand(ndim) * (b - a) + a
    dim = torch.arange(1, 1 + ndim)
    return kt.filter.blur(x, fwhm, dim)


@utility.batch(batch=True)
def bias(
    x,
    /,
    floor=(0, 0.5),
    points=4,
    *,
    prob=1,
    shared=False,
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
    bit = kt.random.chance(prob, size=channels, device=x.device)
    a, b = floor
    floor = torch.rand(channels, device=x.device) * (b - a) + a
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
    points = torch.rand(1, ndim) * (b - a) + a
    points = points.to(torch.int64)

    # Field.
    field = kt.noise.perlin(size, points, batch=channels[0], device=x.device)

    # Channel-wise normalization.
    field = kt.utility.normalize(field, dim=range(1, x.ndim))
    field = field * (1 - floor) + floor
    x = x.mul(field)

    return (x, field) if return_bias else x


@utility.batch(batch=True)
def downsample(x, /, factor=4, *, mode='linear', prob=1):
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
    mode : {'nearest', 'linear'}, optional
        Upsampling mode. Use nearest for discrete-valued label maps.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Downsampling probability.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Downsampled tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    size = x.shape[1:]
    if not x.dtype.is_floating_point:
        x = x.to(torch.get_default_dtype())

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
    a, b = factor[0::2], factor[1::2]
    factor = torch.rand(ndim) * (b - a) + a
    bit = kt.random.chance(prob)
    factor = factor * bit + ~bit
    factor = 1 / factor

    if mode == 'nearest':
        mode = 'nearest-exact'
    if mode == 'linear' and ndim == 2:
        mode = 'bilinear'
    if mode == 'linear' and ndim == 3:
        mode = 'trilinear'

    # The built-in function is way faster than applying scaling matrices.
    f = torch.nn.functional.interpolate
    x = f(x.unsqueeze(0), scale_factor=factor.tolist(), mode='nearest-exact')
    return f(x, size=size, mode=mode).squeeze(0)


@kt.utility.batch(batch=True)
def remap(x, /, points=8, bins=256, *, min=0, max=0.99, prob=1, shared=False):
    """Remap image intensities using random lookup tables.

    Parameters
    ----------
    x : (..., C, *size) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    points : int or sequence of int, optional
        Control-point range, at least 2. Pass 1 value to set the upper bound
        `b`, keeping the lower bound `a` at 2. Pass 2 values to set `(a, b)`.
    bins : int, optional
        Number of grayscale levels to remap.
    min : float, optional
        Lower quantile in [0, 1] to clip to before discretization.
    max : float, optional
        Upper quantile in [0, 1] to clip to before discretization.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Remapping probability.
    shared : bool, optional
        Remap all channels the same way.

    Returns
    -------
    (..., C, *size) torch.Tensor
        Remapped intensities in [0, 1].

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = x.ndim - 1
    channels = 1 if shared else x.size(0)

    # Control-point bounds.
    points = torch.as_tensor(points).ravel().tolist()
    if len(points) not in (1, 2):
        raise ValueError(f'points {points} is not of length 1 or 2')
    if len(points) == 1:
        points = [2, *points]

    a, b = points
    if a < 2:
        raise ValueError(f'lower control-point bound {a} is less than 2')
    if a > b:
        raise ValueError(f'control-point bounds {points} are not ordered')
    if b >= bins:
        raise ValueError(f'upper control-point bound {b} is not below {bins}')

    # Control-point count. Sample on CPU to avoid synchronizing with GPU.
    points = int(torch.randint(a, b + 1, size=()))

    # Discretization. Clip to quantiles so outliers do not compress tissue.
    dim = range(0 if shared else 1, x.ndim)
    x = kt.utility.normalize(x, dim=dim, min=min, max=max)
    copy = x
    x = x * (bins - 1e-3)
    x = x.to(torch.int64)

    # Lookup tables. Uniform random values at evenly spaced control points.
    knots = torch.rand(channels, 1, points, device=x.device)
    lut = torch.nn.functional.interpolate(
        knots,
        size=bins,
        mode='linear',
        align_corners=True,
    )
    lut = lut.view(channels, bins)

    # Rank transform. Replace each LUT value by its rank among all values,
    # scaled to [0, 1]. This spaces the output levels evenly to keep contrast.
    lut = lut.argsort(-1).argsort(-1).float() / (bins - 1)

    # Indexing into LUT.
    offset = torch.arange(channels, device=x.device) * bins
    ind = x + offset.view(-1, *[1] * ndim)
    out = lut.reshape(-1)[ind]

    # Randomization.
    bit = kt.random.chance(prob, size=(channels, *[1] * ndim), device=x.device)
    return out.where(bit, copy)


@utility.batch(batch=True)
def crop(x, /, crop=0.33, *, prob=1, return_mask=False):
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
    a, b = crop
    crop = torch.rand(1) * (b - a) + a
    crop = crop * kt.random.chance(prob)

    # Distribute cropping between lower and upper end of random axis.
    dim = torch.randint(ndim, size=())
    crop = crop * size[dim]
    low = torch.rand(1)
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
def lines(x, /, lines=3, *, prob=1):
    """Fill lines along an axis with a random value between 0 and 1.

    Affects all channels the same way.

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
    dim = torch.randint(ndim, size=())

    # Number of lines.
    a, b = lines
    lines = torch.randint(a, b + 1, size=())
    lines = lines * kt.random.chance(prob)

    # Line selection.
    ind = torch.randperm(size[dim], device=x.device)[:lines]

    # Fill value.
    value = torch.rand(size=())
    return x.index_fill(dim=dim + 1, index=ind, value=value)


@utility.batch(batch=True)
def roll(x, /, shift=0.1, *, prob=1):
    """Roll a tensor along a random spatial axis.

    Affects all channels the same way.

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

    if not kt.random.chance(prob):
        return x

    # Draw shifts.
    a, b = shift
    shift = torch.rand(1) * (b - a) + a
    shift = shift * (-1) ** kt.random.chance(0.5)

    dim = torch.randint(ndim, size=()) + 1
    return x.roll(shifts=int(shift * x.shape[dim]), dims=int(dim))


@utility.batch(batch=True)
def flip(x, /, dim=0, labels=None, *, prob=1):
    """Flip N-dimensional tensors along a random axis.

    Applies the same operation across all channels. Providing label names for
    label-map inputs results in left-right remapping on any flip.

    Parameters
    ----------
    x : (..., C, *space) torch.Tensor
        Tensors with or without batch dimension, depending on `batch`.
    dim : int or sequence of int, optional
        Spatial dimensions to draw from, in [-N, N). None means all.
    labels : os.PathLike or dict, optional
        Label-name mapping for left-right remapping. Applies to first tensor.
    batch : bool, optional
        Expect batched inputs.
    prob : {0, 1}, optional
        Use 0 to disable, 1 to enable.

    Returns
    -------
    (..., C, *space) torch.Tensor
        Flipped tensor.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x)
    ndim = len(x.shape[1:])

    if prob not in (0, 1):
        raise ValueError(f'probability {prob} must be 0 or 1')
    if prob == 0:
        return x

    # Dimensions. Account for channel dimension if non-negative.
    if dim is None:
        dim = range(ndim)
    dim = torch.as_tensor(dim).ravel()
    if dim.lt(-ndim).any() or dim.ge(ndim).any():
        raise ValueError(f'dimensions {dim} not in [-{ndim}, {ndim - 1}]')
    dim += (dim > -1)

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

    # Randomization. No flip as likely as any flip.
    ind = torch.randint(-1, len(dim), size=())
    if ind < 0:
        return x

    return kt.labels.remap(x, mapping=labels).flip(dim[ind])


@utility.batch(batch=True)
def permute(x, /, *, prob=1):
    """Randomly permute the channels of a tensor.

    Parameters
    ----------
    x : (..., C, *space) torch.Tensor
        Input with or without batch dimension, depending on `batch`.
    batch : bool, optional
        Expect batched inputs.
    prob : {0, 1}, optional
        Use 0 to disable, 1 to enable.

    Returns
    -------
    (..., C, *space) torch.Tensor
        Permuted channels.

    """
    # Input of shape (C, *space). Batches permuted separately by decorator.
    x = torch.as_tensor(x)
    channels = x.size(0)

    if prob not in (0, 1):
        raise ValueError(f'probability {prob} must be 0 or 1')
    if prob == 0:
        return x

    ind = torch.randperm(channels, device=x.device)
    return x[ind]


@utility.batch(batch=True)
def motion(x, /, blocks=10, moves=3, *, shift=10, angle=10, prob=1):
    """Motion-corrupt an N-dimensional image by moving k-space segments.

    Partitions k-space into an odd number of blocks along a random axis and
    moves a subset. This simulates motion during sequential acquisition along
    the segmentation axis, as with MPRAGE. The central low-frequency block
    never moves. Supports 2D and 3D. Each batch of channels moves the same way.

    Parameters
    ----------
    x : (..., C, ...) torch.Tensor
        Image with or without batch dimension, depending on `batch`.
    blocks : int or sequence of int, optional
        Range of k-space segments. Pass 1 value `a` to sample from [3, a].
        Pass 2 values `(a, b)` to sample from [a, b]. Must be greater than 2.
    moves : int, optional
        Maximum number of moving segments. Must be less than `blocks`.
    shift : float or sequence of float, optional
        See `katy.random.affine`.
    angle : float or sequence of float, optional
        See `katy.random.affine`.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Probability of corrupting a batch entry.

    Returns
    -------
    (..., C, ...) torch.Tensor
        Corrupted image.

    """
    # Input of shape `(C, *size)`. Batches handled by decorator.
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if not kt.random.chance(prob):
        return x

    blocks = torch.as_tensor(blocks).ravel()
    moves = torch.as_tensor(moves).ravel()
    if len(blocks) not in (1, 2):
        raise ValueError(f'blocks {blocks} is not of length 1, or 2')
    if blocks.lt(3).any():
        raise ValueError(f'blocks {blocks} is less than 3')
    if len(blocks) == 1:
        blocks = torch.cat((blocks.new_tensor([3]), blocks))
    if moves < 1 or blocks.max() <= moves:
        raise ValueError(f'moves {moves} is not in range [1, blocks)')

    # Number of blocks and moving blocks.
    a, b = blocks
    low = (a - 1) // 2
    upp = (b - 1) // 2
    blocks = 2 * torch.randint(low, upp + 1, size=()) + 1
    moves = torch.randint(1, min(moves + 1, blocks), size=())

    # Stationary base and moved blocks as batch entries.
    base = x.unsqueeze(0)
    x = x.expand(moves, *x.shape)
    trans = kt.random.affine(x, shift, angle, scale=0, shear=0)
    x = kt.transform.apply(x, trans)

    # Base and moved blocks: moves + 1, *rest, to_segment.
    dim = torch.randint(2, x.ndim, size=())
    space = tuple(range(2, x.ndim))
    x = torch.cat((base, x))
    k = torch.fft.fftn(x, dim=space)
    k = k.movedim(dim, -1)

    # Indices of blocks to move, skipping the stationary central block.
    # Mapping from all blocks to batch index.
    cen = blocks // 2
    to_move = torch.randperm(blocks - 1, device=x.device)[:moves]
    to_move = to_move + (to_move >= cen)
    mapping = torch.zeros(blocks, dtype=torch.int64, device=x.device)
    mapping[to_move] = torch.arange(1, moves + 1, device=x.device)

    # Map line indices along segmentation axis to blocks, to batches. Compute
    # block indices as if we had fftshifted the segmentation axis, since we
    # want the "central" low-frequency lines to remain stationary.
    _, *rest, width = k.shape
    ind = torch.arange(width, device=x.device)
    ind = (ind + width // 2) % width  # Logical fftshift.
    ind = ind.mul(blocks) // width
    ind = mapping[ind]

    # Gather over shot axis, restore shape.
    ind = ind.expand(1, *rest, width)
    k = torch.gather(k, dim=0, index=ind)
    k = k.movedim(-1, dim)
    return torch.fft.ifftn(k, dim=space).real.squeeze(0)


def resize(x, /, *, min=None, max=None, batch=True, prob=1):
    """Resize a tensor along a random spatial axis.

    Selects a random spatial axis and samples its output size using `min`,
    `max`, and the current size. `min` sets a lower bound; `max` sets an
    upper bound. One-sided bounds use the current size as the other bound,
    but sampled sizes are never smaller than `min` or larger than `max`.

    Parameters
    ----------
    x : (..., C, ...) torch.Tensor
        Input tensor of spatial `N`-element size. Batch depending on `batch`.
    min : int or sequence of int, optional
        Minimum output size. Pass 1 or N values.
    max : int or sequence of int, optional
        Maximum output size. Pass 1 or N values.
    batch : bool, optional
        Expect batched inputs.
    prob : float, optional
        Resizing probability.

    Returns
    -------
    (..., C, ...) torch.Tensor
        Resized tensor.

    """
    # Spatial dimensions start after channel, and possibly batch.
    x = torch.as_tensor(x)
    ndim = x.ndim - (2 if batch else 1)
    size = torch.tensor(x.shape[-ndim:])

    def bound(v, name):
        if v is None:
            return None
        v = torch.as_tensor(v).ravel()
        if len(v) not in (1, ndim):
            raise ValueError(f'{name} {v} not of length 1 or {ndim}')
        if v.lt(1).any():
            raise ValueError(f'{name} {v} is less than 1')
        return v.expand(ndim)

    lo = bound(min, 'min')
    hi = bound(max, 'max')
    if lo is not None and hi is not None:
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            if a > b:
                raise ValueError(f'min {lo} exceeds max {hi} along axis {i}')

    if not kt.random.chance(prob) or (min is None and max is None):
        return x

    # Spatial axis and bounds. Missing bounds use the current size, unless it
    # is smaller than the minimum or larger than the maximum.
    dim = torch.randint(ndim, size=())
    a = torch.minimum(size[dim], hi[dim]) if lo is None else lo[dim]
    b = torch.maximum(size[dim], lo[dim]) if hi is None else hi[dim]
    size[dim] = torch.randint(a, b + 1, size=())
    return kt.utility.resize(x, size, batch=batch)
