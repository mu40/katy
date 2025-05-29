"""Generation module."""

import functools
import katy as kt
import torch


def chance(prob=0.5, size=1, *, device=None, generator=None):
    """Return True with given probability.

    Parameters
    ----------
    prob : float, optional
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

    if isinstance(size, torch.Tensor):
        size = size.tolist()

    return torch.rand(size, device=device, generator=generator) < prob


def affine(x, /, shift=30, angle=30, scale=0.1, shear=0.1, *, generator=None):
    """Draw N-dimensional matrix transforms.

    Uniformly samples parameters from [a, b]. For each parameter, pass one
    value `a` to sample from [-a, a], pass both `(a, b)`, or pass `2 * N`
    values to set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes. Supports
    2D and 3D. In 2D, there is only 1 angle and 1 shearing parameter.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor, defining batch size, space, and device.
    shift : float or sequence of float, optional
        Translation range in voxels.
    angle : float or sequence of float, optional
        Rotation range in degrees, about the center of the input tensor.
    scale : float or sequence of float, optional
        Scaling range, offset from 1. Pass `a` for factors in [1 - a, 1 + a].
    shear : float or sequence of float, optional
        Shearing range.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, N + 1, N + 1) torch.Tensor
        Matrix transforms.

    """
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(f'{x.shape} is not a 2D or 3D tensor shape')

    def conform(f, name, sizes):
        f = torch.as_tensor(f, device=x.device).ravel()

        if len(f) not in (1, 2, 2 * ndim):
            raise ValueError(f'size of {ndim}D {name} {f} is not in {sizes}')
        if len(f) == 1:
            f = torch.cat((-f, f))
        if len(sizes) == 3 and len(f) == 2:
            f = f.repeat(ndim)

        return f

    # Conform bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    len_1 = (1, 2, 2 * ndim)
    len_2 = len_1 if ndim == 3 else (1, 2)
    shift = conform(shift, name='shift', sizes=len_1)
    angle = conform(angle, name='angle', sizes=len_2)
    scale = conform(scale, name='scale', sizes=len_1) + 1
    shear = conform(shear, name='shear', sizes=len_2)
    if scale.le(0).any():
        raise ValueError(f'scale {scale} includes non-positive values')

    # Sample parameters.
    prop = dict(device=x.device, generator=generator)
    bounds = torch.cat((shift, angle, scale, shear))
    a, b = bounds[0::2], bounds[1::2]
    par = torch.rand(x.size(0), *a.shape, **prop) * (b - a) + a

    # Construct matrix, with double precision.
    prop = dict(device=x.device)
    splits = (3, 3, 3, 3) if ndim == 3 else (2, 1, 2, 1)
    par = torch.split(par, splits, dim=-1)
    mat = kt.transform.compose_affine(*par, **prop)

    # Apply in centered frame.
    return kt.transform.center_matrix(x.shape[2:], mat)


def warp(x, /, disp=25, points=16, damp=0.33, steps=0, *, generator=None):
    """Draw N-dimensional displacement fields.

    Uniformly samples control points and the maximum displacement strength.
    For the displacement and control-point ranges, pass 1 value to set the
    upper bound `b`, keeping the lower bound `a` at 0 and 2, respectively.
    Pass 2 values to set `(a, b)`. Pass `2 * N` for the N spatial axes, to
    to set `(a_1, b_1, ..., a_N, b_N)` for the N spatial axes.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Input tensor, defining batch size, space, and device.
    disp : float or sequence of float, optional
        Per-axis displacement range, in voxels. Applies to lowest frequency.
    points : int or sequence of int, optional
        Control-point range, greater 1. Controls the spatial frequencies.
    damp : float, optional
        Positive damping factor. Reduces amplitudes of higher frequencies.
    steps : int, optional
        Number of integration steps. Consider damping less when integrating.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, N, ...) torch.Tensor
        Displacement fields.

    """
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    batch = x.size(0)
    space = x.shape[2:]
    dev = dict(device=x.device)

    # Conform control-point bounds to (a_1, b_1, a_2, b_2, ..., a_N, b_N).
    points = torch.as_tensor(points, **dev).ravel()
    if len(points) not in (1, 2, 2 * ndim):
        raise ValueError(f'points {points} is not of length 1, 2, or 2N')
    if len(points) == 1:
        points = torch.cat((torch.tensor([2], **dev), points))
    if len(points) == 2:
        points = points.repeat(ndim)

    # Control-point sampling.
    prop = dict(**dev, generator=generator)
    a, b = points[0::2], points[1::2] + 1
    if a.lt(2).any() or torch.tensor(space, **dev).lt(b).any():
        raise ValueError(f'controls points {points} is not all in [2, size)')
    points = torch.rand(batch, ndim, **prop) * (b - a) + a
    points = points.to(torch.int64)

    # Damping factor, reducing amplitudes of higher (effective) frequencies.
    damp = torch.as_tensor(damp, **dev).ravel()
    if len(damp) != 1 or damp.lt(0):
        raise ValueError(f'damping factor {damp} is not a positive scalar')
    feff_2 = points.square().sum(-1)
    fmin_2 = a.square().sum()
    damp = ((1 + fmin_2 ) / (1 + feff_2)).pow(damp)
    damp = damp.view(batch, 1, *[1] * ndim)

    # Conform displacement range.
    disp = torch.as_tensor(disp, **dev).ravel()
    if len(disp) not in (1, 2, 2 * ndim):
        raise ValueError(f'displacement {disp} is not of length 1, 2, or 2N')
    if len(disp) == 1:
        disp = torch.cat((torch.zeros_like(disp), disp))
    if len(disp) == 2:
        disp = disp.repeat(ndim)

    # Displacement-strength sampling.
    a, b = disp[0::2], disp[1::2]
    disp = torch.rand(batch, ndim, **prop) * (b - a) + a
    disp = disp.view(batch, ndim, *[1] * ndim)

    # Field of zero mean.
    field = torch.empty(batch, ndim, *space, **dev)
    for i, p in enumerate(points):
        field[i] = kt.noise.perlin(space, p, batch=ndim, **prop)
    dim = tuple(range(2, x.ndim))
    field -= field.mean(dim, keepdim=True)
    field *= disp * damp / field.abs().amax(dim, keepdim=True)

    return kt.transform.integrate(field, steps)


def replay(f, /, *, generator=None, device=None):
    """Initialize each call to a function with the same random state.

    Calls to a wrapped augmentation function should generally apply the same
    augmentation each time. However, changing the function arguments between
    calls may lead to different results depending on the wrapped function.

    Parameters
    ----------
    f : callable
        Callable taking a `torch.Generator` as keyword argument `generator`.
    generator : torch.Generator, optional
        Pseudo-random number generator.
    device : torch.device, optional
        Device used if `generator` is None.

    Returns
    -------
    function
        Wrapper function.

    """
    if generator is None:
        generator = torch.Generator(device)

    state = generator.get_state()

    # Reset before call so the state has advanced after. Otherwise, the next
    # call to `f` would depart from the same state even without the wrapper.
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        generator.set_state(state)
        return f(*args, **kwargs, generator=generator)

    return wrapper
