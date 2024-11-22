"""Generation module."""


import torch
import kathryn as kt


def crop(x, crop=0.33, prob=1, generator=None):
    """Generate a boolean mask for multiplicative cropping.

    The output mask will crop the input along a random spatial axis.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Boolean mask.
    crop : float, optional
        Cropping range, in [0, 1]. Pass 1 value `b` to sample from [0, b].
        Pass 2 values `a` and `b` to sample from [a, b].
    prob : float, optional
        Cropping probability.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, 1, ...) torch.Tensor
        Cropping mask.

    """
    # Inputs.
    x = torch.as_tensor(x)
    ndim = x.ndim - 2
    size = x.shape[2:]

    # Conform bounds to (a, b).
    crop = torch.as_tensor(crop, device=x.device).ravel()
    if len(crop) not in (1, 2):
        raise ValueError(f'crop {crop} not of length 1, or 2')
    if crop.lt(0).any() or crop.gt(1).any():
        raise ValueError(f'crop {crop} includes values outside range [0, 1]')
    if len(crop) == 1:
        crop = torch.cat((torch.zeros_like(crop), crop))

    # Draw cropping amount, proportion to apply to lower end.
    prop = dict(device=x.device, generator=generator)
    batch = x.shape[:1]
    bit = kt.utility.chance(prob, size=batch, **prop)
    a, b = crop
    crop = bit * torch.rand(batch, **prop) * (b - a) + a
    dist = torch.rand(batch, **prop)

    # Treat channels as one.
    x = x.any(dim=1)
    out = torch.zeros_like(x)
    for i, batch in enumerate(x):
        # Mask extent along random axis.
        dim = torch.randint(ndim, size=())
        batch = batch.any(dim=[n for n in range(ndim) if n != dim])
        low, upp = batch.nonzero().aminmax()

        # Distribute cropping proportion between lower and upper end.
        cut = upp.sub(low).add(1) * crop[i]
        add = cut.mul(dist[i]).add(0.5).type(torch.int32)
        sub = cut.add(0.5).type(torch.int32) - add

        # Everything True except along axis.
        ind = [slice(0, s) for s in size]
        ind[dim] = slice(low + add, upp - sub + 1)
        out[i, *ind] = True

    # Restore channel dimension.
    return out.unsqueeze(1)


def affine(x, shift=30, angle=30, scale=0.1, shear=0.1, generator=None):
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
    print(scale)
    if scale.le(0).any():
        raise ValueError(f'scale {scale} includes non-positive values')

    # Sample parameters.
    prop = dict(device=x.device, generator=generator)
    bounds = torch.cat((shift, angle, scale, shear))
    a, b = bounds[0::2], bounds[1::2]
    par = torch.rand(x.size(0), *a.shape, **prop) * (b - a) + a

    # Construct matrix, with double precision.
    prop = dict(dtype=torch.float64, device=x.device)
    splits = (3, 3, 3, 3) if ndim == 3 else (2, 1, 2, 1)
    par = torch.split(par, splits, dim=-1)
    mat = kt.transform.compose_affine(*par, **prop)

    # Apply in centered frame.
    cen = torch.eye(ndim + 1, **prop)
    unc = torch.eye(ndim + 1, **prop)
    cen[:-1, -1] = -0.5 * (torch.as_tensor(x.shape[2:]) - 1)
    unc[:-1, -1] = -cen[:-1, -1]

    return unc.matmul(mat).matmul(cen).type(torch.get_default_dtype())


def warp(x, disp=25, points=16, damp=0.33, steps=0, generator=None):
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
    points = points.type(torch.int32)

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
