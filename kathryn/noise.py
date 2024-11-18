"""Noise module."""


import torch
import itertools
import kathryn as kt


def perlin(size, points=2, batch=None, device=None):
    """Generate Perlin noise in N dimensions.

    Inspired by https://adrianb.io/2014/08/09/perlinnoise.html.

    Parameters
    ----------
    size : (N,) sequence of int or torch.Tensor
        Spatial output size.
    points : int or (N,) torch.Tensor, optional
        Number of control points in [2, size). The lower, the smoother. Pass
        1 value to use for all axes or N values - one for each axis.
    batch : sequence of int or torch.Tensor, optional
        Batch size.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    (*batch, *size) torch.Tensor
        Perlin noise, roughly varying between -0.5 and 0.5.

    """
    # Inputs.
    prop = dict(device=device)
    size = torch.as_tensor(size, **prop)
    ndim = size.numel()
    points = torch.as_tensor(points, **prop).ravel().expand(ndim)
    if points.lt(2).any() or points.ge(size).any():
        raise ValueError(f'controls points {points} is not all in [2, size)')
    if batch is None:
        batch = []

    # Grid of gradient directions at integral coordinate locations.
    points = points.ravel().expand(ndim)
    batch = torch.as_tensor(batch, device=device).ravel()
    grad = torch.rand(ndim, *batch, *points, device=device).mul(2).sub(1)
    grad = grad.view(ndim, *batch, -1)

    # Output grid, subsampling between gradient coordinates.
    grid = (torch.linspace(0, c - 1, s, **prop) for c, s in zip(points, size))
    grid = torch.meshgrid(*grid, indexing='ij')
    grid = torch.stack(grid)

    # Integer coordinates of closest corner points.
    x0 = grid.type(torch.int32)
    for i in range(ndim):
        x0[i] = x0[i].clamp(0, points[i] - 2)
    x1 = x0 + 1
    x01 = (x0, x1)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Grid-point distance to corner points and difference vectors from corners
    # to grid points. Interpolation weights are smoothed inverse of distance.
    dx1 = x1 - grid
    dx0 = 1 - dx1
    diff = (dx0, -dx1)
    weights = (fade(dx1), fade(dx0))

    # Aggregate values each corner point. Avoid stacking, as it is slow.
    out = 0
    for corner in itertools.product((0, 1), repeat=ndim):
        # Sample gradients at corner point.
        ind = [x01[c][i] for i, c in enumerate(corner)]
        ind = kt.index.sub2ind(points, *ind)
        vec = grad[..., ind]

        # Dot product of difference and gradient, weight without stacking.
        dot = 0
        weight = 1
        for i, c in enumerate(corner):
            dot += diff[c][i] * vec[i, ...]
            weight *= weights[c][i]
        out += weight * dot

    return out


def octaves(size, points, persistence, batch=None, device=None):
    """Generate and mix M isotropic octaves of Perlin noise in N dimensions.

    Inspired by https://www.arendpeter.com/Perlin_Noise.html.

    Parameters
    ----------
    size : (N,) sequence of int or torch.Tensor
        Spatial output size.
    points : (M,) torch.Tensor
        Number of control points in [2, size). The lower, the smoother.
    persistence : float or torch.Tensor
        Weighting of higher frequencies, in (0, 1]. Must broadcast to `batch`.
    batch : sequence of int or torch.Tensor, optional
        Batch size.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    (*batch, *size) torch.Tensor
        Octaves of Perlin noise, normalized into [0, 1].

    """
    # Inputs.
    prop = dict(device=device)
    size = torch.as_tensor(size, **prop).ravel()
    points = torch.as_tensor(points, **prop).ravel()
    persistence = torch.as_tensor(persistence, **prop)
    if batch is None:
        batch = []
    batch = torch.as_tensor(batch, **prop).ravel()

    # Make sure that multi-persistence inputs have to broadcast to batch only.
    if persistence.le(0).any() or persistence.gt(1).any():
        raise ValueError(f'persistence {persistence} is not in (0, 1]')
    persistence = persistence.view(*persistence.shape, *(1,) * size.numel())

    # With the definitions of `freq = 2 ** i` and `amp = pers ** i`, from the
    # reference, we have `i = log2(freq)` and thus `amp = pers ** log2(freq)`.
    out = 0
    for p in points:
        amp = persistence ** torch.log2(p)
        out += amp * perlin(size, points=p, batch=batch, **prop)

    # Batch-wise normalization.
    dim = tuple(range(len(batch), out.ndim))
    out -= out.amin(dim, keepdim=True)
    out /= out.amax(dim, keepdim=True)

    return out
