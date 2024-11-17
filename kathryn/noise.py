"""Noise module."""


import torch
import itertools
import kathryn as kt


def perlin(size, cells=1, batch=None, device=None):
    """Generate Perlin noise in N dimensions.

    Inspired by https://adrianb.io/2014/08/09/perlinnoise.html.

    Parameters
    ----------
    size : (N,) sequence of int or torch.Tensor
        Spatial output size.
    cells : sequence of int or torch.Tensor, optional
        Number of gradient-grid cells in [1, size). The fewer, the smoother.
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
    cells = torch.as_tensor(cells, **prop).ravel().expand(ndim)
    if cells.lt(1).any() or cells.ge(size).any():
        raise ValueError(f'cells {cells} are not all in [1, size)')
    if batch is None:
        batch = []

    # Grid of gradient directions at integral coordinate locations. Make
    # `cells` the number of gradient-grid points rather than cells.
    cells = cells.add(1).ravel().expand(ndim)
    batch = torch.as_tensor(batch, device=device).ravel()
    grad = torch.rand(ndim, *batch, *cells, device=device).mul(2).sub(1)
    grad = grad.view(ndim, *batch, -1)

    # Output grid, subsampling between gradient coordinates.
    grid = (torch.linspace(0, c - 1, s, **prop) for c, s in zip(cells, size))
    grid = torch.meshgrid(*grid, indexing='ij')
    grid = torch.stack(grid)

    # Integer coordinates of closest corner points.
    x0 = grid.type(torch.int32)
    for i in range(ndim):
        x0[i] = x0[i].clamp(0, cells[i] - 2)
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
        ind = kt.index.sub2ind(cells, *ind)
        vec = grad[..., ind]

        # Dot product of difference and gradient, weight without stacking.
        dot = 0
        weight = 1
        for i, c in enumerate(corner):
            dot += diff[c][i] * vec[i, ...]
            weight *= weights[c][i]
        out += weight * dot

    return out


def octaves(size, freq, pers, batch=None, device=None):
    """Generate and mix octaves of Perlin noise in N dimensions.

    Inspired by https://www.arendpeter.com/Perlin_Noise.html.

    Parameters
    ----------
    size : (N,) sequence of int or torch.Tensor
        Spatial output size.
    freq : sequence of int or torch.Tensor
        Perlin-noise frequency levels to add. Will use `2 ** freq` cells, which
        must be less than `size`, along any dimension.
    pers : float or torch.Tensor
        Persistence in (0, 1]. Higher values emphasize higher frequencies.
        Must broadcast to batch shape.
    batch : sequence of int or torch.Tensor, optional
        Batch size.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    (*batch, *size) torch.Tensor
        Octaves of Perlin noise, normalized into [0, 1].

    """
    prop = dict(device=device)
    size = torch.as_tensor(size, **prop).ravel()
    freq = torch.as_tensor(freq, **prop).ravel()
    pers = torch.as_tensor(pers, **prop)
    if pers.le(0).any() or pers.gt(1).any():
        raise ValueError(f'persistence {pers} is not in (0, 1]')

    # Make sure that multi-persistence inputs have to broadcast to batch only.
    pers = pers.view(*pers.shape, *(1,) * size.numel())

    out = 0
    for f in freq:
        out += perlin(size, cells=2 ** f, batch=batch, **prop) * pers ** f

    # Batch-wise normalization.
    dim = None if batch is None else tuple(range(len(batch), out.ndim))
    out -= out.amin(dim, keepdim=True)
    out /= out.amax(dim, keepdim=True)

    return out
