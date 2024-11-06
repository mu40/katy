"""Coordinate transformation."""


import torch


def grid(size, dim=0, **kwargs):
    """Create a grid of N-dimensional index coordinates.

    Parameters
    ----------
    size : (N,) torch.Tensor
       Output shape.
    dim : int or None, optional
        Stack output along this dimension.

    Returns
    -------
    torch.Tensor or tuple of (*size,) torch.Tensor
        Index coordinates, stacked along `dim` if provided.

    """
    size = torch.as_tensor(size, dtype=torch.int32).ravel()
    grid = (torch.arange(s, **kwargs) for s in size)
    grid = torch.meshgrid(*grid)
    return grid if dim is None else torch.stack(grid, dim=dim)


def index_to_torch(size, align_corners=False):
    """Construct a matrix transforming index to "PyTorch" coordinates.

    Constructs a transform from indices to the normalized space used by
    `torch.nn.functional.grid_sample` and `torch.nn.functional.affine_grid`.

    Parameters
    ----------
    size : (N,) torch.Tensor
        Shape of the N-dimensional image space.
    align_corners : bool, optional
        Normalization type used for PyTorch sampling functions.

    Returns
    -------
    (N + 1, N + 1) torch.Tensor
        Matrix transform.

    """
    size = torch.as_tensor(size).ravel()
    ndim = size.numel()

    # Coordinates indicate the location of pixel centers. With the default
    # `align_corners=False`, values -1 and 1 refer to the outmost pixel
    # borders instead of their centers. For indices, this means values 0 and
    # N - 1 refer to the borders instead of the centers. That is, we have to
    # move the first pixel center from 0 to 0.5, and change the FOV width
    # between oustmost pixel borders from N to N - 1.
    if not align_corners:
        shift = torch.eye(ndim + 1)
        shift[:-1, -1] = 0.5
        scale = torch.cat(((size - 1) / size, torch.tensor([1])))
        align = torch.diag(scale) @ shift

    # Normalize indices from [0, N - 1] into [-1, 1].
    norm = torch.cat((2 / (size - 1), torch.tensor([1])))
    norm = torch.diag(norm)
    norm[:-1, -1] = -1

    # Reverse the dimensions that serves to index into spactial axes.
    swap = torch.eye(ndim).flipud()
    swap = torch.block_diag(swap, torch.tensor([1]))

    return swap @ norm if align_corners else swap @ norm @ align


def torch_to_index(*args, **kwargs):
    """Construct a matrix transforming "PyTorch" to index coordinates.

    See `index_to_torch` for details.

    """
    return torch.inverse(index_to_torch(*args, **kwargs))


def integrate(x, steps, grid=None):
    """Integrate a stationary N-dimensional vector field.

    Implements the "scaling and squaring" algorithm.

    You can pass an existing mesh grid, for efficiency. The shape of the grid
    must match the shape of a displacement field. For matrices, it controls
    the spatial shape of the output, which defaults to the input shape.

    Parameters
    ----------
    x : (B, N, *size) torch.Tensor
        Displacement vector field of `N`-element spatial shape `size`.
    steps : int
        Number of integration steps.
    grid : (N, *size) torch.Tensor, optional
       Index coordinate grid.

    Returns
    -------
    (B, N, *size) torch.Tensor
        Integral of the vector field.

    """
    x = torch.as_tensor(x)
    if steps == 0:
        return x

    if grid is None:
        grid = grid(x.shape[2:], device=x.device)

    # Avoid in-place addition for gradients. Re-use border values for
    # extrapolation instead of zeros, to avoid steep cliffs.
    x = x / (2 ** steps)
    for _ in range(steps):
        x = x + transform(x, x, grid=grid, padding='border')

    return x


def interpolate(x, points, method='linear', padding='zeros'):
    """Interpolate an N-dimensional (ND) tensor at new locations.

    The function currently supports 2D and 3D inputs. For interpolation,
    it uses `torch.nn.functional.grid_sample` but expects zero-based indices
    ("ij-indexing") instead of coordinates normalized into [-1, 1].

    Parameters
    ----------
    x : (B, C, *size_in) torch.Tensor
        Gridded input data of spatial `N`-element shape `size_in`.
    points : (..., N, *size_out) torch.Tensor
       New index coordinates to sample the input data at. The batch dimension
       has to broadcast to `B`, `size_out` be of length `N`.
    method : str, optional
        Interpolation method, 'nearest' or 'linear'.
    padding : str, optional
        Extrapolation method: 'zeros', 'border', or 'reflection'.

    Returns
    -------
    (B, C, *size_out) torch.Tensor
        Interpolated values.

    """
    # Inputs.
    x = torch.as_tensor(x)
    points = torch.as_tensor(points)

    # Convert to PyTorch's normalized coordinates. Using `align_corners=True`
    # should be slightly more efficient. The result will be identical here.
    align = True
    conv = index_to_torch(size=x.shape[2:], align_corners=align)

    # Broadcast to data batch size at the very end to convert only P batches
    # of coordinates, where P is 1 or the batch size of the coordinates (if
    # present). Reshape to [P, N, voxels], convert, reshape to [B, N, *space].
    # Finally, move the dimension indexing into spatial axes to end.
    ndim = x.dim() - 2
    batch = 1 if points.ndim < x.ndim else points.size(0)
    tmp = points.expand(batch, ndim, -1)
    tmp = conv[:ndim, :-1] @ tmp + conv[:ndim, -1:]
    points = tmp.expand(x.shape[0], *points.shape[-ndim - 1:]).movedim(1, -1)

    mode = 'bilinear' if method == 'linear' else method
    return torch.nn.functional.grid_sample(x, points, mode, padding, align)


def transform(x, trans, grid=None, method='linear', padding='zeros'):
    """Apply an N-dimensional (ND) spatial transform to a tensor.

    The function currently supports 2D and 3D inputs. For interpolation,
    it uses `torch.nn.functional.grid_sample` but expects index coordinates
    ("ij-indexing") instead of coordinates normalized into [-1, 1].

    Parameters
    ----------
    x : (B, C, *size_in) torch.Tensor
        Input data of spatial `N`-element shape `size`.
    trans : (..., N, *size_out) or (..., N + 1, N + 1) torch.Tensor
        Displacement field or matrix transform. Batch must broadcast to `x`.
    grid : (N, *size_out) torch.Tensor, optional
       Index coordinate grid. For efficiency and controlling `size_out`.
    method : str, optional
        Interpolation method, 'nearest' or 'linear'.
    padding : str, optional
        Extrapolation method: 'zeros', 'border', or 'reflection'.

    Returns
    -------
    (B, C, *size_out) torch.Tensor
        Transformed tensor. `size_out` defaults to `size_in` for matrix
        transforms passed without a grid.

    """
    # Inputs.
    x = torch.as_tensor(x)
    trans = torch.as_tensor(trans)

    # Grid.
    size = x.shape[2:]
    if grid is None:
        grid = grid(size, device=x.device)

    # For matrices, reshape grid to [P, N, voxels], where P is 1 or the batch
    # size of the transform (if present) but not B, to keep the number of
    # batches that undergo coordinate conversion in `interpolate` low.
    ndim = size.numel()
    if trans.size(-1) == trans.size(-2) == ndim + 1:
        batch = 1 if trans.ndim == 2 else trans.size(0)
        points = grid.expand(batch, ndim, -1)
        points = trans[..., :ndim, :-1] @ points + trans[..., :ndim, -1:]
        points = points.expand(points.size(0), *grid.shape)

    # Add displacement.
    else:
        points = grid + trans

    return interpolate(x, points, method, padding)
