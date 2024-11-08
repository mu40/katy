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
    grid = torch.meshgrid(*grid, indexing='ij')
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
    """Interpolate an N-dimensional tensor at new locations.

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
    """Apply an N-dimensional spatial transform to a tensor.

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


def compose_rotation(angle, deg=True):
    """Compose an N-dimensional rotation matrices from angles.

    The function composes intrinsic 2D or 3D rotation matrices in a
    right-handed "ij-indexing" space. That is, rotations apply in the
    body-centered frame of reference. With your right thumb indicating the
    rotation axis, a positive angles describes a rotation in the direction
    that the other fingers point to.

    Parameters
    ----------
    angle : (..., M) torch.Tensor
        Rotation angles of any batch dimensions. M is 1 in 2D and 3 in 3D.
    deg : bool, optional
        Interpret the rotation angles as degrees instead of radians.

    Raises
    ------
    ValueError
        If M is not 1 or 3.

    Returns
    -------
    (..., N, N) torch.Tensor
        Rotation matrices.

    """
    angle = torch.as_tensor(angle, dtype=torch.float64)
    if angle.ndim == 0:
        angle = angle.view(1)

    if deg:
        angle = angle.deg2rad()
    c = torch.cos(angle)
    s = torch.sin(angle)

    if angle.size(-1) == 1:
        row_1 = torch.cat((c, -s), dim=-1)
        row_2 = torch.cat((s, c), dim=-1)
        return torch.stack((row_1, row_2), dim=-2)

    elif angle.size(-1) == 3:
        one = torch.tensor(1).expand(angle.shape[:-1])
        zero = torch.tensor(0).expand(angle.shape[:-1])

        row_1 = torch.stack((one, zero, zero), dim=-1)
        row_2 = torch.stack((zero, c[..., 0], -s[..., 0]), dim=-1)
        row_3 = torch.stack((zero, s[..., 0], c[..., 0]), dim=-1)
        mat_x = torch.stack((row_1, row_2, row_3), dim=-2)

        row_1 = torch.stack((c[..., 1], zero, s[..., 1]), dim=-1)
        row_2 = torch.stack((zero, one, zero), dim=-1)
        row_3 = torch.stack((-s[..., 1], zero, c[..., 1]), dim=-1)
        mat_y = torch.stack((row_1, row_2, row_3), dim=-2)

        row_1 = torch.stack((c[..., 2], -s[..., 2], zero), dim=-1)
        row_2 = torch.stack((s[..., 2], c[..., 2], zero), dim=-1)
        row_3 = torch.stack((zero, zero, one), dim=-1)
        mat_z = torch.stack((row_1, row_2, row_3), dim=-2)

        return mat_x @ mat_y @ mat_z

    raise ValueError(f'Expected 1 or 3 angles, not {angle.size(-1)}')


def decompose_rotation(mat, deg=True):
    """Decompose an N-dimensional rotation matrix into Euler angles.

    We decompose right-handed intrinsic rotations R = X @ Y @ Z, where X, Y,
    and Z are matrices describing rotations about the x, y, and z-axis,
    respectively. Labeling these axes 1-3, we want to decompose the matrix

            [            c2*c3,             -c2*s3,      s2]
        R = [ s1*s2*c3 + c1*s3,  -s1*s2*s3 + c1*c3,  -s1*c2],
            [-c1*s2*c3 + s1*s3,   c1*s2*s3 + s1*c3,   c1*c2]

    where si and ci are sine and cosine of the rotation angle about i. When
    the rotation angle about the y-axis is 90 or -90 degrees, we run into
    "gimbal lock". The system loses one degree of freedom, the solution is no
    longer unique. In this case, we set angle 1 to zero and solve for angle 2.

    Parameters
    ----------
    mat : (..., N, N) torch.Tensor
        Rotation matrices of any batch dimensions, where N is 2 or 3.
    deg : bool, optional
        Return rotation angles in degrees instead of radians.

    Raises
    ------
    ValueError
        For inputs that are not square matrices or if N is not 2 or 3.

    Returns
    -------
    (..., M) torch.Tensor
        Rotation angles, where M is 3 in 3D and 1 in 2D.

    """
    mat = torch.as_tensor(mat, dtype=torch.float64)
    dim = mat.size(-1)

    if mat.dim() < 2 or mat.size(-2) != dim or dim not in (2, 3):
        raise ValueError(f'Size {mat.shape} is not (..., 2, 2) or (..., 3, 3)')

    if dim == 2:
        y = mat[..., 1, 0]
        x = mat[..., 0, 0]
        ang = torch.atan2(y, x).unsqueeze(-1)

    elif dim == 3:
        ang2 = torch.asin(mat[..., 0, 2])

        # Initialize, check if gimbal lock for each matrix. Reduce relative
        # tolerance to improve precision.
        ang1 = torch.zeros_like(ang2)
        ang3 = torch.zeros_like(ang2)
        lock = 0.5 * torch.tensor(torch.pi, dtype=mat.dtype)
        ind = ang2.abs().isclose(lock, rtol=1e-8)

        # Case abs(ang2) == 90 deg. Keep ang1 zero, as solution is not unique.
        y = mat[..., 1, 0][ind]
        x = mat[..., 1, 1][ind]
        ang3[ind] = torch.atan2(y, x)

        # Case abs(ang2) != 90 deg.
        ind = ind.logical_not()
        c2 = torch.cos(ang2[ind])
        y = -mat[..., 1, 2][ind].div(c2)
        x = mat[..., 2, 2][ind].div(c2)
        ang1[ind] = torch.atan2(y, x)

        # Other angle.
        y = -mat[..., 0, 1][ind].div(c2)
        x = mat[..., 0, 0][ind].div(c2)
        ang3[ind] = torch.atan2(y, x)

        ang = torch.stack((ang1, ang2, ang3), dim=-1)

    return ang.rad2deg() if deg else ang


def compose_affine(
    shift=None,
    angle=None,
    scale=None,
    shear=None,
    deg=True,
    device=None,
):
    """Compose N-dimensional matrix transforms.

    The function composes 2D or 3D matrix transforms from parameters defining
    translation, rotation, scaling, and shear. Specify at least one parameter.
    Parameters can have any batch dimensions, but they must be broadcastable.

    Parameters
    ----------
    shift : (..., N) torch.Tensor, optional
        Translation vector.
    angle : (..., M) torch.Tensor, optional
        Rotation angles. M is 3 in 3D and 1 in 2D.
    scale : (..., N) torch.Tensor, optional
        Scaling factors.
    shear : (..., M) torch.Tensor, optional
        Shearing factors. M is 3 in 3D and 1 in 2D.
    deg : bool, optional
        Interpret the rotation angles as degrees.
    device : torch.device, optional
        Device of the returned tensor.

    Raises
    ------
    ValueError
        If all parameters are None or their shapes are incompatible.

    Returns
    -------
    (..., N + 1, N + 1) torch.Tensor
        Matrix transforms.

    """
    prop = dict(dtype=torch.float64, device=device)

    def conform(x, name, sizes, ndim=None):
        # Conform to tensor of at least one dimension.
        x = torch.as_tensor(x, **prop)
        if x.ndim == 0:
            x = x.view(1)

        # Validate size of trailing dimension.
        n = 3 if x.size(-1) == 3 else 2
        if ndim is not None and n != ndim:
            raise ValueError(f'cannot mix {ndim}D with {n}D {name} size')
        if x.size(-1) not in sizes:
            raise ValueError(f'{name} size {x.size(-1)} is not in {sizes}')

        return x, n

    # Initialization. Start working with N-by-N matrices. Matrix
    # multiplications will broadcast batch dimensions or error out.
    ndim = None
    out = None

    if angle is not None:
        angle, ndim = conform(angle, 'angle', sizes=(1, 3), ndim=ndim)
        out = compose_rotation(angle, deg)

    if scale is not None:
        scale, ndim = conform(scale, 'scale', sizes=(2, 3), ndim=ndim)
        mat = torch.diag_embed(scale)
        out = mat if out is None else out @ mat

    if shear is not None:
        shear, ndim = conform(shear, 'shear', sizes=(1, 3), ndim=ndim)
        i, j = torch.triu_indices(ndim, ndim, offset=1)
        mat = torch.ones(size=(*shear.shape[:-1], ndim), **prop).diag_embed()
        mat[..., i, j] = shear
        out = mat if out is None else out @ mat

    # Process shifts last.
    if shift is not None:
        shift, ndim = conform(shift, 'shift', sizes=(2, 3), ndim=ndim)

        # Broadcast batch dimensions of shifts and any existing transform.
        batch = shift.shape[:-1]
        if out is not None:
            batch = torch.broadcast_shapes(batch, out.shape[:-2])

        # Copy into identity matrix of broadcasted batch size. Will require
        # expanding first. Cheaper than multiplying or adding full matrices.
        full = torch.ones(size=(*batch, ndim + 1), **prop).diag_embed()
        full[..., :-1, -1] = shift.expand(*batch, -1)
        if out is not None:
            full[..., :-1, :-1] = out.expand(*batch, -1, -1)

        out = full

    if ndim is None:
        raise ValueError('expected at least one specified parameter')

    return out.type(torch.get_default_dtype())
