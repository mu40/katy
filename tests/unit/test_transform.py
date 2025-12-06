"""Unit tests for transform module."""

import katy as kt
import pytest
import torch


def test_grid():
    """Test index-coordinate grid creation with ij-indexing."""
    size = (4, 9)
    grid = kt.transform.grid(size)
    assert grid.shape == (len(size), *size)

    # Test ij-indexing. Coordinate i should increase along i-axis only.
    x = grid[0]
    y = grid[1]
    for i in (0, 1):
        for j in (0, 1):
            assert x[i, j] == i
            assert y[i, j] == j


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('align', [True, False])
def test_index_to_torch(ndim, align):
    """Test conversion from indices to PyTorch's normalized coordinates."""
    size = [5] * ndim

    # Default normalized PyTorch grid.
    eye = torch.eye(ndim, ndim + 1).unsqueeze(0)
    f = torch.nn.functional.affine_grid
    y = f(eye, size=(1, 1, *size), align_corners=align)

    # Standard index coordinates.
    x = (torch.arange(s, dtype=y.dtype) for s in size)
    x = torch.meshgrid(*x, indexing='ij')
    x = torch.stack(x).view(ndim, -1)

    # Convert indices to Pytorch coordinates.
    mat = kt.transform.index_to_torch(size, align_corners=align)
    x = mat[:-1, :-1] @ x + mat[:-1, -1:]
    x = x.movedim(0, -1).view(1, *size, ndim)

    assert y.allclose(x)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('align', [True, False])
def test_torch_to_index(ndim, align):
    """Test conversion from PyTorch's normalized coordinates to indices."""
    size = (5, 6, 7)[:ndim]
    fw = kt.transform.index_to_torch(size, align_corners=align)
    bw = kt.transform.torch_to_index(size, align_corners=align)
    assert fw.inverse().allclose(bw)


@pytest.mark.parametrize('deg', [1, -1])
def test_compose_rotation_direction_2d(deg):
    """Test the direction of 2D rotations."""
    # In 2D, a rotation by a small positive angle should move the unit vector
    # along x towards positive y and a small negative angle towards negative y.
    vec = torch.tensor((1., 0.))
    deg = torch.tensor(deg)

    x, y = out = kt.transform.compose_rotation(deg) @ vec
    assert torch.linalg.vector_norm(out) == 1
    assert 0 < x < 1
    assert 0 < y.abs() < 1
    assert y.sign() == deg.sign()


@pytest.mark.parametrize('dim', [0, 1, 2])
@pytest.mark.parametrize('deg', [1, -1])
def test_compose_rotation_direction_3d(dim, deg):
    """Test the direction of 3D rotations."""
    # Let i be one of the (x, y, or z) axes of a Cartesian coordinate system.
    # We will call j the axis that comes after i, following the right-hand
    # rule: x -> y, y -> z, z -> x. Similarly, let k be the axis after j.
    deg = torch.tensor(deg)
    ijk = torch.arange(3)

    # A rotating by a small positive angle about i should move the unit vector
    # along j towards positive k. A small negative angle should move it towards
    # negative k. Both should leave the component along i unchanged.
    i = dim
    j = ijk.roll(-1)[i]
    k = ijk.roll(-2)[i]

    # Unit vector along j.
    vec = torch.zeros(3)
    vec[j] = 1

    # Rotation about i.
    ang = torch.zeros(3)
    ang[i] = deg
    out = kt.transform.compose_rotation(ang) @ vec

    assert torch.linalg.vector_norm(out) == 1
    assert out[i] == vec[i]
    assert out[j].sign() > 0
    assert out[k].sign() == deg.sign()


def test_decompose_rotation_3d_degrees():
    """Test if decomposing a 3D rotation matrix recovers degrees."""
    # Degree range [-30, 30] avoids differences from periodicity.
    ang = torch.tensor((-30.1, -5.2, 0, 1, 7.9, 30)).view(2, 3)
    out = kt.transform.compose_rotation(ang)
    out = kt.transform.decompose_rotation(out)
    assert out.allclose(ang)


def test_decompose_rotation_2d_radians():
    """Test if decomposing a 2D rotation matrix recovers radians."""
    ang = torch.pi * torch.tensor((-0.99, -0.8, 0, 0.001, 0.5, 0.6))
    ang = ang.view(3, 2, 1)

    out = kt.transform.compose_rotation(ang, deg=False)
    out = kt.transform.decompose_rotation(out, deg=False)
    assert out.allclose(ang)


@pytest.mark.parametrize('ndim', [2, 3])
def test_compose_affine_values(ndim):
    """Test creation of translation, rotation, scaling, and shear matrices."""
    values = torch.arange(3.) + 7

    # Translation.
    inp = values[:ndim]
    out = torch.eye(ndim + 1)
    out[:ndim, -1] = inp
    assert kt.transform.compose_affine(shift=inp).allclose(out)

    # Rotation.
    inp = values[:1 if ndim == 2 else 3]
    out = torch.eye(ndim + 1)
    out[:ndim, :ndim] = kt.transform.compose_rotation(inp)
    assert kt.transform.compose_affine(angle=inp).allclose(out)

    # Scaling.
    inp = values[:ndim]
    out = torch.eye(ndim + 1)
    out.diagonal()[:ndim] = inp
    assert kt.transform.compose_affine(scale=inp).allclose(out)

    # Shear.
    inp = values[:1 if ndim == 2 else 3]
    out = torch.eye(ndim + 1)
    out[*torch.triu_indices(ndim, ndim, offset=1)] = inp
    assert kt.transform.compose_affine(shear=inp).allclose(out)


@pytest.mark.parametrize('ndim', [2, 3])
def test_compose_affine_broadcasting(ndim):
    """Test batch-size broadcasting for affine-matrix composition."""
    num = 3 if ndim == 3 else 1
    shift = torch.ones(         ndim)
    angle = torch.ones(   4, 6, num)
    scale = torch.ones(2, 1, 6, ndim)
    shear = torch.ones(1, 4, 1, num)

    input = (shift, angle, scale, shear)
    shape = (f.shape[:-1] for f in input)
    shape = torch.broadcast_shapes(*shape)
    shape = (*shape, ndim + 1, ndim + 1)
    assert kt.transform.compose_affine(*input).shape == shape


@pytest.mark.parametrize('ndim', [2, 3])
def test_decompose_affine(ndim):
    """Test if decomposing an affine transform recovers parameters."""
    dtype = torch.float64

    # Low angles avoid differences from periodicity. Use positive scaling.
    num = 3 if ndim == 3 else 1
    shift = torch.tensor((-29, -10, 20.7), dtype=dtype)[:ndim]
    angle = torch.tensor((-30, 3.5, 25.0), dtype=dtype)[:num]
    scale = torch.tensor((0.3, 0.8, 1.55), dtype=dtype)[:ndim]
    shear = torch.tensor((-0.5, 0.001, 1), dtype=dtype)[:num]

    inp = (shift, angle, scale, shear)
    mat = kt.transform.compose_affine(*inp, dtype=dtype)
    out = kt.transform.decompose_affine(mat, dtype=dtype)
    assert out[0].allclose(inp[0])
    assert out[1].allclose(inp[1])
    assert out[2].allclose(inp[2])
    assert out[3].allclose(inp[3])


@pytest.mark.parametrize('ndim', [2, 3])
def test_compose_matrices(ndim):
    """Test composition of matrix transforms."""
    zoom = torch.tensor((*[2.] * ndim, 1)).diag().unsqueeze(0)

    # A single matrix input should be returned as is.
    assert kt.transform.compose(zoom) is zoom

    # Composing matrix should be equivalent to their matrix product.
    out = kt.transform.compose(zoom, zoom)
    assert out.allclose(zoom @ zoom)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('batch', [(), (4,)])
def test_compose_matrices_grid(ndim, batch):
    """Test conversion of matrix transforms to displacements or coordinates."""
    # Matrix and grid.
    zoom = 3.
    mat = torch.tensor((*[zoom] * ndim, 1)).diag()
    mat.expand(*batch, *mat.shape)
    grid = kt.transform.grid(size=[5] * ndim, dtype=mat.dtype)

    # With a grid, a matrix should become a displacement field.
    out = kt.transform.compose(mat, grid=grid)
    assert out.allclose(grid * zoom - grid)

    # The same for several matrix inputs.
    out = kt.transform.compose(mat, mat, grid=grid)
    assert out.allclose(grid * zoom * zoom - grid)

    # If requested, a single matrix should become absolute coordinates.
    out = kt.transform.compose(mat, grid=grid, absolute=True)
    assert out.allclose(grid * zoom)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('batch', [1, 4])
def test_compose_fields(ndim, batch):
    """Test composition of displacement fields."""
    # Data. Input of shape (batch, dimensionality, *space).
    size = [5] * ndim
    disp = torch.ones(batch, ndim, *size)
    grid = kt.transform.grid(size, dtype=disp.dtype)

    # A single field should be returned as is.
    assert kt.transform.compose(disp) is disp

    # N shifts of one should be a shift of N.
    out = kt.transform.compose(disp, disp, disp)
    assert out.allclose(disp * 3)

    # The same with conversion to absolute locations.
    out = kt.transform.compose(disp, disp, absolute=True)
    assert out.allclose(disp * 2 + grid)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('batch', [1, 4])
def test_compose_field_matrix(ndim, batch):
    """Test composition of displacement fields and matrix transforms."""
    size = [6] * ndim
    shift = 13

    # Transforms. Displacement of shape: (batch, ndim, *space).
    disp = torch.ones(batch, ndim, *size)
    mat = torch.eye(ndim + 1)
    mat[:-1, -1] = shift
    mat = mat.unsqueeze(0).expand(batch, -1, -1)

    # Shifts should add up.
    out = kt.transform.compose(disp, mat)
    assert out.allclose(disp + shift)

    # The same for reversed inputs.
    out = kt.transform.compose(mat, -disp)
    assert out.allclose(shift - disp)

    # With a matrix on the right, the grid sets the shape.
    small = tuple(i - 1 for i in size)
    grid = kt.transform.grid(small, dtype=disp.dtype)
    out = kt.transform.compose(disp, mat, grid=grid)
    assert out.shape[1:] == grid.shape


def test_center_matrix_unchanged():
    """Test if centering matrix leaves input unchanged."""
    size = (128, 128)
    ndim = len(size)
    mat = torch.eye(ndim + 1)
    orig = mat.clone()
    kt.transform.center_matrix(size, mat)
    assert mat.equal(orig)


def test_center_matrix_illegal():
    """Test centering matrices of illegal size."""
    size = (3, 3, 3)
    with pytest.raises(ValueError):
        mat = torch.eye(3, 3)
        kt.transform.center_matrix(size, mat)

    with pytest.raises(ValueError):
        mat = torch.eye(3, 4)
        kt.transform.center_matrix(size, mat)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('batch', [(), (2,), (2, 1)])
def test_center_matrix_values(ndim, batch):
    """Test centering matrices in 2D and 3D, with different batch sizes."""
    size = torch.tensor(256).expand(ndim)
    ang = 7 + torch.arange(3 if ndim == 3 else 1).expand(*batch, -1)
    inp = kt.transform.compose_affine(angle=ang)

    cen = torch.eye(ndim + 1)
    unc = torch.eye(ndim + 1)
    cen[:-1, -1] = -0.5 * (size - 1)
    unc[:-1, -1] = -cen[:-1, -1]
    expected = unc @ inp @ cen
    assert kt.transform.center_matrix(size, inp).allclose(expected)


def test_jacobian_unchanged():
    """Test if computing Jacobian leaves input unchanged."""
    # Input of shape: batch, channel, space.
    size = (3, 3, 3)
    ndim = len(size)
    inp = torch.ones(1, ndim, *size)

    orig = inp.clone()
    kt.transform.jacobian(inp, det=torch.tensor(False))
    assert inp.equal(orig)


def test_jacobian_illegal_shape():
    """Test computing Jacobian with illegal field shape."""
    size = (3, 3, 3)
    ndim = len(size)
    field = torch.ones(1, ndim + 1, *size)

    with pytest.raises(ValueError):
        kt.transform.jacobian(field)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('is_disp', [True, False])
def test_jacobian_values(ndim, is_disp):
    """Test computing Jacobian determinants."""
    batch = 5
    size = torch.tensor(6).expand(ndim)

    # Determinant of affine transform, broadcastable to field (batch, *size).
    zoom = 1.5
    mat = torch.tensor((*[zoom] * ndim, 1)).diag().expand(batch, -1, -1)
    det = torch.tensor(zoom).pow(ndim).expand(batch, *[1] * ndim)

    # Recover same determinants from displacement or deformation fields.
    grid = kt.transform.grid(size)
    inp = kt.transform.compose(mat, grid=grid, absolute=not is_disp)
    out = kt.transform.jacobian(inp, is_disp=is_disp)
    assert out.allclose(det)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('batch', [(), (3, 4)])
def test_fit_matrix_trivial(ndim, batch):
    """Test output shape and dtype when fitting matrix transforms."""
    # Expect a square matrix of size (N + 1) in N dimensions, of default type.
    x = torch.ones(*batch, 5, ndim)
    out = kt.transform.fit_matrix(x, x, ridge=1e-4)
    assert out.shape == (*batch, ndim + 1, ndim + 1)
    assert out.dtype == torch.get_default_dtype()


def test_fit_matrix_scale():
    """Test if matrix fit recovers scaling, in 3D."""
    # Inputs.
    ndim = 3
    scale = 2
    x = kt.transform.grid(size=[3] * ndim, dim=-1).view(-1, ndim)
    y = scale * x

    expected = (*[scale] * ndim, 1)
    expected = torch.tensor(expected, dtype=torch.get_default_dtype()).diag()
    out = kt.transform.fit_matrix(x, y, ridge=0)
    assert out.allclose(expected)


def test_fit_matrix_shift():
    """Test if matrix fit recovers translation, in 2D."""
    # Inputs.
    ndim = 2
    shift = 7
    x = kt.transform.grid(size=[3] * ndim, dim=-1).view(-1, ndim)
    y = x + shift

    expected = torch.eye(ndim + 1)
    expected[:-1, -1] = shift
    out = kt.transform.fit_matrix(x, y, ridge=0)
    assert out.allclose(expected)


def test_fit_matrix_weighted():
    """Test matrix fit with down-weighted outlier."""
    # Inputs.
    ndim = 2
    shift = 5
    x = kt.transform.grid(size=[4] * ndim, dim=-1).view(-1, ndim)
    y = x + shift

    # Outlier with reduced weight.
    y[-1, :] = 100
    weights = torch.ones(x.shape[0])
    weights[-1] = 0

    expected = torch.eye(ndim + 1)
    expected[:-1, -1] = shift
    out = kt.transform.fit_matrix(x, y, weights=weights, ridge=0)
    assert out.allclose(expected)
