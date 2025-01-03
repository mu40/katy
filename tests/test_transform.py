"""Tests for transform module."""


import torch
import pytest
import katy as kt


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


def test_index_to_torch():
    """Test conversion from indices to PyTorch's normalized coordinates."""
    shape = (17, 32, 63)

    # Test 2D and 3D.
    for dim in (2, 3):
        size = shape[:dim]

        for align in (True, False):
            # Default normalized PyTorch grid.
            eye = torch.eye(dim, dim + 1).unsqueeze(0)
            y = torch.nn.functional.affine_grid(
                eye,
                size=(1, 1, *size),
                align_corners=align,
            )

            # Standard index coordinates.
            x = (torch.arange(s, dtype=y.dtype) for s in size)
            x = torch.meshgrid(*x, indexing='ij')
            x = torch.stack(x).view(dim, -1)

            # Convert indices to Pytorch coordinates.
            mat = kt.transform.index_to_torch(size, align_corners=align)
            x = mat[:-1, :-1] @ x + mat[:-1, -1:]
            x = x.movedim(0, -1).view(1, *size, dim)

            assert y.allclose(x, atol=1e-6)


def test_torch_to_index():
    """Test conversion from PyTorch's normalized coordinates to indices."""
    size = (17, 32, 63)

    # Test 2D and 3D.
    for dim in (2, 3):
        for align in (True, False):
            fw = kt.transform.index_to_torch(size[:dim], align_corners=align)
            bw = kt.transform.torch_to_index(size[:dim], align_corners=align)
            assert fw.inverse().allclose(bw)


def test_compose_rotation_direction_2d():
    """Test the direction of 2D rotations.

    In 2D, a rotation by a small positive angle should move the unit vector
    along x towards positive y. Conversely, a small negative angle should move
    it towards negative y.

    """
    degrees = torch.tensor((1, -1))
    vec = torch.tensor((1., 0.))

    for deg in degrees:
        x, y = out = kt.transform.compose_rotation(deg) @ vec
        assert torch.linalg.vector_norm(out) == 1
        assert 0 < x < 1
        assert 0 < y.abs() < 1
        assert y.sign() == deg.sign()


def test_compose_rotation_direction_3d():
    """Test the direction of 3D rotations.

    Let i be one of the (x, y, or z) axes of a Cartesian coordinate system. We
    will call j the axis that comes "next" after i, following the right-hand
    rule: x -> y, y -> z, z -> x. Similarly, let k be the axis "next" after j.

    A rotating by a small positive angle about i should move the unit vector
    along j towards positive k. Conversely, a small negative angle should move
    it towards negative k. Both should leave the component along i unchanged.

    """
    degrees = torch.tensor((1, -1))
    dim = torch.arange(3)

    for i in dim:
        j = dim.roll(-1)[i]
        k = dim.roll(-2)[i]

        for deg in degrees:
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
    """Test if decomposing a 3D rotation matrix recovers angles."""
    # Degree range [-30, 30] avoids differences from periodicity.
    ang = torch.rand(130, 40, 3)
    ang = 30 * (2 * ang - 1)

    out = kt.transform.compose_rotation(ang)
    out = kt.transform.decompose_rotation(out)
    assert out.allclose(ang)


def test_decompose_rotation_2d_radians():
    """Test if decomposing a 2D rotation matrix recovers angles."""
    ang = torch.rand(130, 40, 1)
    ang = torch.pi * (2 * ang - 1)

    out = kt.transform.compose_rotation(ang, deg=False)
    out = kt.transform.decompose_rotation(out, deg=False)
    assert out.allclose(ang)


def test_compose_affine_values():
    """Test creation of translation, rotation, scaling, and shear matrices."""
    values = torch.arange(3, dtype=torch.get_default_dtype()) + 7

    # Test 2D and 3D.
    for dim in (2, 3):
        # Translation.
        inp = values[:dim]
        out = torch.eye(dim + 1)
        out[:dim, -1] = inp
        assert kt.transform.compose_affine(shift=inp).allclose(out)

        # Rotation.
        inp = values[:1 if dim == 2 else 3]
        out = torch.eye(dim + 1)
        out[:dim, :dim] = kt.transform.compose_rotation(inp)
        assert kt.transform.compose_affine(angle=inp).allclose(out)

        # Scaling.
        inp = values[:dim]
        out = torch.eye(dim + 1)
        out.diagonal()[:dim] = inp
        assert kt.transform.compose_affine(scale=inp).allclose(out)

        # Shear.
        inp = values[:1 if dim == 2 else 3]
        out = torch.eye(dim + 1)
        out[*torch.triu_indices(dim, dim, offset=1)] = inp
        assert kt.transform.compose_affine(shear=inp).allclose(out)


def test_compose_affine_broadcasting():
    """Test batch-size broadcasting for affine-matrix composition."""
    # Test 2D and 3D.
    for dim in (2, 3):
        num = 3 if dim == 3 else 1
        shift = torch.ones(         dim)
        angle = torch.ones(   4, 6, num)
        scale = torch.ones(2, 1, 6, dim)
        shear = torch.ones(1, 4, 1, num)

        input = (shift, angle, scale, shear)
        shape = (f.shape[:-1] for f in input)
        shape = torch.broadcast_shapes(*shape)
        shape = (*shape, dim + 1, dim + 1)
        assert kt.transform.compose_affine(*input).shape == shape


def test_decompose_affine():
    """Test if decomposing an affine transform recovers parameters."""
    batch = (8, 8)
    dtype = torch.float64

    for dim in (2, 3):
        # Low angles avoid differences from periodicity. Use positive scaling.
        num = 3 if dim == 3 else 1
        shift = torch.rand(*batch, dim, dtype=dtype).sub(0.5).mul(2) * 30
        angle = torch.rand(*batch, num, dtype=dtype).sub(0.5).mul(2) * 30
        scale = torch.rand(*batch, dim, dtype=dtype).add(0.5)
        shear = torch.rand(*batch, num, dtype=dtype).sub(0.5)

        inp = (shift, angle, scale, shear)
        mat = kt.transform.compose_affine(*inp, dtype=dtype)
        out = kt.transform.decompose_affine(mat, dtype=dtype)

        assert out[0].allclose(inp[0])
        assert out[1].allclose(inp[1])
        assert out[2].allclose(inp[2])
        assert out[3].allclose(inp[3])


def test_compose_matrices():
    """Test composition of matrix transforms."""
    zoom = 2.

    for dim in (2, 3):
        mat = torch.tensor((*[zoom] * dim, 1)).diag().unsqueeze(0)

        # A single matrix input should be returned as is.
        out = kt.transform.compose(mat)
        assert out is mat

        # Composing matrices should be equivalent to their matrix product.
        out = kt.transform.compose(mat, mat)
        assert out.allclose(mat @ mat)


def test_compose_matrices_grid():
    """Test conversion of matrix transforms to displacements or coordinates."""
    zoom = 3.
    size = (8, 8, 8)

    for dim in (2, 3):
        for batch in ([1], [4]):
            # Data.
            mat = torch.tensor((*[zoom] * dim, 1)).diag()
            mat.expand(*batch, *mat.shape)
            grid = kt.transform.grid(size[:dim], dtype=mat.dtype)

            # With a grid, a matrix should become a displacement field.
            out = kt.transform.compose(mat, grid=grid)
            assert out.allclose(grid * zoom - grid)

            # The same for several matrix inputs.
            out = kt.transform.compose(mat, mat, grid=grid)
            assert out.allclose(grid * zoom * zoom - grid)

            # If requested, a single matrix should become absolute coordinates.
            out = kt.transform.compose(mat, grid=grid, absolute=True)
            assert out.allclose(grid * zoom)


def test_compose_fields():
    """Test composition of displacement fields."""
    size = (8, 8, 8)

    for dim in (2, 3):
        for batch in (1, 4):
            # Data. Input of shape (batch, dimensionality, *space).
            disp = torch.ones(batch, dim, *size[:dim])
            grid = kt.transform.grid(size[:dim], dtype=disp.dtype)

            # A single field should be returned as is.
            out = kt.transform.compose(disp)
            assert out is disp

            # N shifts of one should be N.
            out = kt.transform.compose(disp, disp, disp)
            assert out.allclose(disp * 3)

            # The same with conversion to absolute locations.
            out = kt.transform.compose(disp, disp, absolute=True)
            assert out.allclose(disp * 2 + grid)


def test_compose_field_matrix():
    """Test composition of displacement fields and matrix transforms."""
    size = (8, 8, 8)
    shift = 13

    for dim in (2, 3):
        for batch in (1, 4):
            # Transforms. Displacement of shape: (batch, dim, *space).
            disp = torch.ones(batch, dim, *size[:dim])
            mat = torch.eye(dim + 1)
            mat[:-1, -1] = shift
            mat = mat.unsqueeze(0).expand(batch, -1, -1)

            # Shifts should add up.
            out = kt.transform.compose(disp, mat)
            assert out.allclose(disp + shift)

            # The same for reversed inputs.
            out = kt.transform.compose(mat, -disp)
            assert out.allclose(shift - disp)

            # With a matrix on the right, the grid sets the shape.
            small = tuple(i - 1 for i in size[:dim])
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
    assert mat.eq(orig).all()


def test_center_matrix_incompatible():
    """Test centering matrices of incompatible size."""
    size = (3, 3, 3)

    with pytest.raises(ValueError):
        mat = torch.eye(3, 3)
        kt.transform.center_matrix(size, mat)

    with pytest.raises(ValueError):
        mat = torch.eye(3, 4)
        kt.transform.center_matrix(size, mat)


def test_center_matrix_values():
    """Test centering matrices in 2D and 3D, with different batch sizes."""
    width = 256

    for dim in (2, 3):
        for batch in ([], [4], [5, 4]):
            size = torch.tensor(width).expand(dim)
            inp = torch.rand(*batch, dim + 1, dim + 1)

            cen = torch.eye(dim + 1)
            unc = torch.eye(dim + 1)
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
    assert inp.eq(orig).all()


def test_jacobian_illegal_shape():
    """Test computing Jacobian with illegal field shape."""
    size = (3, 3, 3)
    ndim = len(size)
    field = torch.ones(1, ndim + 1, *size)

    with pytest.raises(ValueError):
        kt.transform.jacobian(field)


def test_jacobian_values():
    """Test computing Jacobian determinants."""
    width = 8
    batch = 7

    for dim in (2, 3):
        size = torch.tensor(width).expand(dim)
        x = torch.empty(batch, 1, *size)

        # Determinants of batch of random affine transforms. Make shape
        # broadcastable to field shape (batch, *size).
        mat = kt.random.affine(x)
        det = mat.det().view(batch, *[1] * dim)

        # Recover same determinants from displacement or deformation fields.
        grid = kt.transform.grid(size)
        for is_disp in (True, False):
            inp = kt.transform.compose(mat, grid=grid, absolute=not is_disp)
            out = kt.transform.jacobian(inp, is_disp=is_disp)
            assert out.allclose(det)


def test_fit_matrix_trivial():
    """Test output shape and dtype when fitting matrix transforms."""
    # Expect a square matrix of size (N + 1) in N dimensions.
    dim = 3
    x = torch.rand(5, dim)
    out = kt.transform.fit_matrix(x, x, ridge=1e-4)
    assert out.shape == (dim + 1, dim + 1)

    # Expect output to have input batch dimensions.
    batch = (3, 4)
    dim = 2
    x = torch.rand(*batch, 5, dim)
    out = kt.transform.fit_matrix(x, x)
    assert out.shape == (*batch, dim + 1, dim + 1)

    # Expect default data type.
    assert out.dtype == torch.get_default_dtype()


def test_fit_matrix_scale():
    """Test if matrix fit recovers scaling without noise, in 3D."""
    dim = 3
    scale = 2
    x = torch.rand(20, dim)
    y = scale * x

    expected = (*[scale] * dim, 1)
    expected = torch.tensor(expected, dtype=torch.get_default_dtype()).diag()

    out = kt.transform.fit_matrix(x, y, ridge=0)
    assert out.allclose(expected)


def test_fit_matrix_shift():
    """Test if matrix fit recovers translation without noise, in 2D."""
    dim = 2
    shift = 7
    x = torch.rand(10, dim)
    y = x + shift

    expected = torch.eye(dim + 1)
    expected[:-1, -1] = shift

    out = kt.transform.fit_matrix(x, y, ridge=0)
    assert out.allclose(expected, atol=1e-6)


def test_fit_matrix_weighted():
    """Test matrix fit with down-weighted outlier."""
    dim = 2
    shift = -5
    points = 10

    # Noise-free data.
    x = torch.rand(points, dim)
    y = x + shift

    # Outlier with reduced weight.
    y[-1, :] = 100
    weights = torch.ones(points)
    weights[-1] = 0

    expected = torch.eye(dim + 1)
    expected[:-1, -1] = shift

    out = kt.transform.fit_matrix(x, y, weights=weights)
    assert out.allclose(expected, atol=1e-4)
