"""Tests for transform module."""


import torch
import kathryn as kt


def test_grid():
    """Test index-coordinate grid creation with ij-indexing."""
    size = (4, 9)
    grid = kt.transform.grid(size)
    assert grid.size() == (len(size), *size)

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
