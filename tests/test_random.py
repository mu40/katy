"""Tests for generation module."""


import torch
import pytest
import katy as kt


def test_crop_unchanged():
    """Test if crop-mask generation leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4)
    orig = inp.clone()
    kt.random.crop(inp)
    assert inp.eq(orig).all()


def test_crop_properties():
    """Test type, shape, and channel count of cropping mask, in 3D."""
    inp = torch.ones(2, 3, 4, 4, 4)

    out = kt.random.crop(inp, crop=torch.tensor(1))
    assert out.dtype == torch.bool
    assert inp[:, 0, ...].shape == out[:, 0, ...].shape


def test_crop_single_axis():
    """Test if cropping mask crops only along a single axis."""
    size = torch.tensor((8, 8))
    inp = torch.ones(1, 1, *size)

    out = kt.random.crop(inp, crop=0.5)
    low, upp = out.squeeze().nonzero().aminmax(dim=0)
    width = upp - low + 1
    assert width.lt(size).type(torch.int32).sum() <= 1


def test_crop_illegal_values():
    """Test crop-mask generation with illegal input arguments, in 2D."""
    x = torch.empty(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.random.crop(x, crop=-0.1)

    with pytest.raises(ValueError):
        kt.random.crop(x, crop=1.1)


def test_affine_unchanged():
    """Test if matrix-transform generation leaves input unchanged."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4, 4)
    orig = inp.clone()
    kt.random.affine(inp)
    assert inp.eq(orig).all()


def test_affine_batches():
    """Test if generating affine-transforms, with per-axis tensor input."""
    # Input of shape: batch, channel, space.

    for dim in (2, 3):
        x = torch.empty(7, 1, *[4] * dim)
        shift = torch.tensor((0, 10) * dim)
        out = kt.random.affine(x, shift=shift)
        assert out.shape == (x.size(0), dim + 1, dim + 1)


def test_affine_ranges():
    """Test defining affine sampling ranges in various ways."""
    value = 0.5

    for dim in (2, 3):
        x = torch.empty(1, 1, *[4] * dim)
        num = 1 if dim == 2 else 3
        for k, n in dict(shift=dim, angle=num, scale=dim, shear=num).items():
            kt.random.affine(x, **{k: value})
            kt.random.affine(x, **{k: [value]})
            kt.random.affine(x, **{k: [value] * 2})
            kt.random.affine(x, **{k: [value] * 2 * n})
            kt.random.affine(x, **{k: torch.tensor(value)})
            kt.random.affine(x, **{k: torch.tensor([value])})
            kt.random.affine(x, **{k: torch.tensor([value] * 2)})
            kt.random.affine(x, **{k: torch.tensor([value] * 2 * n)})


def test_affine_values():
    """Test generating translation, rotation, scaling, and shear matrices."""
    # Expect deterministic transforms for "fixed" sampling range.
    par = 7

    # Test 2D and 3D.
    for dim in (2, 3):
        space = [4] * dim
        x = torch.empty(1, 1, *space)

        # Translation.
        out = torch.eye(dim + 1)
        out[:dim, -1] = par
        out = kt.transform.center_matrix(space, out).unsqueeze(0)
        ranges = dict(shift=(par, par), angle=0, scale=0, shear=0)
        assert kt.random.affine(x, **ranges).allclose(out)

        # Rotation.
        angle = [par] * (3 if dim == 3 else 1)
        out = torch.eye(dim + 1)
        out[:dim, :dim] = kt.transform.compose_rotation(angle)
        out = kt.transform.center_matrix(space, out).unsqueeze(0)
        ranges = dict(shift=0, angle=(par, par), scale=0, shear=0)
        assert kt.random.affine(x, **ranges).allclose(out, rtol=1e-4)

        # Scaling. Function takes offset from 1.
        out = torch.eye(dim + 1)
        out.diagonal()[:dim] = par + 1
        out = kt.transform.center_matrix(space, out).unsqueeze(0)
        ranges = dict(shift=0, angle=0, scale=(par, par), shear=0)
        assert kt.random.affine(x, **ranges).allclose(out)

        # Shear.
        out = torch.eye(dim + 1)
        out[*torch.triu_indices(dim, dim, offset=1)] = par
        out = kt.transform.center_matrix(space, out).unsqueeze(0)
        ranges = dict(shift=0, angle=0, scale=0, shear=(par, par))
        assert kt.random.affine(x, **ranges).allclose(out)


def test_warp_unchanged():
    """Test if warp generation leaves input unchanged, in 2D."""
    # Input of shape: batch, channel, space. Fewer control points than voxels.
    inp = torch.ones(1, 1, 3, 3)
    orig = inp.clone()
    kt.random.warp(inp, points=2)
    assert inp.eq(orig).all()


def test_warp_shape():
    """Test the generated warp shape, with various inputs."""
    batch = 6
    channels = 5
    space = (4, 4, 4)

    for dim in (2, 3):
        inp = torch.empty(batch, channels, *space[:dim])
        out = kt.random.warp(inp, damp=torch.tensor(0.1), points=(2, 3))
        assert out.shape == (batch, dim, *space[:dim])


def test_warp_maximum():
    """Test the maximum displacement in 3D, with tensor range."""
    inp = torch.ones(3, 1, 8, 8, 8)
    disp = torch.tensor(20.)
    space = tuple(range(2, inp.ndim))

    out = kt.random.warp(inp, disp=torch.stack((disp, disp)), points=2)
    assert out.abs().amax(dim=space).allclose(disp)


def test_warp_illegal_values():
    """Test warp generation with illegal input arguments, in 1D."""
    x = torch.zeros(1, 1, 4)

    # Damping factor should be non-negative.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, damp=-0.1)

    # Control points should be less than tensor width.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=4)

    # Displacement strength should be of length in `(1, 2, 2 * dim)`.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, disp=torch.ones(3))

    # Integration steps should not be negative.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, steps=-1)
