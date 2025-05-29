"""Tests for utility module."""

import katy as kt
import pytest
import torch


def test_resize_dtype():
    """Test data type persistence when resizing tensors."""
    for dtype in (torch.int64, torch.float32):
        x = torch.ones(1, 1, 3, 3, dtype=dtype)
        assert kt.utility.resize(x, size=2).dtype == dtype


def test_resize_size():
    """Test output shape when resizing tensors in 1D, 2D, and 3D."""
    batch = 1
    channel = 2
    size_old = (8, 4, 7)
    size_new = (5, 6, 8)

    for dim in (1, 2, 3):
        inp = torch.ones(batch, channel, *size_old[:dim])

        # Expect output of requested shape.
        out = kt.utility.resize(inp, size_new[:dim])
        assert out.shape == (batch, channel, *size_new[:dim])

        # Expect expansion of scalar sizes.
        out = kt.utility.resize(inp, size_new[0])
        assert out.shape == (batch, channel, *[size_new[0]] * dim)


def test_resize_identity():
    """Test if resizing to the same shape returns the same tensor."""
    size = (3, 3)
    x = torch.ones(1, 1, *size)

    assert kt.utility.resize(x, size) is x


def test_resize_fill():
    """Test resizing fill value."""
    x = torch.ones(1, 1, 5)
    size = x.shape[-1] + 1

    # Default fill value should be zero.
    out = kt.utility.resize(x, size=size)
    assert out[..., -1] == 0

    # Test setting specific fill value.
    fill = -3
    out = kt.utility.resize(x, size=size, fill=fill)
    assert out[..., -1] == fill


def test_resize_symmetry():
    """Test symmetry of resize operation."""
    x = [1, 2, 3, 4]
    inp = torch.tensor(x).unsqueeze(0).unsqueeze(0)

    # Expect even distribution of increase in size.
    out = kt.utility.resize(inp, size=6).squeeze()
    assert out.tolist() == [0, *x, 0]

    # If increase is odd, expect one more element at upper end.
    out = kt.utility.resize(inp, size=7).squeeze()
    assert out.tolist() == [0, *x, 0, 0]

    # Expect symmetric reduction in size.
    out = kt.utility.resize(inp, size=2).squeeze()
    assert out.tolist() == x[1:-1]

    # If decrease is odd, expect one fewer element at upper end.
    out = kt.utility.resize(inp, size=3).squeeze()
    assert out.tolist() == x[:-1]


def test_barycenter_trivial():
    """Test type and shape of barycenter, in 3D."""
    space = (4, 4, 4)
    x = torch.zeros(2, 3, *space)
    out = kt.utility.barycenter(x)

    # Expect shape `(batch, channel, dimensions)`.
    assert out.shape == (*x.shape[:2], len(space))

    # Expect zero barycenter for zero input.
    assert out.eq(0).all()

    # Expect default type.
    assert out.dtype == torch.get_default_dtype()


def test_barycenter_negative():
    """Test barycenter of negative values, in 1D."""
    x = torch.full(size=(1, 1, 8), fill_value=-1)

    # Expect clamping of negative values to zero.
    out = kt.utility.barycenter(x)
    assert out.eq(0).all()


def test_barycenter_values():
    """Test barycenter computation, in 2D."""
    # Single voxel.
    x = torch.zeros(1, 1, 8, 8)
    x[..., 2, 3] = 3
    out = kt.utility.barycenter(x).squeeze()
    assert out[0] == 2
    assert out[1] == 3

    # Last line.
    x = torch.zeros(1, 1, 8, 8)
    x[..., -1] = 1
    out = kt.utility.barycenter(x).squeeze()
    assert out[0] == 3.5
    assert out[1] == 7


def test_barycenter_batch():
    """Test barycenter computation without batch dimension."""
    # Single voxel.
    x = torch.zeros(1, 8, 8)
    x[..., 2, 3] = 3
    out = kt.utility.barycenter(x, batch=False)
    assert out[0, 0] == 2
    assert out[0, 1] == 3

    # Last line.
    x = torch.zeros(1, 8, 8)
    x[..., -1] = 1
    out = kt.utility.barycenter(x, batch=False)
    assert out[0, 0] == 3.5
    assert out[0, 1] == 7


def test_quantile_scalar():
    """Test if scalar quantiles are equivalent to `torch.quantile`."""
    x = torch.arange(10, dtype=torch.float32)
    x = torch.stack((x, x))

    q = 0.31
    for dim in (0, 1, None):
        for keepdim in (True, False):
            out = kt.utility.quantile(x, q, dim, keepdim)
            ref = x.quantile(q, dim, keepdim)
            assert out.shape == ref.shape
            assert out.equal(ref)


def test_quantile_shape():
    """Test the output shape when computing quantiles."""
    x = torch.ones(3, 4, 5)
    q = torch.ones(2)

    for dim in (0, 1, 2):
        size = list(x.shape)
        size[dim] = len(q)
        assert kt.utility.quantile(x, q, dim=dim).shape == tuple(size)


def test_quantile_illegal_inputs():
    """Test if quantiles outside range [0, 1] raise errors."""
    x = torch.ones(3)

    with pytest.raises(ValueError):
        kt.utility.quantile(x, q=-1)

    with pytest.raises(ValueError):
        kt.utility.quantile(x, 1.01)


def test_normalize_trivial():
    """Test min-max normalization along all dimensions."""
    x = torch.arange(9).reshape(3, 3)
    orig = x.clone()

    out = kt.utility.normalize(x)
    assert out.shape == x.shape
    assert out.dtype == torch.get_default_dtype()
    assert out.min() == 0
    assert out.max() == 1
    assert x.equal(orig)


def test_normalize_zero():
    """Test if min-max normalization avoids dividing by zero."""
    x = torch.ones(3)
    assert kt.utility.normalize(x, dim=None).eq(0).all()


def test_normalize_dim():
    """Test min-max normalization along specific dimensions."""
    x = torch.arange(10)
    x = torch.stack((x, x))

    # Flat values along dimension 0.
    out = kt.utility.normalize(x, dim=-2)
    assert out.shape == x.shape
    assert out.eq(0).all()

    # Changing values along dimension 1.
    out = kt.utility.normalize(x, dim=-1)
    assert out[:, 0].eq(0).all()
    assert out[:, -1].eq(1).all()
    assert out[:, 1:-1].gt(0).all()
    assert out[:, 1:-1].lt(1).all()


def test_normalize_quantile():
    """Test quantile normalization."""
    x = torch.arange(100)

    dim = None
    out = kt.utility.normalize(x, dim, min=0.1, max=0.9)
    assert out[:10].eq(0).all()
    assert out[-10:].eq(1).all()
    assert out[10:-10].gt(0).all()
    assert out[10:-10].lt(1).all()
