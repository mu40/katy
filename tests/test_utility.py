"""Tests for utility module."""


import torch
import pytest
import katy as kt


def test_chance_properties():
    """Test uniform boolean sampling properties."""
    size = (2, 3)

    out = kt.utility.chance(prob=0.3)
    assert out.dtype == torch.bool
    assert out.numel() == 1

    out = kt.utility.chance(prob=0, size=size)
    assert not out.all()
    assert out.shape == size

    out = kt.utility.chance(prob=1, size=size)
    assert out.all()
    assert out.shape == size


def test_chance_tensor_size():
    """Test passing chance sizes of various types."""
    prob = torch.tensor(0.5)

    size = torch.Size((2, 3))
    out = kt.utility.chance(prob, size)
    assert all(o == s for o, s in zip(out.shape, size))

    size = torch.tensor((2, 3))
    out = kt.utility.chance(prob, size)
    assert all(o == s for o, s in zip(out.shape, size))


def test_chance_illegal_values():
    """Test passing probabilities outside the [0, 1] range."""
    with pytest.raises(ValueError):
        kt.utility.chance(prob=-0.1)

    with pytest.raises(ValueError):
        kt.utility.chance(prob=1.1)


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
