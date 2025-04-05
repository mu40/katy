"""Tests for losses module."""

import katy as kt
import pytest
import torch


def test_dice_shape():
    """Test if Dice loss returns scalar."""
    size = (2, 3, 4, 4, 4)

    for dim in (1, 2, 3):
        inp = torch.ones(size[:2 + dim])
        out = kt.losses.dice(inp, inp)
        assert out.ndim == 0


def test_dice_broadcastable_shapes():
    """Test Dice loss on tensors with broadcastable shapes."""
    x = torch.ones(2, 3, 4, 4)
    y = torch.ones(2, 1, 4, 4)

    # Expect failure as different shapes are likely a bug.
    with pytest.raises(ValueError):
        kt.losses.dice(x, y)

def test_dice_values():
    """Test Dice loss values in 2D."""
    half = 4
    size = (1, 1, half * 2, half * 2)

    # Same-size inputs that half overlap.
    x = torch.zeros(size)
    y = torch.zeros(size)
    x[..., :half, :] = 1
    y[..., :, :half] = 1

    # Identical inputs should yield Dice 1, loss value 0.
    assert kt.losses.dice(x, x).eq(0)
    assert kt.losses.dice(y, y).eq(0)

    # Half-overlapping same-size should yield Dice and loss value 0.5.
    assert kt.losses.dice(x, y).eq(0.5)
    assert kt.losses.dice(y, x).eq(0.5)


def test_dice_channels():
    """Test Dice loss average over channels, in 3D."""
    half = 4
    size = (1, 1, half * 2, half * 2)

    # Same-size inputs that do not overlap.
    x = torch.zeros(size)
    y = torch.zeros(size)
    x[..., :half] = 1
    y[..., half:] = 1

    # Average over batch entries should be 0.5.
    xx = torch.cat((x, x), dim=0)
    xy = torch.cat((x, y), dim=0)
    assert kt.losses.dice(xx, xy).eq(0.5)

    # Average over channels should be 0.5.
    xx = torch.cat((x, x), dim=1)
    xy = torch.cat((x, y), dim=1)
    assert kt.losses.dice(xx, xy).eq(0.5)


def test_ncc_trivial():
    """Test computing NCC."""
    inp = torch.zeros(4, 3, 2, 2, 2, dtype=torch.int64)
    out = kt.losses.ncc(inp, inp, width=5)

    assert out.dtype == torch.get_default_dtype()
    assert out.ndim == 0
    assert out == -1


def test_ncc_noise():
    """Test NCC on uncorrelated noise."""
    x = torch.rand(2, 3, 20)
    y = torch.rand_like(x)

    # Expect square of mid-range score for uncorrelated noise.
    out = kt.losses.ncc(x, y)
    assert -0.5 < out < -0.1
