"""Tests for losses module."""


import torch
import pytest
import katy as kt


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
