"""Unit tests for losses module."""

import katy as kt
import pytest
import torch


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_dice_shape(dim):
    """Test if Dice loss returns scalar."""
    x = torch.ones(2, 3, *[4] * dim)
    assert kt.losses.dice(x, x).ndim == 0


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


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_axial_shape(dim):
    """Test if axial diffusion loss returns scalar."""
    x = torch.ones(2, dim, *[4] * dim)
    assert kt.losses.axial(x, norm=2).ndim == 0


def test_axial_shift():
    """Test if regularization loss on translation is zero."""
    x = torch.ones(1, 2, 5, 5)
    assert kt.losses.axial(x).eq(0)


def test_axial_positive():
    """Test if regularization positive."""
    x = torch.ones(1, 2, 4, 4)
    x[:, 0, 2, :] = 0
    assert kt.losses.axial(x).gt(0)


def test_axial_illegal_values():
    """Test passing illegal shapes and norm orders."""
    # Expect failure if not a vector field.
    x = torch.ones(1, 1, 3, 3)
    with pytest.raises(ValueError):
        kt.losses.axial(x)

    # Expect failure for norm order outside `(1, 2)`.
    x = torch.ones(1, 2, 3, 3)
    with pytest.raises(ValueError):
        kt.losses.axial(x, norm=0)
    with pytest.raises(ValueError):
        kt.losses.axial(x, norm=3)
