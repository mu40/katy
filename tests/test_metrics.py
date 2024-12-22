"""Tests for metrics module."""


import torch
import pytest
import katy as kt


def test_dice_shape():
    """Test Dice metric output shape."""
    # Channel count of one-hot inputs defines output channels.
    inp = torch.ones(2, 3, 4, 4)
    out = kt.metrics.dice(inp, inp)
    assert out.shape == inp.shape[:2]

    # Number of labels define output channels.
    inp = torch.ones(1, 1, 4, 4, dtype=torch.int64)
    out = kt.metrics.dice(inp, inp, labels=7)
    assert out.shape == (inp.size(0), 1)

    # Number of labels define output channels.
    inp = torch.ones(1, 1, 4, 4, dtype=torch.int64)
    out = kt.metrics.dice(inp, inp, labels=(10, 11))
    assert out.shape == (inp.size(0), 2)


def test_dice_indices():
    """Test Dice metric on index-valued label maps, in 1D."""
    x = torch.tensor((0, 3, 3, 2)).unsqueeze(0).unsqueeze(0)
    y = torch.tensor((0, 1, 3, 3)).unsqueeze(0).unsqueeze(0)

    dice = kt.metrics.dice(x, y, labels=(0, 1, 2, 3, 4)).squeeze()
    assert dice[0] == 1
    assert dice[1] == 0
    assert dice[2] == 0
    assert dice[3] == 0.5
    assert dice[4] == 0


def test_dice_illegal_inputs():
    """Test computing Dice metric with illegal arguments."""
    # Discrete-valued label maps require label definition.
    x = torch.zeros(1, 1, 3, 3)
    y = torch.zeros(1, 1, 3, 3)
    with pytest.raises(ValueError):
        kt.metrics.dice(x, y)

    # Input tensors should have the same size.
    x = torch.zeros(1, 3, 3, 3)
    y = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError):
        kt.metrics.dice(x, y)

    with pytest.raises(ValueError):
        kt.metrics.dice(x, y, labels=1)


def test_dice_probabilities():
    """Test Dice metric on probabilities, in 2D."""
    half = 4
    size = (1, 2, half * 2, half * 2)

    # Same-size inputs that half overlap.
    x = torch.zeros(size)
    y = torch.zeros(size)
    x[:, 0, half:, :] = 0.3
    x[:, 1, :half, :] = 0.3
    y[:, 0, :, half:] = 0.4
    y[:, 1, :, :half] = 0.4

    # Argmax of single channel should yield 1.
    assert kt.metrics.dice(x, y).eq(0.5).all()
    assert kt.metrics.dice(y, x).eq(0.5).all()
