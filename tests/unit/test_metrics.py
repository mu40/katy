"""Unit tests for metrics module."""

import katy as kt
import pytest
import torch


@pytest.mark.parametrize('labels', [7, (10, 11)])
def test_dice_shape(labels):
    """Test if number of labels define Dice metric output shape."""
    inp = torch.ones(1, 1, 4, 4, dtype=torch.int64)
    out = kt.metrics.dice(inp, inp, labels=labels)
    channels = torch.tensor(labels).ravel().numel()
    assert out.shape == (inp.size(0), channels)


def test_dice_memory():
    """Test Dice metric for labels held in memory."""
    x = torch.tensor((0, 3, 3, 2)).unsqueeze(0).unsqueeze(0)
    y = torch.tensor((0, 1, 3, 3)).unsqueeze(0).unsqueeze(0)
    z = (0, 1, 2, 3, 4)

    for labels in (tuple(z), list(z), torch.tensor(z), {i: 'hi' for i in z}):
        dice = kt.metrics.dice(x, y, labels).squeeze()
        assert dice[0] == 1
        assert dice[1] == 0
        assert dice[2] == 0
        assert dice[3] == 0.5
        assert dice[4] == 0


def test_dice_disk(tmp_path):
    """Test Dice metric for labels in file."""
    x = torch.tensor((0, 1, 2, 0, 176)).unsqueeze(0).unsqueeze(0)
    z = (1, 2)

    f = tmp_path / 'labels.json'
    for labels in (tuple(z), list(z), {i: 'hi' for i in z}):
        kt.io.save(labels, f)
        dice = kt.metrics.dice(x, x, labels=f).squeeze()
        assert dice[0] == 1
        assert dice[1] == 1


def test_dice_illegal_inputs():
    """Test computing Dice raises an error for different input shapes."""
    x = torch.zeros(1, 3, 3, 3)
    y = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError):
        kt.metrics.dice(x, y, labels=1)


def test_ncc_trivial():
    """Test computing NCC."""
    inp = torch.zeros(4, 3, 2, 2, dtype=torch.int64)
    out = kt.metrics.ncc(inp, inp, width=3)

    assert out.dtype == torch.get_default_dtype()
    assert out.shape == inp.shape[:2]
    assert out.eq(1).all()


def test_ncc_channels():
    """Test computing NCC channel-wise."""
    width = 8
    x = torch.zeros(1, 2, width, width)
    x[..., width // 2] = 1
    y = x * 2
    y[:, 1] = y[:, 1].mT.clone()

    # Expect 1 for scaled image. Lower score for transposed image.
    out = kt.metrics.ncc(x, y).squeeze()
    assert out[0] == 1
    assert 0.1 < out[1] < 0.3
