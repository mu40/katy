"""Tests for utility module."""


import torch
import pytest
import kathryn as kt


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
