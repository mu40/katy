"""Tests for generation module."""


import torch
import pytest
import kathryn as kt


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
    x = torch.rand(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.random.crop(x, crop=-0.1)

    with pytest.raises(ValueError):
        kt.random.crop(x, crop=1.1)
