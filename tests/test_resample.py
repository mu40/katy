"""Tests for resampling module."""


import torch
import kathryn as kt


def test_interpolate_identity():
    """Test interpolation at the grid locations.."""
    size = (32, 16, 9)

    for dim in (2, 3):
        # Add singleton batch and channel dimensions.
        x = torch.rand(size[:dim]).unsqueeze(0).unsqueeze(0)
        grid = kt.transform.grid(size[:dim])

        y = kt.transform.interpolate(x, grid, method='nearest')
        assert y.allclose(x)

        y = kt.transform.interpolate(x, grid, method='linear')
        assert y.allclose(x, atol=1e-5)
