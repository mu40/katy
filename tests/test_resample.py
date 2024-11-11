"""Tests for resampling module."""


import torch
import kathryn as kt


def test_interpolate_identity():
    """Test interpolation at the grid locations."""
    size = (8, 8, 8)

    for dim in (2, 3):
        # Data of shape: batch, channels, space.
        inp = torch.rand(5, 4, *size[:dim])
        grid = kt.transform.grid(size[:dim])

        # Coordinates without batch dimension.
        out = kt.transform.interpolate(inp, grid)
        assert out.allclose(inp, atol=1e-5)

        # Coordinates with batch dimension.
        grid = grid.unsqueeze(0)
        out = kt.transform.interpolate(inp, grid)
        assert out.allclose(inp, atol=1e-5)



def test_transform_identity():
    """Test applying an identity matrix and field with various options."""
    size = (10, 10, 10)

    for dim in (2, 3):
        inp = torch.rand(size[:dim]).unsqueeze(0).unsqueeze(0)
        matrix = torch.eye(dim + 1)
        shifts = torch.zeros_like(inp).expand(-1, dim, *size[:dim])

        for trans in (matrix, shifts):
            for method in ('nearest', 'linear'):
                for padding in ('zeros', 'border', 'reflection'):
                    prop = dict(method=method, padding=padding)
                    out = kt.transform.transform(inp, trans, **prop)
                    assert out.allclose(inp, atol=1e-5)


def test_transform_shift():
    """Test if shifting equivalent to rolling, except at the border."""
    size = (10, 10, 10)
    shift = 2

    for dim in (2, 3):
        inp = torch.rand(size[:dim]).unsqueeze(0).unsqueeze(0)

        # Transform shifting along the training axis.
        trans = torch.eye(dim + 1)
        trans[-2, -1] = shift
        out = kt.transform.transform(inp, trans, method='nearest')

        # Roll volume by the same amount. Remove border from comparison.
        ref = inp.roll(-shift, dims=-1)
        assert out[..., :-shift].allclose(ref[..., :-shift])
