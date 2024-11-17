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
    size = (8, 8, 8)

    for dim in (2, 3):
        inp = torch.rand(size[:dim]).unsqueeze(0).unsqueeze(0)

        # Test matrix and displacement field.
        matrix = torch.eye(dim + 1)
        field = torch.zeros_like(inp).expand(1, dim, *size[:dim])
        for trans in (matrix, field):

            # Transform with and without batch dimension.
            for batch in (True, False):
                if not batch:
                    trans = trans.squeeze(0)

                for method in ('nearest', 'linear'):
                    for padding in ('zeros', 'border', 'reflection'):
                        prop = dict(method=method, padding=padding)
                        out = kt.transform.transform(inp, trans, **prop)
                        assert out.allclose(inp, atol=1e-5)


def test_transform_shift():
    """Test if shifting is equivalent to rolling, except at the border."""
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


def test_integrate():
    """Test if integrating a negated SVF yields the inverse warp."""
    fov = 128
    fwhm = fov / 4
    steps = 7
    dim = 2
    size = [fov] * dim

    # Smooth SVF of maximum amplitude 1.
    svf = torch.randn(4, dim, *size)
    svf = kt.filter.blur(svf, fwhm, dim=(-3, -2, -1))
    svf /= svf.norm(dim=1).max()

    # Integrate, compose, compute norm.
    fw = kt.transform.integrate(+svf, steps)
    bw = kt.transform.integrate(-svf, steps)
    out = kt.transform.compose(fw, bw)
    out = torch.linalg.vector_norm(out, dim=1)

    # Ignore border values.
    ind = torch.arange(1, fov - 1)
    for i in range(dim):
        out = out.index_select(dim=i + 1, index=ind)

    # Composition should yield identity. Expect less than 1% of maximum shift.
    assert out.max() < 0.01
