"""Unit tests for resampling module."""

import katy as kt
import pytest
import torch


@pytest.mark.parametrize('dim', [2, 3])
def test_interpolate_identity(dim):
    """Test interpolating the grid at the grid locations."""
    # Data of shape: batch, channels, space.
    grid = kt.transform.grid(size=[2] * dim)
    inp = grid.unsqueeze(0)

    # Coordinates without batch dimension.
    out = kt.transform.interpolate(inp, grid, mode='nearest', padding='border')
    assert out.allclose(inp)

    # Coordinates with batch dimension.
    grid = grid.unsqueeze(0)
    out = kt.transform.interpolate(inp, grid, mode='linear', padding='zeros')
    assert out.allclose(inp)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('mode', ['nearest', 'linear'])
@pytest.mark.parametrize('padding', ['zeros', 'border', 'reflection'])
def test_apply_identity(dim, mode, padding):
    """Test applying an identity matrix and field with batch dimension."""
    size = [5] * dim
    inp = torch.ones(size).view(1, 1, *size)

    # Test matrix and displacement field.
    matrix = torch.eye(dim + 1).unsqueeze(0)
    field = torch.zeros_like(inp).expand(1, dim, *size)
    for trans in (matrix, field):
        out = kt.transform.apply(inp, trans, mode=mode, padding=padding)
        assert out.allclose(inp)


@pytest.mark.parametrize('dim', [2, 3])
def test_apply_shift(dim):
    """Test if shifting is equivalent to rolling, except at the border."""
    shift = 2
    width = 5
    inp = torch.arange(width ** dim).view(1, 1, *[width] * dim)

    # Transform shifting along the trailing axis.
    trans = torch.eye(dim + 1)
    trans[-2, -1] = shift
    out = kt.transform.apply(inp, trans, mode='nearest')

    # Roll volume by the same amount. Remove border from comparison.
    inp = inp.roll(-shift, dims=-1).to(out.dtype)
    assert out[..., :-shift].allclose(inp[..., :-shift])


def test_integrate_properties():
    """Test vector integration properties."""
    inp = torch.arange(50.).reshape(1, 2, 5, 5)
    orig = inp.clone()

    # Input should not change, but output differ from input.
    out = kt.transform.integrate(inp, steps=3)
    assert inp.equal(orig)
    assert not out.equal(inp)


def test_integrate_zero_steps():
    """Test if integrating with zero steps returns the same field."""
    x = torch.zeros(1, 3, 4, 4, 4)
    assert kt.transform.integrate(x, steps=0) is x


def test_integrate_illegal_arguments():
    """Test integration with illegal input arguments."""
    # The number of steps should not be negative.
    with pytest.raises(ValueError):
        inp = torch.zeros(1, 2, 4, 4)
        kt.transform.integrate(inp, steps=-1)

    # Field shape should be: batch, dimension, space.
    with pytest.raises(ValueError):
        inp = torch.zeros(1, 4, 4)
        kt.transform.integrate(inp, steps=1)


def test_integrate_inverse():
    """Test if integrating a negated SVF yields the inverse warp."""
    # SVF of maximum amplitude 1.
    svf = torch.arange(800.).view(1, 2, 20, 20)
    svf /= svf.norm(dim=1).max()

    # Integrate, compose, compute norm. Remove border values.
    fw = kt.transform.integrate(+svf, steps=5)
    bw = kt.transform.integrate(-svf, steps=5)
    out = kt.transform.compose(fw, bw)
    out = torch.linalg.vector_norm(out, dim=1)
    out = out[..., 1:-1, 1:-1]

    # Expect identity with error below 0.1% of maximum shift away from border.
    assert out.max() < 1e-3
