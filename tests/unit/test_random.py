"""Tests for generation module."""

import katy as kt
import pytest
import torch


def test_chance_properties():
    """Test uniform boolean sampling properties."""
    size = (2, 3)

    out = kt.random.chance()
    assert out.dtype == torch.bool
    assert out.numel() == 1

    out = kt.random.chance(prob=0, size=size)
    assert not out.all()
    assert out.shape == size

    out = kt.random.chance(prob=1, size=size)
    assert out.all()
    assert out.shape == size


def test_chance_tensor_size():
    """Test passing chance sizes of various types."""
    prob = torch.tensor(0.5)

    out = kt.random.chance(prob, size=[])
    assert out.ndim == 0

    size = torch.Size((2, 3))
    out = kt.random.chance(prob, size)
    assert all(o == s for o, s in zip(out.shape, size, strict=True))

    size = torch.tensor((2, 3))
    out = kt.random.chance(prob, size)
    assert all(o == s for o, s in zip(out.shape, size, strict=True))


@pytest.mark.parametrize('prob', [-0.1, 1.1])
def test_chance_illegal_value(prob):
    """Test passing probabilities outside the [0, 1] range."""
    with pytest.raises(ValueError):
        kt.random.chance(prob=prob)


def test_affine_unchanged():
    """Test if matrix-transform generation leaves input unchanged."""
    # Input of shape: batch, channel, space.
    x = torch.ones(1, 1, 4, 4)
    orig = x.clone()
    kt.random.affine(x)
    assert x.equal(orig)


@pytest.mark.parametrize('ndim', [2, 3])
def test_affine_batches(ndim):
    """Test if generating affine-transforms, with per-axis tensor input."""
    # Input of shape: batch, channel, space.
    x = torch.ones(7, 1, *[4] * ndim)
    shift = torch.tensor((0, 10) * ndim)
    out = kt.random.affine(x, shift=shift)
    assert out.shape == (x.size(0), ndim + 1, ndim + 1)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('par', ['shift', 'angle', 'scale', 'shear'])
def test_affine_ranges(ndim, par):
    """Test defining affine sampling ranges in various ways."""
    x = torch.ones(1, 1, *[4] * ndim)

    n = 1 if ndim == 2 else 3
    if par in ('shift', 'scale'):
        n = ndim

    value = 0.5
    ranges = (value, [value], [value] * 2, [value] * 2 * n)
    ranges = (*ranges, *(torch.tensor(r) for r in ranges))
    for r in ranges:
        assert kt.random.affine(x, **{par: r}).shape == (1, ndim + 1, ndim + 1)


@pytest.mark.parametrize('ndim', [2, 3])
def test_affine_values(ndim):
    """Test generating translation, rotation, scaling, and shear matrices."""
    # Expect deterministic transforms for "fixed" sampling range.
    par = 7
    space = [4] * ndim
    x = torch.zeros(1, 1, *space)

    # Translation.
    out = torch.eye(ndim + 1)
    out[:ndim, -1] = par
    out = kt.transform.center_matrix(space, out).unsqueeze(0)
    ranges = dict(shift=(par, par), angle=0, scale=0, shear=0)
    assert kt.random.affine(x, **ranges).allclose(out)

    # Rotation.
    angle = [par] * (3 if ndim == 3 else 1)
    out = torch.eye(ndim + 1)
    out[:ndim, :ndim] = kt.transform.compose_rotation(angle)
    out = kt.transform.center_matrix(space, out).unsqueeze(0)
    ranges = dict(shift=0, angle=(par, par), scale=0, shear=0)
    assert kt.random.affine(x, **ranges).allclose(out, rtol=1e-4)

    # Scaling. Function takes offset from 1.
    out = torch.eye(ndim + 1)
    out.diagonal()[:ndim] = par + 1
    out = kt.transform.center_matrix(space, out).unsqueeze(0)
    ranges = dict(shift=0, angle=0, scale=(par, par), shear=0)
    assert kt.random.affine(x, **ranges).allclose(out)

    # Shear.
    out = torch.eye(ndim + 1)
    out[*torch.triu_indices(ndim, ndim, offset=1)] = par
    out = kt.transform.center_matrix(space, out).unsqueeze(0)
    ranges = dict(shift=0, angle=0, scale=0, shear=(par, par))
    assert kt.random.affine(x, **ranges).allclose(out)


def test_warp_unchanged():
    """Test if warp generation leaves input unchanged, in 2D."""
    # Input of shape: batch, channel, space. Fewer control points than voxels.
    x = torch.ones(1, 1, 3, 3)
    orig = x.clone()
    kt.random.warp(x, points=2)
    assert x.equal(orig)


@pytest.mark.parametrize('ndim', [2, 3])
def test_warp_shape(ndim):
    """Test the generated warp shape, with various inputs."""
    batch = 2
    channels = 3
    space = [4] * ndim

    x = torch.ones(batch, channels, *space)
    out = kt.random.warp(x, damp=torch.tensor(0.1), points=(2, 3))
    assert out.shape == (batch, ndim, *space)


def test_warp_maximum():
    """Test the maximum displacement in 3D, with tensor range."""
    x = torch.ones(3, 1, 8, 8, 8)
    disp = torch.tensor(20.)
    space = tuple(range(2, x.ndim))

    out = kt.random.warp(x, disp=torch.stack((disp, disp)), points=2)
    assert out.abs().amax(dim=space).allclose(disp)


def test_warp_illegal_values():
    """Test warp generation with illegal input arguments, in 1D."""
    x = torch.zeros(1, 1, 4)

    # Damping factor should be non-negative.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, damp=-0.1)

    # Control points should be less than tensor width.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=4)

    # Displacement strength should be of length in `(1, 2, 2 * ndim)`.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, disp=torch.ones(3))

    # Integration steps should not be negative.
    with pytest.raises(ValueError):
        kt.random.warp(x, points=2, steps=-1)


def test_replay_without_generator():
    """Test replaying randomized operations without passing generator."""
    def add_noise(x, generator=None):
        return x + torch.rand(x.shape, generator=generator)

    # Expect same noise.
    x = torch.zeros(4, 4)
    replay = kt.random.replay(add_noise, device=None)
    a = replay(x)
    b = replay(x)
    assert b.equal(a)


def test_replay_with_generator():
    """Test replaying randomized operations, passing a generator."""
    def noise(generator):
        return torch.rand(10, generator=generator)

    # Expect same noise.
    gen = torch.Generator()
    replay = kt.random.replay(noise, generator=gen)
    a = replay()
    b = replay()
    assert a.equal(b)


def test_replay_illegal_values():
    """Test replaying randomizations with illegal arguments."""
    def without_generator():
        return

    def with_generator(generator=None):
        return

    # Expect wrapped function to require keyword argument `generator`.
    f = kt.random.replay(without_generator)
    with pytest.raises(TypeError):
        f()

    # Expect not to be able to pass `generator` to wrapped function.
    f = kt.random.replay(with_generator)
    f()
    with pytest.raises(TypeError):
        f(generator=None)
