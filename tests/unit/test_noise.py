"""Unit tests for index module."""

import katy as kt
import pytest
import torch


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_perlin_dim(dim):
    """Test Perlin noise shape in various dimensions."""
    size = (6, 7, 8)
    out = kt.noise.perlin(size=size[:dim], points=4)
    assert out.shape == size[:dim]


def test_perlin_scalar_size():
    """Test Perlin noise generation passing a scalar size."""
    size = 32
    assert kt.noise.perlin(size).shape == (size,)


@pytest.mark.parametrize('batch', [3, (3, 4), torch.tensor((4, 3))])
def test_perlin_batches(batch):
    """Test generating batches of Perlin noise, with the size as a tensor."""
    size = torch.tensor((5, 5))
    out = kt.noise.perlin(size, batch=batch)
    if torch.as_tensor(batch).ndim == 0:
        batch = [batch]

    assert out.shape == (*batch, *size)


@pytest.mark.parametrize('points', [2, (3,), (2, 3), torch.tensor((2, 3))])
def test_perlin_points(points):
    """Test generating 2D Perlin noise with various control point types."""
    size = (5, 5)
    assert kt.noise.perlin(size, points=points).shape == size


def test_perlin_illegal_points():
    """Test generating Perlin noise with illegal control-point numbers."""
    n = 10
    size = (n, n)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, points=1)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, points=n)


@pytest.mark.parametrize('pers', [0, 1.1])
def test_octaves_illegal_persistence(pers):
    """Test generating Perlin octaves with illegal persistence values."""
    size = (4, 4)
    points = (2, 3)
    with pytest.raises(ValueError):
        kt.noise.octaves(size, points, pers=pers)


def test_octaves_illegal_frequency():
    """Test generating Perlin octaves with too many points for size."""
    size = (4, 4)
    points = 4
    with pytest.raises(ValueError):
        kt.noise.octaves(size, points, pers=1)


def test_octaves_values():
    """Test normalization of Perlin octaves, with tensor inputs."""
    size = torch.Size((5, 5))
    points = torch.tensor((2, 3))

    out = kt.noise.octaves(size, points, pers=0.5)
    assert out.min() == 0
    assert out.max() == 1


def test_octaves_batches():
    """Test generating batches of octaves with different persistence."""
    size = (4, 4)
    points = (2, 3)
    batch = (3, 4)
    pers = 0.3 * torch.ones(*batch)

    out = kt.noise.octaves(size, points, pers, batch=batch)
    assert out.shape == (*batch, *size)
