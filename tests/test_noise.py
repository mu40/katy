"""Tests for index module."""


import torch
import pytest
import kathryn as kt


def test_perlin_dimensions():
    """Test Perlin noise shape with various dimensions."""
    size = (20, 21, 22, 23)

    for dim in (1, 2, 3):
        for points in (2, 4, 8):
            out = kt.noise.perlin(size=size[:dim], points=points)
            assert out.shape == size[:dim]


def test_perlin_batches():
    """Test generating batches of Perlin noise, with the size as a tensor."""
    size = torch.as_tensor((8, 8))

    batch = 4
    out = kt.noise.perlin(size, batch=batch)
    assert out.shape == (batch, *size)

    batch = (4, 5)
    out = kt.noise.perlin(size, batch=batch)
    assert out.shape == (*batch, *size)

    batch = torch.tensor(batch)
    out = kt.noise.perlin(size, batch=batch)
    assert out.shape == (*batch, *size)


def test_perlin_points_2d():
    """Test generating 2D Perlin noise with control points of various types."""
    size = (8, 8)

    out = kt.noise.perlin(size, points=(4,))
    assert out.shape == size

    out = kt.noise.perlin(size, points=(4, 3))
    assert out.shape == size

    out = kt.noise.perlin(size, points=torch.tensor((4, 3)))
    assert out.shape == size


def test_perlin_illegal_points():
    """Test generating Perlin noise with illegal control-point numbers."""
    n = 10
    size = (n, n)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, points=1)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, points=n)


def test_octaves_illegal_persistence():
    """Test generating Perlin octaves with illegal persistence values."""
    size = (4, 4)
    points = (2, 3)

    with pytest.raises(ValueError):
        kt.noise.octaves(size, points, persistence=0)

    with pytest.raises(ValueError):
        kt.noise.octaves(size, points, persistence=1.1)


def test_octaves_illegal_frequency():
    """Test generating Perlin octaves with too many points for size."""
    size = (4, 4)
    points = 4

    with pytest.raises(ValueError):
        kt.noise.octaves(size, points, persistence=1)


def test_octaves_normalization():
    """Test normalization of Perlin octaves, with tensor inputs."""
    size = torch.Size((5, 5))
    points = torch.tensor((2, 3))

    out = kt.noise.octaves(size, points, persistence=0.5)
    assert out.min() == 0
    assert out.max() == 1


def test_octaves_batches():
    """Test generating batches of octaves with different persistence."""
    size = (5, 5)
    points = (2, 3)
    batch = (3, 4)

    # Persistence must be in (0, 1].
    persistence = torch.rand(*batch).mul(0.5).add(0.1)

    out = kt.noise.octaves(size, points, persistence, batch=batch)
    assert out.shape == (*batch, *size)
