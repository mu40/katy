"""Tests for index module."""


import torch
import pytest
import kathryn as kt


def test_perlin_dimensions():
    """Test Perlin noise shape with various dimensions."""
    size = (20, 21, 22, 23)

    for dim in (1, 2, 3):
        for cells in (1, 4, 8):
            out = kt.noise.perlin(size=size[:dim], cells=cells)
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


def test_perlin_cells_2d():
    """Test generating 2D Perlin noise with cell numbers of various types."""
    size = (8, 8)

    cells = (4,)
    out = kt.noise.perlin(size, cells)
    assert out.shape == size

    cells = (4, 3)
    out = kt.noise.perlin(size, cells)
    assert out.shape == size

    cells = torch.tensor(cells)
    out = kt.noise.perlin(size, cells)
    assert out.shape == size


def test_perlin_illegal_cells():
    """Test generating Perlin noise with illegal gradient cell numbers."""
    n = 10
    size = (n, n)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, cells=0)

    with pytest.raises(ValueError):
        kt.noise.perlin(size, cells=n)


def test_octaves_illegal_persistence():
    """Test generating Perlin octaves with illegal persistence values."""
    size = (4, 4)
    freq = (0, 1)

    with pytest.raises(ValueError):
        kt.noise.octaves(size, freq, pers=0)

    with pytest.raises(ValueError):
        kt.noise.octaves(size, freq, pers=1.1)


def test_octaves_illegal_frequency():
    """Test generating Perlin octaves with frequency too high for size."""
    size = (4, 4)
    freq = 2

    with pytest.raises(ValueError):
        kt.noise.octaves(size, freq, pers=1)


def test_octaves_normalization():
    """Test normalization of Perlin octaves, with tensor inputs."""
    size = torch.Size((4, 4))
    freq = torch.tensor((0, 1))

    out = kt.noise.octaves(size, freq, pers=0.5)
    assert out.min() == 0
    assert out.max() == 1


def test_octaves_batches():
    """Test generating batches of octaves with different persistence."""
    size = (5, 5)
    freq = (1, 2)
    batch = (3, 4)

    # Persistence must be in (0, 1].
    pers = torch.rand(*batch).mul(0.5).add(0.1)

    out = kt.noise.octaves(size, freq, pers, batch=batch)
    assert out.shape == (*batch, *size)
