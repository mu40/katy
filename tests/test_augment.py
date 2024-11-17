"""Tests for augmentation module."""


import torch
import pytest
import kathryn as kt


def test_gamma_unchanged():
    """Test if gamma-augmentation leaves input unchanged."""
    # Input of shape: batch, channel, space.
    inp = torch.rand(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.gamma(inp)
    assert inp.eq(orig).all()


def test_gamma_probability():
    """Test if gamma augmentation with zero probability is normalization."""
    inp = torch.rand(1, 1, 8, 8)
    out = kt.augment.gamma(inp, prob=0)

    inp -= inp.min()
    inp /= inp.max()
    assert out.eq(inp).all()


def test_gamma_normalization():
    """Test batch and channel-wise normalization of gamma augmentation."""
    inp = torch.rand(3, 2, 4, 4)

    # Expect separate normalization of every channel of every batch.
    out = kt.augment.gamma(inp)
    for batch in out:
        for channel in batch:
            assert channel.min() == 0
            assert channel.max() == 1

    # With `shared=True`, expect per-batch normalization.
    out = kt.augment.gamma(inp, shared=True)
    for batch in out:
        assert batch.min() == 0
        assert batch.max() == 1

        # Shared normalization of channels.
        with pytest.raises(AssertionError):
            for channel in batch:
                assert channel.min() == 0
                assert channel.max() == 1


def test_gamma_shared_channels():
    """Test shared gamma augmentation across channels."""
    # One batch of two identical input channels.
    inp = torch.rand(1, 1, 4, 4).expand(-1, 2, -1, -1).clone()

    # Identical channels after shared augmentation.
    out = kt.augment.gamma(inp, shared=True).squeeze()
    assert out[0].eq(out[1]).all()

    # Different channels after separate augmentation.
    out = kt.augment.gamma(inp, shared=False).squeeze()
    assert out[0].ne(out[1]).any()


def test_gamma_illegal_values():
    """Test passing gamma values outside (0, 1) range."""
    x = torch.rand(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=0)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=1)


def test_noise_unchanged():
    """Test if adding noise leaves input unchanged."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.noise(inp)
    assert inp.eq(orig).all()


def test_noise_change():
    """Test if adding noise changes the input."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 8, 8)

    # Adding noise should change the input.
    out = kt.augment.noise(inp, prob=1)
    assert out.ne(inp).any()

    # Nothing should change at zero probability.
    out = kt.augment.noise(inp, prob=torch.tensor(0))
    assert out.eq(inp).all()
