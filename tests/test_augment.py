"""Tests for augmentation module."""

import katy as kt
import pytest
import torch


def test_gamma_unchanged():
    """Test if gamma-augmentation leaves input unchanged, in 2D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.gamma(inp)
    assert inp.equal(orig)


def test_gamma_probability():
    """Test if gamma with zero probability is normalization, in 1D."""
    inp = torch.rand(1, 1, 8)
    out = kt.augment.gamma(inp, prob=0)

    inp -= inp.min()
    inp /= inp.max()
    assert out.equal(inp)


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
    """Test shared gamma augmentation across channels, in 3D."""
    # One batch of two identical input channels.
    inp = torch.rand(1, 1, 3, 3, 3).expand(-1, 2, -1, -1, -1).clone()

    # Identical channels after shared augmentation.
    out = kt.augment.gamma(inp, gamma=(0.5, 2), shared=True).squeeze()
    assert out[0].allclose(out[1])

    # Different channels after separate augmentation.
    out = kt.augment.gamma(inp, gamma=0.5, shared=False).squeeze()
    assert not out[0].equal(out[1])


def test_gamma_illegal_values():
    """Test passing gamma values leading to zero or negative exponents."""
    x = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=1)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=(0, 1))


def test_noise_unchanged():
    """Test if adding noise leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4)

    orig = inp.clone()
    kt.augment.noise(inp)
    assert inp.equal(orig)


def test_noise_change():
    """Test if adding noise changes the input, in 2D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4, 4)

    # Adding noise should change the input.
    out = kt.augment.noise(inp, prob=1)
    assert not out.equal(inp)

    # Nothing should change at zero probability.
    out = kt.augment.noise(inp, prob=torch.tensor(0))
    assert out.equal(inp)


def test_blur_unchanged():
    """Test if randomly blurring leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4)

    orig = inp.clone()
    kt.augment.blur(inp, fwhm=1)
    assert inp.equal(orig)


def test_blur_illegal_fwhm():
    """Test passing an illegal number of FWHM."""
    x = torch.zeros(1, 1, 4, 4)

    # FWHM should be of length 1, 2, or 2 * N, in N-dimensional space.
    with pytest.raises(ValueError):
        kt.augment.blur(x, fwhm=(1, 2, 3))


def test_blur_shared_channels():
    """Test if the function blurs all channels identically, in 2D."""
    inp = torch.rand(3, 1, 8, 8).expand(-1, 2, -1, -1)

    out = kt.augment.blur(inp, fwhm=5)
    for batch in out:
        assert batch[0].equal(batch[1])


def test_blur_deterministic():
    """Test random blurring at pinned down FWHM."""
    size = (4, 4, 4)
    fwhm = 5

    for dim in (1, 2, 3):
        # Blur input using specific FWHM.
        inp = torch.rand(2, 1, *size[:dim])
        ref = kt.filter.blur(inp, fwhm, dim=range(2, dim + 2))

        # Sampling FWHM between identical values should yield the same result.
        out = kt.augment.blur(inp, fwhm=(fwhm, fwhm))
        assert out.equal(ref)

        # The same applies when specifying identical bounds for each axis.
        out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm).repeat(2 * dim))
        assert out.equal(ref)

        # However, expect a different result when sampling between [0, fwhm).
        out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm))
        assert not out.equal(ref)


def test_bias_unchanged():
    """Test if bias leaves input unchanged, with tensor inputs in 3D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4, 4, 4)

    orig = inp.clone()
    kt.augment.bias(inp, floor=torch.tensor(0), points=torch.tensor((2, 3)))
    assert inp.equal(orig)


def test_bias_normalization():
    """Test if fixing minimum yields a bias field between 0 and 1, in 2D."""
    inp = torch.ones(2, 3, 4, 4)
    out = kt.augment.bias(inp, floor=(0, 0), points=2)

    # Expect separate normalization of each channel of each batch.
    for batch in out:
        for channel in batch:
            assert channel.min() == 0
            assert channel.max() == 1


def test_bias_shared_channels():
    """Test if with sharing, batches differ and channels do not, in 1D."""
    x = torch.rand(1, 4).expand(3, -1)
    y = kt.augment.bias(x, floor=0, points=(2, 3), shared=True, batch=False)

    # Channels should be identical.
    for channel in y[1:]:
        assert channel.allclose(y[0])


def test_bias_illegal_values():
    """Test bias modulation with illegal input arguments."""
    x = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.augment.bias(x, points=4)

    with pytest.raises(ValueError):
        kt.augment.bias(x, floor=-0.1)

    with pytest.raises(ValueError):
        kt.augment.bias(x, floor=+1.1)


def test_bias_return_field():
    """Test returning the bias field."""
    inp = torch.rand(1, 2, 5, 5)

    # Expect field matching input shape.
    _, field = kt.augment.bias(inp, return_bias=True)
    assert field.shape == inp.shape

    # Expect single-channel field with `shared`.
    out, field = kt.augment.bias(inp, return_bias=True, shared=True)
    assert field.shape == (inp.size(0), 1, *inp.shape[2:])

    # Expect multiplicative application.
    assert out.allclose(inp * field)


def test_downsample_unchanged():
    """Test if downsampling leaves input unchanged, in 3D."""
    space = (4, 4, 4)
    sizes = {True: (1, 1, *space), False: (1, *space)}

    for batch, size in sizes.items():
        x = torch.ones(size)
        orig = x.clone()

        kt.augment.downsample(x, factor=torch.tensor(2), batch=batch)
        assert x.equal(orig)


def test_downsample_illegal_values():
    """Test downsampling modulation with illegal input arguments, in 1D."""
    x = torch.zeros(1, 1, 4)

    # Factors should be positive.
    with pytest.raises(ValueError):
        kt.augment.downsample(x, factor=0)

    # Factors should be less than tensor size.
    with pytest.raises(ValueError):
        kt.augment.downsample(x, factor=x.size(-1))


def test_downsample_shared_channels():
    """Test if channels differ, with per-axis factor."""
    inp = torch.rand(1, 1, 4, 4).expand(2, 3, -1, -1)
    out = kt.augment.downsample(inp, factor=(1, 3, 1, 3))

    # Within each batch, all channels should be identical.
    for batch in out:
        assert batch[0].equal(batch[1])
        assert batch[1].equal(batch[2])


def test_remap_unchanged():
    """Test if intensity remapping leaves input unchanged, in 2D."""
    # Do not test on flat image.
    inp = torch.rand(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.remap(inp)
    assert inp.equal(orig)


def test_remap_probability():
    """Test if remapping with zero probability is normalization, in 1D."""
    inp = torch.rand(1, 1, 256)
    out = kt.augment.remap(inp, prob=0)

    inp -= inp.min()
    inp /= inp.max()
    assert out.allclose(inp, rtol=1e-2, atol=1e-2)


def test_remap_shared_channels():
    """Test if channel sharing yields identical channels, in 3D."""
    x = torch.rand(1, 8, 8, 8).expand(4, -1, -1, -1)
    y = kt.augment.remap(x, bins=torch.tensor(99), shared=True, batch=False)

    # Channels should be identical.
    for channel in y[1:]:
        assert channel.allclose(y[0])


def test_crop_unchanged():
    """Test if crop-mask generation leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4)
    orig = inp.clone()
    kt.augment.crop(inp)
    assert inp.equal(orig)


def test_crop_properties():
    """Test type, shape, and channel count after cropping, in 3D."""
    types = (torch.int64, torch.float32, torch.float64)

    for dtype in types:
        inp = torch.ones(2, 3, 4, 4, 4, dtype=dtype)

        out = kt.augment.crop(inp, crop=torch.tensor(1))
        assert out.dtype == inp.dtype
        assert inp[:, 0, ...].shape == out[:, 0, ...].shape


def test_crop_single_axis():
    """Test if cropping operates only along a single axis, in 2D."""
    size = torch.tensor((8, 8))
    inp = torch.ones(1, 1, *size)

    out = kt.augment.crop(inp, crop=0.5)
    low, upp = out.squeeze().nonzero().aminmax(dim=0)
    width = upp - low + 1
    assert width.lt(size).type(torch.int32).sum() <= 1


def test_crop_illegal_values():
    """Test cropping with illegal input arguments, in 2D."""
    x = torch.empty(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.augment.crop(x, crop=-0.1)

    with pytest.raises(ValueError):
        kt.augment.crop(x, crop=1.1)


def test_crop_half():
    """Test cropping half the FOV, with and without batch dimension in 1D."""
    width = torch.tensor(8)
    sizes = {True: (1, 1, width), False: (1, width)}

    for batch, size in sizes.items():
        x = torch.ones(size)

        # Cropping by half the FOV should halve the FOV.
        out = kt.augment.crop(x, crop=(0.5, 0.5), batch=batch)
        assert out.sum().eq(0.5 * width)


def test_crop_return_mask():
    """Test returned cropping mask."""
    inp = torch.ones(1, 2, 3, 3)
    out, mask = kt.augment.crop(inp, crop=(0.5, 0.5), return_mask=True)

    # Expect single-channel mask.
    assert mask.shape == (inp.size(0), 1, *inp.shape[2:])

    # Expect multiplicative application.
    assert out.allclose(inp * mask)


def test_lines_count():
    """Test corrupting lines of tensor with or without batch dimension."""
    sizes = {True: (1, 1, 20), False: (1, 20)}

    for batch, size in sizes.items():
        x = torch.full(size, fill_value=-1)
        out = kt.augment.lines(x, lines=(4, 4), batch=batch)

        # Expect unchanged size, some positive lines. Duplicates possible.
        assert out.shape == size
        assert 1 <= out.squeeze().ge(0).sum() <= 4


def test_lines_probability():
    """Test if line corruption with zero probability is identity."""
    inp = torch.zeros(1, 1, 8, 8)
    out = kt.augment.lines(inp, prob=0)
    assert out.equal(inp)


def test_lines_illegal_value():
    """Test corrupting an illegal number of lines."""
    x = torch.zeros(1, 1, 4, 4)

    # Number of lines should be greater zero.
    with pytest.raises(ValueError):
        kt.augment.lines(x, lines=0)


def test_roll_illegal_value():
    """Test rolling tensors with illegal shift values."""
    x = torch.zeros(1, 1, 4)

    # Should pass 1 or 2 shifts.
    with pytest.raises(ValueError):
        kt.augment.roll(x, shift=(0, 1, 2))

    # Should shift by value in [0, 1].
    with pytest.raises(ValueError):
        kt.augment.roll(x, shift=-1)

    # Should shift by value in [0, 1].
    with pytest.raises(ValueError):
        kt.augment.roll(x, shift=(0, 1.1))


def test_roll_unchanged():
    """Test if rolling leaves input unchanged, in 3D."""
    # Input of shape: batch, channel, space.
    x = torch.ones(1, 1, 2, 2, 2)
    y = x.clone()
    kt.augment.roll(x)
    assert x.equal(y)


def test_roll_properties():
    """Test if rolled type and shape remain the same, in 2D."""
    types = (torch.int64, torch.float32, torch.float64)

    for dtype in types:
        x = torch.ones(2, 3, 4, 4, dtype=dtype)
        y = kt.augment.roll(x, shift=torch.tensor(0.3))
        assert y.dtype == x.dtype
        assert y.shape == x.shape


def test_roll_half():
    """Test effect of rolling by half the tensor, in 1D."""
    x = torch.zeros(1, 1, 10)
    x[..., :5] = 1

    # Roll by half.
    y = kt.augment.roll(x, shift=(0.5, 0.5))

    # Expected results.
    a = x.roll(+5)
    b = x.roll(-5)
    assert y.equal(a) or y.equal(b)


def test_flip_default():
    """Test randomly flipping tensors along default dimension."""
    # Input of shape: batch, channel, space.
    inp = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    flipped = (inp.flip(2), inp.flip(3))

    # Expect flip along first spatial axis or no flip.
    for _ in range(10):
        out = kt.augment.flip(inp)
        assert out.equal(inp) or out.equal(flipped[0])
        assert not out.equal(flipped[1])


def test_flip_dim():
    """Test randomly flipping tensors along specific dimensions."""
    # Input of shape: batch, channel, space.
    inp = torch.arange(4).reshape(1, 1, 2, 2)
    flipped = (inp.flip(2), inp.flip(3))

    for dim in (0, 1):
        for _ in range(10):
            out = kt.augment.flip(inp, dim)
            assert out.equal(inp) or out.equal(flipped[dim])
            assert not out.equal(flipped[1 - dim])


def test_flip_remap():
    """Test tensor flipping with left-right remapping, negative dimension."""
    # Input of shape: batch, channel, space.
    inp = torch.as_tensor((
        (0, 1),
        (2, 3),
    )).unsqueeze(0).unsqueeze(0)

    labels = {0: 'Left-Unknown', 1: 'Right-Unknown', 2: 'Banana'}
    mapped = torch.as_tensor((
        (0, 1),
        (3, 2),
    )).unsqueeze(0).unsqueeze(0)

    # Expect unchanged first line due to 0-1 remapping.
    for _ in range(10):
        out = kt.augment.flip(inp, dim=-1, labels=labels)
        assert out.equal(inp) or out.equal(mapped)
        assert not out.equal(inp.flip(-1))
        assert not out.equal(inp.flip(-2))


def test_flip_illegal_values():
    """Test tensors flipping with illegal arguments."""
    x = torch.zeros(1, 1, 4)

    # Should pass spatial dimension in [0, N).
    with pytest.raises(ValueError):
        kt.augment.flip(x, dim=1)

    with pytest.raises(ValueError):
        kt.augment.flip(x, dim=-2)
