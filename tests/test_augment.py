"""Tests for augmentation module."""


import torch
import pytest
import katy as kt


def test_gamma_unchanged():
    """Test if gamma-augmentation leaves input unchanged, in 2D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.gamma(inp)
    assert inp.eq(orig).all()


def test_gamma_probability():
    """Test if gamma with zero probability is normalization, in 1D."""
    inp = torch.rand(1, 1, 8)
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
    """Test shared gamma augmentation across channels, in 3D."""
    # One batch of two identical input channels.
    inp = torch.rand(1, 1, 3, 3, 3).expand(-1, 2, -1, -1, -1).clone()

    # Identical channels after shared augmentation.
    out = kt.augment.gamma(inp, shared=True).squeeze()
    assert out[0].allclose(out[1])

    # Different channels after separate augmentation.
    out = kt.augment.gamma(inp, shared=False).squeeze()
    assert out[0].ne(out[1]).any()


def test_gamma_illegal_values():
    """Test passing gamma values outside (0, 1) range."""
    x = torch.zeros(1, 1, 4, 4)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=0)

    with pytest.raises(ValueError):
        kt.augment.gamma(x, gamma=1)


def test_noise_unchanged():
    """Test if adding noise leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4)

    orig = inp.clone()
    kt.augment.noise(inp)
    assert inp.eq(orig).all()


def test_noise_change():
    """Test if adding noise changes the input, in 2D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4, 4)

    # Adding noise should change the input.
    out = kt.augment.noise(inp, prob=1)
    assert out.ne(inp).any()

    # Nothing should change at zero probability.
    out = kt.augment.noise(inp, prob=torch.tensor(0))
    assert out.eq(inp).all()


def test_blur_unchanged():
    """Test if randomly blurring leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.zeros(1, 1, 4)

    orig = inp.clone()
    kt.augment.blur(inp, fwhm=1)
    assert inp.eq(orig).all()


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
        assert batch[0].eq(batch[1]).all()


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
        assert out.eq(ref).all()

        # The same applies when specifying identical bounds for each axis.
        out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm).repeat(2 * dim))
        assert out.eq(ref).all()

        # However, expect a different result when sampling between [0, fwhm).
        out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm))
        assert out.ne(ref).any()


def test_bias_unchanged():
    """Test if bias leaves input unchanged, with tensor inputs in 3D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4, 4, 4)

    orig = inp.clone()
    kt.augment.bias(inp, floor=torch.tensor(0), points=torch.tensor((2, 3)))
    assert inp.eq(orig).all()


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
    inp = torch.rand(1, 1, 4).expand(2, 3, -1)
    out = kt.augment.bias(inp, floor=0, points=(2, 3), shared=True)

    # Batches should differ.
    for channel in range(inp.shape[1]):
        assert out[0, channel].ne(out[1, channel]).any()

    # Within each batch, all channels should be identical.
    for batch in out:
        assert batch[0].eq(batch[1]).all()
        assert batch[1].eq(batch[2]).all()


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
    """Test if bias leaves input unchanged, with tensor input and in 3D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4, 4, 4)

    orig = inp.clone()
    kt.augment.downsample(inp, factor=torch.tensor(2))
    assert inp.eq(orig).all()


def test_downsample_illegal_values():
    """Test bias modulation with illegal input arguments, in 1D."""
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
        assert batch[0].eq(batch[1]).all()
        assert batch[1].eq(batch[2]).all()


def test_remap_unchanged():
    """Test if intensity remapping leaves input unchanged, in 2D."""
    # Do not test on flat image.
    inp = torch.rand(1, 1, 8, 8)

    orig = inp.clone()
    kt.augment.remap(inp)
    assert inp.eq(orig).all()


def test_remap_probability():
    """Test if remapping with zero probability is normalization, in 1D."""
    inp = torch.rand(1, 1, 256)
    out = kt.augment.remap(inp, prob=0)

    inp -= inp.min()
    inp /= inp.max()
    assert out.allclose(inp, rtol=1e-2, atol=1e-2)


def test_remap_shared_channels():
    """Test if with sharing, batches differ and channels do not, in 3D."""
    inp = torch.rand(1, 1, 8, 8, 8).expand(2, 3, -1, -1, -1)
    out = kt.augment.remap(inp, bins=torch.tensor(128), shared=True)

    # Batches should differ.
    for channel in range(inp.shape[1]):
        assert out[0, channel].ne(out[1, channel]).any()

    # Within each batch, all channels should be identical.
    for batch in out:
        assert batch[0].eq(batch[1]).all()
        assert batch[1].eq(batch[2]).all()


def test_crop_unchanged():
    """Test if crop-mask generation leaves input unchanged, in 1D."""
    # Input of shape: batch, channel, space.
    inp = torch.ones(1, 1, 4)
    orig = inp.clone()
    kt.augment.crop(inp)
    assert inp.eq(orig).all()


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
    assert out.eq(inp).all()


def test_lines_illegal_value():
    """Test corrupting an illegal number of lines."""
    x = torch.zeros(1, 1, 4, 4)

    # Number of lines should be greater zero.
    with pytest.raises(ValueError):
        kt.augment.lines(x, lines=0)
