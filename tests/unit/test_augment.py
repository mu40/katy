"""Unit tests for augmentation module."""

import katy as kt
import pytest
import torch


def arange(*size, **kwargs):
    """Return a non-deterministic tensor that does not have flat values."""
    n = torch.tensor(size).prod()
    return torch.arange(n, **kwargs).view(*size)


FUNCTIONS_ALL = {
    kt.augment.gamma,
    kt.augment.noise,
    kt.augment.blur,
    kt.augment.bias,
    kt.augment.downsample,
    kt.augment.remap,
    kt.augment.crop,
    kt.augment.lines,
    kt.augment.roll,
    kt.augment.flip,
    kt.augment.permute,
}


FUNCTIONS_NORM = {
    kt.augment.gamma,
    kt.augment.remap,
}


@pytest.mark.parametrize('func', FUNCTIONS_ALL)
def test_input_unchanged(func):
    """Test if the function does not modify the input."""
    inp = arange(1, 1, 5, 5)
    orig = inp.clone()
    func(inp)
    assert inp.equal(orig)


@pytest.mark.parametrize('func', FUNCTIONS_ALL)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int64])
def test_dtype_same_or_default(func, dtype):
    """Test if functions accept and return the same or default data type."""
    inp = torch.zeros(1, 1, 5, dtype=dtype)
    assert func(inp).dtype in (torch.get_default_dtype(), dtype)


@pytest.mark.parametrize('func', FUNCTIONS_ALL - FUNCTIONS_NORM)
def test_off_identity(func):
    """Test if augmentation with zero probability is identity."""
    inp = arange(1, 1, 5)
    out = func(inp, prob=torch.tensor(0))
    assert out.equal(inp)


@pytest.mark.parametrize('func', FUNCTIONS_NORM)
def test_off_normalization(func):
    """Test if augmentation with zero probability is normalization."""
    inp = arange(1, 1, 2, dtype=torch.float32)
    out = func(inp, prob=0)
    inp -= inp.min()
    inp /= inp.max()
    assert out.equal(inp)


def test_gamma_shared_channels():
    """Test shared gamma augmentation across channels, in 3D."""
    # One batch of two identical input channels.
    inp = arange(1, 1, 3, 3, 3).expand(-1, 2, -1, -1, -1)

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


def test_noise_change():
    """Test if adding noise changes the input, in 2D."""
    inp = torch.zeros(1, 1, 4, 4)
    out = kt.augment.noise(inp)
    assert not out.equal(inp)


def test_blur_illegal_fwhm():
    """Test passing an illegal number of FWHM."""
    x = torch.zeros(1, 1, 4, 4)

    # FWHM should be of length 1, 2, or 2 * N, in N-dimensional space.
    with pytest.raises(ValueError):
        kt.augment.blur(x, fwhm=(1, 2, 3))


def test_blur_shared_channels():
    """Test if the function blurs all channels identically, in 2D."""
    inp = arange(3, 1, 4, 4).expand(-1, 2, -1, -1)
    out = kt.augment.blur(inp, fwhm=5)
    assert out[:, 0].equal(out[:, 1])


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_blur_deterministic(ndim):
    """Test random blurring at pinned down FWHM."""
    # Blur input using specific FWHM.
    fwhm = 5
    inp = arange(1, 1, *[4] * ndim)
    ref = kt.filter.blur(inp, fwhm, dim=range(2, ndim + 2))

    # Sampling FWHM between identical values should yield the same result.
    out = kt.augment.blur(inp, fwhm=(fwhm, fwhm))
    assert out.equal(ref)

    # The same applies when specifying identical bounds for each axis.
    out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm).repeat(2 * ndim))
    assert out.equal(ref)

    # However, expect a different result when sampling between [0, fwhm).
    out = kt.augment.blur(inp, fwhm=torch.tensor(fwhm))
    assert not out.equal(ref)


def test_bias_normalization():
    """Test if fixing minimum yields a bias field between 0 and 1, in 2D."""
    inp = torch.ones(2, 3, 4, 4)
    out = kt.augment.bias(inp, floor=(0, 0), points=torch.tensor((2, 3)))

    # Expect separate normalization of each channel of each batch.
    assert out.amin(dim=(-2, -1)).eq(0).all()
    assert out.amax(dim=(-2, -1)).eq(1).all()


def test_bias_shared_channels():
    """Test if with sharing, batches differ and channels do not, in 1D."""
    x = arange(1, 4).expand(3, -1)
    y = kt.augment.bias(x, floor=0, points=(2, 3), shared=True, batch=False)

    # Channels should be identical.
    assert y[1:].allclose(y[:1])


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
    inp = torch.ones(1, 2, 5, 5)

    # Expect field matching input shape.
    _, field = kt.augment.bias(inp, return_bias=True)
    assert field.shape == inp.shape

    # Expect single-channel field with `shared`.
    out, field = kt.augment.bias(inp, return_bias=True, shared=True)
    assert field.shape == (inp.size(0), 1, *inp.shape[2:])

    # Expect multiplicative application.
    assert out.allclose(inp * field)


def test_downsample_illegal_values():
    """Test downsampling modulation with illegal input arguments, in 1D."""
    # Factors should be positive.
    x = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError):
        kt.augment.downsample(x, factor=0)

    # Factors should be less than tensor size.
    with pytest.raises(ValueError):
        kt.augment.downsample(x, factor=x.size(-1))


def test_downsample_shared_channels():
    """Test if shared channels are identical, with per-axis factor."""
    inp = arange(1, 4, 4).expand(3, -1, -1)
    out = kt.augment.downsample(inp, factor=(1, 3, 1, 3), batch=False)
    out[:1].equal(out[1:])


@pytest.mark.parametrize('mode', ['nearest', 'linear'])
def test_downsample_identity_factor(mode):
    """Test if downsampling with a factor of 1 returns the input."""
    inp = arange(1, 1, 4)
    out = kt.augment.downsample(inp, factor=(1, 1), mode=mode)
    out.equal(inp)


def test_remap_shared_channels():
    """Test if channel sharing yields identical channels, in 3D."""
    x = arange(1, 4, 4, 4).expand(3, -1, -1, -1)
    y = kt.augment.remap(x, bins=torch.tensor(99), shared=True, batch=False)
    assert torch.all(y[0] == y[1:])


@pytest.mark.parametrize('dtype', [torch.int32, torch.float16, torch.float64])
def test_crop_properties(dtype):
    """Test if cropping maintains data type and shape."""
    inp = torch.ones(1, 3, 2, 2, dtype=dtype)
    out = kt.augment.crop(inp, crop=torch.tensor(1))
    assert out.dtype == inp.dtype
    assert inp.shape == out.shape


def test_crop_single_axis():
    """Test if cropping operates only along a single axis."""
    size = torch.tensor((4, 4, 4))
    inp = torch.ones(1, 1, *size)

    # For N-dimensional input tensors `nonzero` returns Z indices of Z non-zero
    # elements. The output is of shape `(Z, N)`.
    out = kt.augment.crop(inp, crop=(0.5, 0.5))
    low, upp = out.squeeze().nonzero().aminmax(dim=0)
    width = upp - low + 1
    assert width.lt(size).count_nonzero() == 1


@pytest.mark.parametrize('crop', [-0.1, 1.1])
def test_crop_illegal_values(crop):
    """Test cropping with illegal input arguments, in 2D."""
    x = torch.empty(1, 1, 4, 4)
    with pytest.raises(ValueError):
        kt.augment.crop(x, crop=crop)


def test_crop_half():
    """Test cropping half the FOV, with and without batch dimension in 1D."""
    width = torch.tensor(8)
    x = torch.ones(1, width)
    out = kt.augment.crop(x, crop=(0.5, 0.5), batch=False)
    assert out.sum().eq(0.5 * width)


def test_crop_return_mask():
    """Test returned cropping mask."""
    inp = torch.ones(1, 2, 3, 3)
    out, mask = kt.augment.crop(inp, crop=(0.5, 0.5), return_mask=True)

    # Expect single-channel mask, multiplicative application.
    assert mask.shape == (inp.size(0), 1, *inp.shape[2:])
    assert out.allclose(inp * mask)


def test_lines_count():
    """Test corrupting lines of tensor without batch dimension."""
    inp = torch.full(size=(1, 20), fill_value=-1)
    out = kt.augment.lines(inp, lines=(4, 4), batch=False)

    # Expect unchanged size, some positive lines. Duplicates possible.
    assert out.shape == inp.shape
    assert out.ge(0).count_nonzero() == 4


def test_lines_illegal_value():
    """Test if corrupting zero lines raises an error."""
    x = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError):
        kt.augment.lines(x, lines=0)


@pytest.mark.parametrize('shift', [(0, 1, 2), -1, (0, 1.1)])
def test_roll_illegal_value(shift):
    """Test rolling tensors with illegal shifts, not 1-2 values in [0, 1]."""
    x = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError):
        kt.augment.roll(x, shift=(0, 1, 2))


@pytest.mark.parametrize('dtype', [torch.int64, torch.float16, torch.float64])
def test_roll_properties(dtype):
    """Test if rolling maintains data type and shape, in 2D."""
    x = torch.ones(2, 3, 4, 4, dtype=dtype)
    y = kt.augment.roll(x, shift=torch.tensor(0.3))
    assert y.dtype == x.dtype
    assert y.shape == x.shape


def test_roll_half():
    """Test effect of rolling by half the width, in 1D."""
    x = torch.zeros(1, 1, 8)
    x[..., :4] = 1
    y = kt.augment.roll(x, shift=(0.5, 0.5))

    # Expect half shift in either direction.
    a = x.roll(+4)
    b = x.roll(-4)
    assert y.equal(a) or y.equal(b)


def test_flip_default():
    """Test randomly flipping tensors along default dimension."""
    # Input of shape: batch, channel, space.
    inp = arange(1, 1, 4, 4)
    flipped = (inp.flip(2), inp.flip(3))

    # Expect flip along first spatial axis or no flip.
    for _ in range(10):
        out = kt.augment.flip(inp)
        assert out.equal(inp) or out.equal(flipped[0])
        assert not out.equal(flipped[1])


@pytest.mark.parametrize('dim', [0, 1])
def test_flip_dim(dim):
    """Test randomly flipping tensors along specific dimensions."""
    inp = arange(1, 1, 4, 4)
    flipped = (inp.flip(2), inp.flip(3))

    for _ in range(10):
        out = kt.augment.flip(inp, dim)
        assert out.equal(inp) or out.equal(flipped[dim])
        assert not out.equal(flipped[1 - dim])


def test_flip_remap():
    """Test tensor flipping with left-right remapping, negative dimension."""
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


@pytest.mark.parametrize('dtype', [torch.int16, torch.float16])
def test_flip_dtype(dtype):
    """Test if flipping maintains data type when remapping."""
    x = torch.zeros((1, 1, 2, 2, 2), dtype=dtype)
    labels = {0: 'Left-Unknown', 1: 'Right-Unknown', 2: 'Banana'}
    for _ in range(10):
        assert kt.augment.flip(x, labels=labels).dtype == x.dtype


@pytest.mark.parametrize('dim', [1, -2])
def test_flip_illegal_dim(dim):
    """Test tensor flipping with spatial dimensions outside [0, N)."""
    x = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError):
        kt.augment.flip(x, dim=dim)


@pytest.mark.parametrize('dtype', [torch.int16, torch.float16, torch.float64])
def test_permute_dtype(dtype):
    """Test if channel permutation maintains data type."""
    x = arange(1, 2, 2)
    y = kt.augment.permute(x)
    assert y.dtype == x.dtype


def test_permute_values():
    """Test if channel permutation maintains shape and elements."""
    x = arange(1, 8)
    y = kt.augment.permute(x)
    assert y.shape == x.shape
    assert y.sort().values.equal(x)
