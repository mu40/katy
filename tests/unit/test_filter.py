"""Unit tests for filter module."""

import katy as kt
import pytest
import torch


def test_gaussian_kernel_properties():
    """Test kernel width control, normalization, dimensionality, data type."""
    width = 13
    k = kt.filter.gaussian_kernel(fwhm=8, width=13)
    assert k.numel() == width
    assert k.ndim == 1
    assert k.dtype == torch.get_default_dtype()
    assert k.sum().sub(1).abs() < 1e-6


def test_gaussian_kernel_default_width_odd():
    """Test if the default kernel size is odd."""
    assert kt.filter.gaussian_kernel(fwhm=8).numel() % 2 == 1


def test_blur_unchanged():
    """Test if blurring leaves the input changed."""
    inp = torch.zeros(2, 3, 4)
    fwhm = torch.tensor(1)
    orig = inp.clone()
    kt.filter.blur(inp, fwhm)
    assert inp.equal(orig)


def test_blur_fwhm_zero():
    """Test if blurring with zero FWHM has no effect."""
    x = torch.ones(2, 3, 4)
    assert kt.filter.blur(x, fwhm=0).allclose(x)


def test_blur_default_dtype():
    """Test the default output data type."""
    x = torch.zeros(5)
    assert kt.filter.blur(x, fwhm=2).dtype == torch.get_default_dtype()


@pytest.mark.parametrize('dtype', [torch.int32, torch.float32])
def test_dilate_trivial(dtype):
    """Test dilation output data type and values."""
    x = torch.tensor((0.0, 0.0, 2.2, 3.3), dtype=dtype)
    y = kt.filter.dilate(x, dim=0)
    assert y.dtype == x.dtype
    assert torch.isin(y, torch.tensor((0, 1))).all()


def test_dilate_iterations():
    """Test iterative dilation, in 1D."""
    x = torch.tensor((0, 0, 1, 0, 0))
    assert kt.filter.dilate(x, n=0).equal(x)
    assert kt.filter.dilate(x, n=1).equal(torch.tensor((0, 1, 1, 1, 0)))
    assert kt.filter.dilate(x, n=2).equal(torch.tensor((1, 1, 1, 1, 1)))


def test_dilate_dimensions():
    """Test dilating specific dimensions, in 2D."""
    # Expected input and output.
    a = torch.tensor((
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    ))
    b = torch.tensor((
        (0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0),
    ))

    # Copies along dimension 0.
    inp = torch.zeros(3, 2, *a.shape)
    inp[:, 0] = a
    out = kt.filter.dilate(inp, dim=(3, 2))

    # Expect zeros in second channel, identical results for each batch.
    assert out[:, 1].eq(0).all()
    assert torch.all(out[:, 0] == b)


@pytest.mark.parametrize('dtype', [torch.long, torch.float32])
def test_erode_trivial(dtype):
    """Test erosion output data type and values."""
    x = torch.tensor((0.0, 1.1, 2.2, 3.3), dtype=dtype)
    y = kt.filter.erode(x, dim=-1)
    assert y.dtype == x.dtype
    assert torch.isin(y, torch.tensor((0, 1))).all()


def test_erode_iterations():
    """Test iterative erosion, in 1D. See Scipy examples."""
    x = torch.tensor((0, 1, 1, 1, 1, 1))
    assert kt.filter.erode(x, n=0).equal(x)
    assert kt.filter.erode(x, n=1).equal(torch.tensor((0, 0, 1, 1, 1, 0)))
    assert kt.filter.erode(x, n=2).equal(torch.tensor((0, 0, 0, 1, 0, 0)))


def test_erode_dimensions():
    """Test eroding specific dimensions, in 2D. See Scipy examples."""
    # Expected input and output.
    a = torch.tensor((
        (0, 0, 0, 0),
        (0, 1, 1, 1),
        (0, 1, 1, 1),
        (0, 1, 1, 1),
    ))
    b = torch.tensor((
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 0),
    ))

    # Copies along dimension 2.
    inp = torch.zeros(*a.shape, 3, 2)
    inp[..., 0, 0] = a
    inp[..., 1, 0] = a
    inp[..., 2, 0] = a
    out = kt.filter.erode(inp, dim=(0, 1))

    # Expect only zeros at end of dimension 3.
    assert out[..., :, 1].eq(0).all()

    # Expect identical results in dimension 2.
    for i in range(out.shape[2]):
        assert out[..., i, 0].equal(b)


def test_close_1d():
    """Test closing, in 1D. See Scipy examples."""
    a = torch.tensor((0, 0, 1, 1, 0, 1))
    b = torch.tensor((0, 0, 1, 1, 1, 0))

    assert kt.filter.close(a, n=0).equal(a)
    assert kt.filter.close(a, n=1).equal(b)


def test_close_2d():
    """Test closing, in 2D. See Scipy examples."""
    a = torch.tensor((
        (0, 0, 0, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 0, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 0, 0, 0),
    ))
    b = torch.tensor((
        (0, 0, 0, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 0, 0, 0),
    ))
    assert kt.filter.close(a).equal(b)


def test_open_1d():
    """Test opening, in 1D. See Scipy examples."""
    a = torch.tensor((0, 1, 1, 1, 0, 1))
    b = torch.tensor((0, 1, 1, 1, 0, 0))
    assert kt.filter.open(a, n=0).equal(a)
    assert kt.filter.open(a, n=1).equal(b)


def test_open_2d():
    """Test opening, in 2D. See Scipy examples."""
    a = torch.tensor((
        (1, 0, 0, 0, 1),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (0, 1, 1, 1, 0),
        (1, 0, 0, 0, 1),
    ))
    b = torch.tensor((
        (0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 1, 1, 1, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0),
    ))
    assert kt.filter.open(a).equal(b)


@pytest.mark.parametrize('dtype', [torch.long, torch.float16])
def test_fill_holes_trivial(dtype):
    """Test hole-filling output data type and values, in 1D."""
    x = torch.tensor((0.0, 1.1, 0.0, 3.3), dtype=dtype)
    y = kt.filter.fill_holes(x, dim=0)
    assert y.dtype == x.dtype
    assert y.equal(torch.tensor((0, 1, 1, 1)))


def test_fill_holes_dimensions():
    """Test hole-filling along specific dimensions, in 2D."""
    # Expected input and output.
    a = torch.tensor((
        (0, 0, 1, 1, 1),
        (0, 0, 1, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 1, 1, 1, 1),
        (0, 1, 1, 0, 0),
    ))
    b = torch.tensor((
        (0, 0, 1, 1, 1),
        (0, 0, 1, 1, 0),
        (0, 1, 1, 1, 1),
        (0, 1, 1, 1, 1),
        (0, 1, 1, 0, 0),
    ))

    # Copies along dimension 0.
    inp = torch.zeros(3, 2, *a.shape)
    inp[:, 0] = a
    out = kt.filter.fill_holes(inp, dim=(-2, -1))

    # Expect zeros in second channel, identical results for each batch.
    assert out[:, 1].eq(0).all()
    assert torch.all(out[:, 0] == b)


def test_fill_holes_unchanged():
    """Test hole-filling without holes has no effect, in 3D."""
    inp = torch.ones(4, 4, 4)
    out = kt.filter.fill_holes(inp)
    assert out.equal(inp)

    # Expect a copy.
    assert out is not inp


def test_fill_holes_illegal():
    """Test if hole-filling a 4D tensor raises an error."""
    x = torch.zeros(2, 2, 2, 2)

    with pytest.raises(AttributeError):
        kt.filter.fill_holes(x)


def test_label_flat():
    """Test labeling flat tensors, including types."""
    x = torch.tensor((0, 0, 0, 0))
    labels, values, sizes = kt.filter.label(x)
    assert labels.dtype == torch.long
    assert labels.eq(0).all()
    assert values.numel() == 0
    assert sizes.numel() == 0

    x = torch.tensor((2.2, 2.2, 2.2))
    labels, values, sizes = kt.filter.label(x)
    assert labels.dtype == torch.long
    assert labels.eq(1).all()
    assert values.equal(torch.tensor([1]))
    assert sizes.equal(torch.tensor([3]))


def test_label_1d():
    """Test labeling a 1D tensor."""
    x = torch.tensor((0, 1, 1, 0, 1, 2, 3, 0, 9))
    labels, values, sizes = kt.filter.label(x)

    # Expect islands labeled by decreasing size.
    assert labels.equal(torch.tensor((0, 2, 2, 0, 1, 1, 1, 0, 3)))
    assert values.equal(torch.tensor((1, 2, 3)))
    assert sizes.equal(torch.tensor((3, 2, 1)))


def test_label_2d():
    """Test labeling a 2D tensor."""
    # Expected input and output.
    a = torch.tensor((
        (1, 4, 0, 0, 5),
        (2, 3, 0, 0, 5),
        (0, 0, 9, 0, 5),
        (0, 0, 0, 0, 5),
        (5, 5, 0, 0, 5),
    ))
    b = torch.tensor((
        (2, 2, 0, 0, 1),
        (2, 2, 0, 0, 1),
        (0, 0, 4, 0, 1),
        (0, 0, 0, 0, 1),
        (3, 3, 0, 0, 1),
    ))

    # Expect islands labeled by decreasing size.
    labels, values, sizes = kt.filter.label(a)
    assert labels.equal(b)
    assert values.equal(torch.tensor((1, 2, 3, 4)))
    assert sizes.equal(torch.tensor((5, 4, 2, 1)))
