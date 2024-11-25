"""Tests for filter module."""


import torch
import katy as kt


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
    inp = torch.zeros(3, 4, 5)
    fwhm = torch.tensor(1)
    orig = inp.clone()
    kt.filter.blur(inp, fwhm)
    assert inp.eq(orig).all()


def test_blur_fwhm_zero():
    """Test if blurring with zero FWHM has no effect."""
    x = torch.rand(3, 4, 5)
    assert kt.filter.blur(x, fwhm=0).allclose(x)


def test_blur_default_dtype():
    """Test the default output data type."""
    x = torch.randint(10, size=(5,))
    assert kt.filter.blur(x, fwhm=2).dtype == torch.get_default_dtype()
