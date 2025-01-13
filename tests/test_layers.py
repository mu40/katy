"""Tests for label-manipulation and image-synthesis module."""


import torch
import katy as kt


def test_hyper_linear_shape():
    """Test hyper linear transform shape."""
    # `in_features is the trailing dimension of the input tensor`.
    batch = 2
    f_in = 3
    f_out = 4

    # Input tensor and hypernetwork output.
    x = torch.zeros(batch, f_in)
    h = torch.zeros(batch, 5)

    out = kt.layers.HyperLinear(in_features=f_in, out_features=f_out)(x, h)
    assert out.shape == (batch, f_out)


def test_hyper_conv_shape():
    """Test hyper convolution shape, in 2D."""
    # `in_features is the trailing dimension of the input tensor`.
    batch = 2
    ci = 1
    co = 4
    space = (3, 3)

    # Input tensor and hypernetwork output.
    x = torch.zeros(batch, ci, *space)
    h = torch.zeros(batch, 5)
    dim = len(space)

    # Without padding, expect 2 voxels less for kernel size 3.
    out = kt.layers.HyperConv(dim, ci, co, kernel_size=3)(x, h)
    assert out.shape == (batch, co, *(s - 2 for s in space))

    # With padding, expect same size.
    out = kt.layers.HyperConv(dim, ci, co, kernel_size=3, padding='same')(x, h)
    assert out.shape == (batch, co, *space)
