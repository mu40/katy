"""Unit tests for layers module."""

import katy as kt
import torch


def test_hyper_linear_shape():
    """Test hyper linear transform shape."""
    # `in_features` is the trailing dimension of the input tensor.
    batch = 2
    i = 3
    o = 4

    # Input tensor and hypernetwork output.
    x = torch.zeros(batch, i)
    h = torch.zeros(batch, 5)

    out = kt.layers.HyperLinear(in_features=i, out_features=o)(x, h)
    assert out.shape == (batch, o)


def test_hyper_conv_shape():
    """Test hyper convolution shape, in 2D."""
    batch = 2
    i = 1
    o = 4
    space = (3, 3)

    # Input tensor and hypernetwork output.
    x = torch.zeros(batch, i, *space)
    h = torch.zeros(batch, 5)
    ndim = len(space)

    # Without padding, expect 2 voxels less for kernel size 3.
    out = kt.layers.HyperConv(ndim, i, o, kernel_size=3)(x, h)
    assert out.shape == (batch, o, *(s - 2 for s in space))

    # With padding, expect same size.
    out = kt.layers.HyperConv(ndim, i, o, kernel_size=3, padding='same')(x, h)
    assert out.shape == (batch, o, *space)
