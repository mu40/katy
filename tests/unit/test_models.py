"""Unit tests for models module."""

import katy as kt
import pytest
import torch


def test_count():
    """Test counting model parameters."""
    layer = torch.nn.Linear(in_features=1, out_features=1)

    # Expect only trainable parameters.
    assert kt.models.count(layer) == 2
    assert kt.models.count(layer, grad_only=True) == 2


def test_count_frozen():
    """Test counting frozen parameters."""
    layer = torch.nn.Linear(in_features=1, out_features=1)
    for p in layer.parameters():
        p.requires_grad = False

    # Expect only frozen parameters.
    assert kt.models.count(layer) == 2
    assert kt.models.count(layer, grad_only=True) == 0


@pytest.mark.parametrize('name', ['ReLU', 'LeakyReLU', 'Softmax', 'Sigmoid'])
def test_make_activation_create(name):
    """Test making activation functions from strings and types."""
    t = getattr(torch.nn, name)

    # Passing the name should yield an instance.
    out = kt.models.make_activation(name)
    assert isinstance(out, t)

    # Passing the type should yield an instance.
    out = kt.models.make_activation(t)
    assert isinstance(out, t)


def test_make_activation_instance():
    """Test if the function returns activation function instances as is."""
    x = torch.nn.Sigmoid()
    assert kt.models.make_activation(x) is x


def test_unet_module_first_layer():
    """Test setting number of U-Net input channels."""
    dim = 1
    inp = 3

    # Get first layer.
    first = kt.models.Unet(dim=dim, inp=inp, enc=(1,), dec=(1,))
    while len(list(first.children())) > 0:
        first = next(first.children())

    # Expect a convolution of specified filters.
    assert isinstance(first, getattr(torch.nn, f'Conv{dim}d'))
    assert first.in_channels == inp


def test_unet_last_layer():
    """Test setting number of U-Net output channels, no final activation."""
    dim = 2
    out = 4

    # Get last layer.
    last = kt.models.Unet(dim=dim, out=out, enc=(1,), dec=(1,), fin=None)
    while len(list(last.children())) > 0:
        last = next(reversed(list(last.children())))

    # Without final activation, expect a convolution of specified filters.
    assert isinstance(last, getattr(torch.nn, f'Conv{dim}d'))
    assert last.out_channels == out


def test_unet_inference_shape():
    """Test U-Net inference and output shape."""
    dim = 3
    size = (4,) * dim
    x = torch.zeros(2, 1, *size)

    # Expect output of input shape for balanced down and upsampling.
    model = kt.models.Unet(dim, enc=(1, 1), dec=(1, 1), act=torch.nn.ReLU)
    assert model(x).shape[2:] == size

    # Expect half-size output when omitting last decoder level.
    model = kt.models.Unet(dim, enc=(1, 1), dec=(1,), add=(1,))
    assert model(x).shape[2:] == tuple(s // 2 for s in size)
