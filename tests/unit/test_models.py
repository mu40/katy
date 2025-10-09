"""Unit tests for models module."""

import katy as kt
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


def test_make_activation_create():
    """Test making activation functions from strings and types."""
    names = ('ReLU', 'ELU', 'LeakyReLU', 'Softmax', 'Sigmoid')

    for name in names:
        act = getattr(torch.nn, name)

        # Passing the name should yield an instance.
        out = kt.models.make_activation(name)
        assert isinstance(out, act)

        # Passing the type should yield an instance.
        out = kt.models.make_activation(act)
        assert isinstance(out, act)


def test_make_activation_instance():
    """Test if the function returns activation function instances as is."""
    inp = torch.nn.Sigmoid()
    assert kt.models.make_activation(inp) is inp


def test_unet_module_first_layer_1d():
    """Test setting number of U-Net input channels."""
    dim = 1
    inp = 3

    # Get first layer.
    layer = kt.models.Unet(dim=dim, inp=inp, enc=(1,), dec=(1,))
    while len(list(layer.children())) > 0:
        layer = next(layer.children())

    # Expect a convolution of specified filters.
    assert isinstance(layer, getattr(torch.nn, f'Conv{dim}d'))
    assert layer.in_channels == inp


def test_unet_last_layer_2d():
    """Test setting number of U-Net output channels, no final activation."""
    dim = 2
    out = 4

    # Get last layer.
    layer = kt.models.Unet(dim=dim, out=out, enc=(1,), dec=(1,), fin=None)
    while len(list(layer.children())) > 0:
        layer = next(reversed(list(layer.children())))

    # Without final activation, expect a convolution of specified filters.
    assert isinstance(layer, getattr(torch.nn, f'Conv{dim}d'))
    assert layer.out_channels == out


def test_unet_inference_shape_3d():
    """Test U-Net inference and output shape."""
    size = (4, 4, 4)
    x = torch.zeros(2, 1, *size)

    # Expect output of input shape for balanced down and upsampling.
    model = kt.models.Unet(dim=3, enc=(1, 1), dec=(1, 1), act=torch.nn.ReLU)
    assert model(x).shape == x.shape

    # Expect half-size output when omitting last decoder level.
    model = kt.models.Unet(dim=3, enc=(1, 1), dec=(1,), add=(1,))
    assert model(x).shape[2:] == tuple(s // 2 for s in size)
