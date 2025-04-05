"""Building blocks for neural networks."""

import functools
import torch
import torch.nn as nn


class HyperConv(nn.Module):
    """Hyper convolution."""

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size,
        *,
        bias=True,
        groups=1,
        **kwargs,
    ):
        """Initialize a hyper convolution, similarly to `torch.nn.Conv{N}d`.

        The module uses `torch.nn.LazyLinear` internally, so it needs a dry
        run on a batch of the correct size for initialization.

        Parameters
        ----------
        ndim : {1, 2, 3}
            Dimensionality.
        in_channels : int
            Hyper convolution input channels.
        out_channels : int
            Hyper convolution output channels.
        kernel_size : int or tuple of int
            Kernel size, of length 1 or `ndim`.
        bias : bool, optional
            Have the hyper convolution apply an additive bias.
        groups : int, optional.
            Number of groups.
        **kwargs : dict, optional
            Key-value `torch.nn.functional.conv{N}d` settings.

        """
        super().__init__()
        self.conv = getattr(nn.functional, f'conv{ndim}d')
        self.conv = functools.partial(self.conv, groups=groups, **kwargs)
        self.cache = []

        # Dimensions.
        size = torch.tensor(kernel_size).ravel().expand(ndim)
        self.size = torch.tensor((out_channels, in_channels // groups, *size))

        # Weight and bias.
        self.hyperweight = nn.LazyLinear(torch.prod(self.size))
        if bias:
            self.hyperbias = nn.LazyLinear(out_channels)

    def forward(self, x, h, reuse=False):
        """Map and apply convolutional weights from a hypernetwork output.

        Parameters
        ----------
        x : (B, in_channels, *size) torch.Tensor
            Input of N-element spatial size.
        h : (B, ...) torch.Tensor
            Hypernetwork output.
        reuse : bool, optional
            Reuse any prior weight and bias, ignoring `h` for serial test.

        Returns
        -------
        (B, out_channels, ...) torch.Tensor
            Output of spatial size depending on `kwargs`.

        """
        batch = h.size(0)
        h = h.view(batch, -1)

        # Hyper bias has shape `(B, self.out`) already.
        if reuse and self.cache:
            par = self.cache

        else:
            par = [self.hyperweight(h).view(batch, *self.size)]
            if hasattr(self, 'hyperbias'):
                par.append(self.hyperbias(h))
            self.cache = par

        # Dummy batch for convolution function.
        x = x.unsqueeze(1)
        return torch.cat([self.conv(*f) for f in zip(x, *par)])


class HyperLinear(nn.Module):
    """Hyper linear transform."""

    def __init__(self, in_features, out_features, bias=True):
        """Initialize a hyper linear transform, similarly to `torch.nn.Linear`.

        The module uses `torch.nn.LazyLinear` internally, so it needs a dry
        run on a batch of the correct size for initialization.

        Parameters
        ----------
        in_features : int
            Trailing dimension of the hyper transform input.
        out_features : int
            Trailing dimension of the hyper transform output.
        bias : bool, optional
            Have the hyper linear layer apply an additive bias.

        """
        super().__init__()
        self.inp = in_features
        self.out = out_features
        self.cache = []

        # A linear layer taking inputs of size `(*, in)` and returning outputs
        # of size `(*, out)` has an `(out, in)` weight and an `(out,)` bias.
        # To return a `(B, out, in)` weight, the internal linear layer needs
        # `out * in` output features. We always use internal layers with bias.
        self.hyperweight = nn.LazyLinear(in_features * out_features)
        if bias:
            self.hyperbias = nn.LazyLinear(out_features)

    def forward(self, x, h, reuse=False):
        """Map and apply linear transform weights from a hypernetwork output.

        Parameters
        ----------
        x : (B, ..., in_features) torch.Tensor
            Input.
        h : (B, ...) torch.Tensor
            Hypernetwork output.
        reuse : bool, optional
            Reuse any prior weight and bias, ignoring `h` for serial test.

        Returns
        -------
        (B, ..., out_features) torch.Tensor
            Output.

        """
        # Fully connect hypernetwork output.
        batch = h.size(0)
        h = h.view(batch, -1)

        # Hyper bias has shape `(B, self.out`) already.
        if reuse and self.cache:
            par = self.cache

        else:
            par = [self.hyperweight(h).view(batch, self.out, self.inp)]
            if hasattr(self, 'hyperbias'):
                par.append(self.hyperbias(h))
            self.cache = par

        return torch.stack([nn.functional.linear(*f) for f in zip(x, *par)])
