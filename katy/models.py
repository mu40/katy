"""Neural networks."""

import katy as kt
import torch
import torch.nn as nn


def count(model, /, grad_only=False):
    """Count the number of module parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Layer or model.
    grad_only : bool, optional
        Only count trainable parameters.

    Returns
    -------
    int
        Number of parameters.

    """
    par =  model.parameters()
    if grad_only:
        par = filter(lambda p: p.requires_grad, par)

    return sum(p.numel() for p in par)


def make_activation(act, /, **kwargs):
    """Instantiate activation function with overridable default.

    Parameters
    ----------
    act : nn.Module or str or type
        Activation function.
    **kwargs : dict, optional
        Key-value settings.

    Returns
    -------
    nn.Module
        Configured instance.

    """
    if isinstance(act, nn.Module):
        return act

    if isinstance(act, str):
        act = getattr(nn, act)

    if act is nn.Softmax:
        kwargs.setdefault('dim', 1)

    if act is nn.LeakyReLU:
        kwargs.setdefault('negative_slope', 0.1)

    if any(act == f for f in (nn.ReLU, nn.ELU, nn.LeakyReLU)):
        kwargs.setdefault('inplace', True)

    return act(**kwargs)


class Unet(nn.Module):
    """A simple N-dimensional U-Net (https://arxiv.org/abs/1505.04597)."""

    def __init__(
        self,
        dim=3,
        inp=1,
        out=1,
        enc=(24, 48, 96, 192, 384),
        dec=(384, 192, 96, 48, 24),
        add=(),
        rep=1,
        act=nn.ELU,
        fin=nn.Softmax,
        clip=(None, None),
    ):
        """Initialize the model.

        Parameters
        ----------
        dim : int, optional
            Number of spatial dimensions N.
        inp : int, optional
            Number of input channels.
        out : int, optional
            Number of output channels.
        enc : sequence of int, optional
            Number of encoding convolutional filters at each level.
        dec : sequence of int, optional
            Number of decoding convolutional filters at each level.
        add : sequence of int, optional
            Number of additional convolutional filters at the final level.
        rep : int, optional
            Number of repeats for each convolutional operation.
        act : str or nn.Module or type, optional
            Activation function after each convolution.
        fin : str or nn.Module or type or None, optional
            Final activaton function.
        clip : tuple of float, optional
            Clip min-max quantiles in [0, 1] before normalizing.

        """
        super().__init__()
        self.clip = clip

        # Layers.
        pool = getattr(nn, f'MaxPool{dim}d')
        conv = getattr(nn, f'Conv{dim}d')
        prop = dict(kernel_size=3, padding='same')

        # Encoder.
        n_inp = inp
        enc = list(enc)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for n_out in enc:
            level = []
            for _ in range(rep):
                level.append(conv(n_inp, n_out, **prop))
                level.append(make_activation(act))
                n_inp = n_out
            self.enc.append(nn.Sequential(*level))
            self.down.append(pool(kernel_size=2))

        # Decoder.
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for n_out in dec:
            level = []
            for _ in range(rep):
                level.append(conv(n_inp, n_out, **prop))
                level.append(make_activation(act))
                n_inp = n_out
            n_inp += enc.pop()
            self.dec.append(nn.Sequential(*level))
            self.up.append(nn.Upsample(scale_factor=2))

        # Additional convolutions.
        level = []
        for n_out in add:
            for _ in range(rep):
                level.append(conv(n_inp, n_out, **prop))
                level.append(make_activation(act))
                n_inp = n_out

        level.append(conv(n_inp, out, **prop))
        if fin is not None:
            level.append(make_activation(fin))

        self.add = nn.Sequential(*level)

    def forward(self, x, /):
        """Define the computation performed by the model call.

        Parameters
        ----------
        x : (batch, inp, ...) torch.Tensor
            Input tensor.

        Returns
        -------
        out : (batch, out, ...) torch.Tensor
            Model output.

        """
        with torch.no_grad():
            dim = range(2, x.ndim)
            x = kt.utility.normalize(x, dim, *self.clip)

        # Encoding convolutions.
        enc = []
        for conv, down in zip(self.enc, self.down, strict=True):
            x = conv(x)
            enc.append(x)
            x = down(x)

        # Decoding convolutions.
        for conv, up in zip(self.dec, self.up, strict=True):
            x = conv(x)
            x = torch.cat([enc.pop(), up(x)], dim=1)

        # Additional convolutions.
        return self.add(x)
