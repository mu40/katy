"""Neural networks."""


import torch
import torch.nn as nn


def make_activation(act, **kwargs):
    """Instantiate activation function, using default values.

    Parameters
    ----------
    act : nn.Module or str or type
        Activation function.
    kwargs : dict, optional
        Key-value settings

    Returns
    -------
    nn.Module
        Configured instance.

    """
    if isinstance(act, nn.Module) or act is None:
        return act

    if isinstance(act, str):
        act = getattr(nn, act)

    if act is nn.Softmax:
        kwargs.setdefault(dim=1)

    if act is nn.LeakyReLU:
        kwargs.setdefault(negative_slope=0.1)

    if any(act in f for f in (nn.ReLU, nn.ELU, nn.LeakyReLU)):
        kwargs.setdefault(in_place=True)

    return act(**kwargs)


class Unet(nn.Module):
    """A simple U-Net (https://arxiv.org/abs/1505.04597)."""

    def __init__(
        self,
        inp=1,
        out=1,
        filters=24,
        mult=2,
        levels=5,
        repeats=1,
        dim=3,
        act=nn.ELU,
        final=nn.Softmax,
    ):
        """Initialize the model.

        Parameters
        ----------
        inp : int, optional
            Number of input channels.
        out : int, optional
            Number of output channels.
        filters : int, optional
            Number of features at the highest (first) level.
        mult : int, optional
            Feature multiplier. Multiply the number of convolutional features
            by this number each time you descend one level.
        levels : int, optional
            Number of U-Net levels, including the highest level.
        repeats : int, optional
            Number of convolutional layers at each encoder and decoder level.
        dim : int, optional
            Number of spatial dimensions.
        act : str or nn.Module or type, optional
            Activation function.
        final : str or nn.Module or type or None, optional
            Final activaton function.

        """
        super().__init__()

        # Layers.
        pool = getattr(nn, f'MaxPool{dim}d')
        conv = getattr(nn, f'Conv{dim}d')
        prop = dict(kernel_size=3, padding='same')

        # Encoder.
        n_inp = inp
        n_out = filters
        n_enc = []
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for _ in range(levels - 1):
            level = []
            for _ in range(repeats):
                level.append(conv(n_inp, n_out, **prop))
                level.append(make_activation(act))
                n_inp = n_out
            self.enc.append(nn.Sequential(*level))
            self.down.append(pool(kernel_size=2))
            n_enc.append(n_out)
            n_out *= mult

        # Decoder.
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for _ in range(levels - 1):
            level = []
            for _ in range(repeats):
                level.append(conv(n_inp, n_out, **prop))
                level.append(make_activation(act))
                n_inp = n_out
            n_inp += n_enc.pop()
            self.dec.append(nn.Sequential(*level))
            self.up.append(nn.Upsample(scale_factor=2))
            n_out //= mult

        # Final layers.
        out = []
        for _ in range(repeats - 1):
            out.append(conv(n_inp, n_out, **prop))
            out.append(nn.ReLU())
            n_inp = n_out
        out.append(conv(n_inp, out, **prop))
        if final is not None:
            out.append(make_activation(final))
        self.out = nn.Sequential(*out)

    def forward(self, x):
        """Define the computation performed by the model call.

        Parameters
        ----------
        x : (batch, inp, *space) torch.Tensor
            Input tensor.

        Returns
        -------
        out : (batch, out, *space) torch.Tensor
            Model output.

        """
        # Encoder.
        hist = []
        for conv, down in zip(self.enc, self.down):
            x = conv(x)
            hist.append(x)
            x = down(x)

        # Decoder.
        for conv, up in zip(self.dec, self.up):
            x = conv(x)
            x = torch.cat([hist.pop(), up(x)], dim=1)

        return self.out(x)
