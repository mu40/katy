"""Filter module."""


import torch


def gaussian_kernel(fwhm, width=None, device=None):
    """Construct a centered, one-dimensional Gaussian kernel.

    Clamps the standard deviation `sd` to a minimum value of 1e-6.

    Parameters
    ----------
    fwhm : float
       Full width at half maximum.
    width : int, optional
        Kernel size. Defaults to `int(3 * sd) * 2 + 1`.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    (width,) torch.Tensor
        Gaussian kernel.

    """
    fwhm = torch.as_tensor(fwhm, device=device).squeeze()
    sd = fwhm.div(2.3548).clamp(min=1e-6)

    if width is None:
        width = torch.round(3 * sd) * 2 + 1
    width = torch.as_tensor(width, device=sd.device).squeeze()

    x = torch.arange(width, device=sd.device) - 0.5 * (width - 1)
    kern = torch.exp(-0.5 * x.div(sd).pow(2))
    return kern.div(kern.sum())


def blur(x, fwhm, dim=None):
    """Blur an N-dimensional tensor by convolving it with a Gaussian.

    Uses explicit convolutions in the spatial domain.

    Parameters
    ----------
    x : torch.Tensor
       Input tensor.
    fwhm : float or sequence of float
        Full widths at half maximum of the Gaussian. Apply to `dim`, in order.
    dim : int or sequence of int, optional
        Dimensions to blur along. None means all.

    Returns
    -------
    torch.Tensor
        Blurred version of the input tensor.

    """
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())

    # Vector of dimensions.
    if dim is None:
        dim = torch.arange(x.ndim)
    dim = torch.as_tensor(dim).ravel()

    # One FWHM per dimension.
    fwhm = torch.as_tensor(fwhm).ravel().expand(dim.shape)
    for i, f in zip(dim, fwhm):
        kern = gaussian_kernel(f, device=x.device).view(1, 1, -1)

        # Move axis to end, make everything else batch, convolve, and restore.
        tmp = x.transpose(i, -1)
        x = tmp.reshape(-1, 1, tmp.size(-1))
        x = torch.nn.functional.conv1d(x, kern, padding='same')
        x = x.view_as(tmp).transpose(i, -1)

    return x
