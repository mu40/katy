"""Filter module."""

import torch


def gaussian_kernel(fwhm, width=None, *, device=None):
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


def blur(x, /, fwhm, dim=None):
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
    for i, f in zip(dim, fwhm, strict=True):
        kern = gaussian_kernel(f, device=x.device).view(1, 1, -1)

        # Move axis to end, make everything else batch, convolve, and restore.
        tmp = x.transpose(i, -1)
        x = tmp.reshape(-1, 1, tmp.size(-1))
        x = torch.nn.functional.conv1d(x, kern, padding='same')
        x = x.view_as(tmp).transpose(i, -1)

    return x


def dilate(x, /, n=1, *, dim=None):
    """Perform binary morphological dilation in 1D, 2D, or 3D.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. We consider non-zero voxels foreground.
    n : int, optional
        Repeat the operation `n` times.
    dim : int or sequence of int, optional
        Operate along 1 to 3 dimensions. None means all.

    Returns
    -------
    torch.Tensor
        Dilated tensor.

    """
    # Input.
    dtype = x.dtype
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())

    # Dimensions.
    if dim is None:
        dim = torch.arange(x.ndim)
    dim = torch.as_tensor(dim).ravel().tolist()
    ndim = len(dim)

    # Kernel: "4-neighborhood" in 2D.
    kern = torch.zeros([3] * ndim)
    for i in range(ndim):
        kern[(*[1] * i, slice(None), *[1] * (ndim - i - 1))] = 1
    kern = kern.unsqueeze(0).unsqueeze(0).to(x.device)

    # Reshaping to: batch, 1 channel, working space.
    end = tuple(range(-ndim, 0))
    x = x.movedim(dim, end)
    y = x.reshape(-1, 1, *x.shape[-ndim:])

    f = getattr(torch.nn.functional, f'conv{ndim}d')
    for _ in range(n):
        y = f(y, kern, padding='same')

    # Restore shape and threshold.
    return y.view_as(x).movedim(end, dim).gt(0).to(dtype)


def erode(x, /, n=1, *, dim=None):
    """Perform binary morphological erosion in 1D, 2D, or 3D.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. We consider non-zero voxels foreground.
    n : int, optional
        Repeat the operation `n` times.
    dim : int or sequence of int, optional
        Operate along 1 to 3 dimensions. None means all.

    Returns
    -------
    torch.Tensor
        Eroded tensor.

    """
    # Dilate inverted input. For this duality to hold at the borders, we need
    # to pad with ones, so we pad with zeros before the inversion.
    dtype = x.dtype
    x = torch.as_tensor(x)
    x = torch.nn.functional.pad(x, pad=[1] * 2 * x.ndim).eq(0)
    x = dilate(x, n, dim=dim).logical_not().to(dtype)
    return x[(slice(1, -1),) * x.ndim]


def close(x, /, n=1, *, dim=None):
    """Perform binary morphological closing in 1D, 2D, or 3D.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. We consider non-zero voxels foreground.
    n : int, optional
        Dilate `n` times, then erode `n` times.
    dim : int or sequence of int, optional
        Operate along 1 to 3 dimensions. None means all.

    Returns
    -------
    torch.Tensor
        Closed tensor.

    """
    return erode(dilate(x, n, dim=dim), n, dim=dim)


def open(x, /, n=1, *, dim=None):
    """Perform binary morphological opening in 1D, 2D, or 3D.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. We consider non-zero voxels foreground.
    n : int, optional
        Erode `n` times, then dilate `n` times.
    dim : int or sequence of int, optional
        Operate along 1 to 3 dimensions. None means all.

    Returns
    -------
    torch.Tensor
        Opened tensor.

    """
    return dilate(erode(x, n, dim=dim), n, dim=dim)


def fill_holes(x, /, *, dim=None):
    """Perform binary morphological hole filling in 1D, 2D, or 3D.

    The algorithm floods the background from the image border. We consider
    holes that touch the border background and will not fill these.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. We consider non-zero voxels foreground.
    dim : int or sequence of int, optional
        Operate along 1 to 3 dimensions. None means all.

    Returns
    -------
    torch.Tensor
        Hole-filled tensor.

    """
    # Inverted input.
    dtype = x.dtype
    x = torch.as_tensor(x).gt(0)
    inv = x.logical_not()

    # Dimensions.
    if dim is None:
        dim = torch.arange(x.ndim)
    dim = torch.as_tensor(dim).ravel()

    # Seeds: background voxels touching the boundary.
    mask = torch.zeros_like(x)
    for i in dim:
        for end in (0, -1):
            ind = [slice(None)] * x.ndim
            ind[i] = end
            mask[*ind] = inv[*ind]

    # Flood background from seeds until no more change.
    prev = torch.zeros_like(mask)
    while not mask.equal(prev):
        prev = mask
        mask = dilate(mask, dim=dim).logical_and(inv)

    return mask.logical_not().to(dtype)


def label(x, /):
    """Label K N-dimensional binary connected foreground components.

    The algorithm uses (minimum) label propagation. We initialize foreground
    pixels to unique index labels and iteratively replace each pixel with the
    lowest label within its neighborhood, until convergence.

    Parameters
    ----------
    x : torch.Tensor
        N-dimensional input tensor. We consider non-zero voxels foreground.

    Returns
    -------
    torch.Tensor
        Labeled components, valued 1 through K. Same shape as `x`.
    (K,) torch.Tensor
        Unique labels 1 through K for lartest to smallest.
    (K,) torch.Tensor
        Corresponding label sizes.

    """
    # Padding prevents components from wrapping around the border.
    x = torch.as_tensor(x)
    x = torch.nn.functional.pad(x, pad=[1] * x.ndim * 2)

    # Initial index labels. Floating point allows infinite background.
    labels = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    labels = labels.view_as(x)
    bg = x.eq(0)
    labels[bg] = torch.inf

    # Kernel: 4-neighborhood in 2D.
    steps = torch.eye(x.ndim, dtype=torch.int32)
    steps = torch.cat((-steps, steps))
    steps = (*(s.tolist() for s in steps), [0] * x.ndim)

    # Minimum label propagation. Restore background after each pass.
    prev = torch.zeros_like(labels)
    dims = tuple(range(x.ndim))
    while not labels.equal(prev):
        prev = labels
        neigh = [torch.roll(labels, shifts=s, dims=dims) for s in steps]
        labels, _ = torch.stack(neigh).min(dim=0)
        labels[bg] = torch.inf

    # Remove padding, order by size.
    labels = labels[(slice(1, -1),) * labels.ndim]
    uniq, size = labels.unique(return_counts=True)
    if uniq.numel() and uniq[-1] == torch.inf:
        uniq, size = uniq[:-1], size[:-1]

    ind = size.argsort(descending=True)
    uniq = uniq[ind]
    size = size[ind]

    # Map to indices. Initialize last label to 0 in case of background only.
    new = 0
    out = torch.zeros_like(labels, dtype=torch.int64)
    for new, old in enumerate(uniq, start=1):
        out[labels == old] = new

    return out, torch.arange(1, 1 + new), size
