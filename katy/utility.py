"""Utility module."""

import functools
import katy as kt
import torch


def batch(*, batch):
    """Add batch support to a function that processes multi-channel tensors.

    Parameters
    ----------
    f : callable
        Callable taking a `torch.Tensor` of shape `(C, *size)` as its first
        argument and returning separately stackable outputs.
    batch : bool
        Default value of the `batch` argument of the output function.

    Returns
    -------
    function
        Function wrapping `f` with an added keyword argument `batch`. If False,
        the function behaves like `f`. If True, the function expects a leading
        batch dimension, mapping `f` and stacking outputs along it.

    """
    def wrapper(f):

        @functools.wraps(f)
        def batch_func(x, *args, batch=batch, **kwargs):
            # Unchanged behavior.
            if not batch:
                return f(x, *args, **kwargs)

            # Batch processing.
            out = [f(batch, *args, **kwargs) for batch in x]
            if isinstance(out[0], torch.Tensor):
                return torch.stack(out)

            # Individual stacking of multiple outputs.
            return tuple(torch.stack(o) for o in zip(*out))

        return batch_func

    return wrapper


def resize(x, size, fill=0):
    """Symmetrically crop or pad an N-dimensional tensor to a new size.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Tensor of `N`-element spatial shape.
    size : int or sequence of int
        New spatial shape. Pass 1 or N values.
    fill : float, optional
        Fill value for padding.

    Returns
    -------
    (B, C, *size) torch.Tensor
        Resized tensor.

    """
    # Dimensions. Keep batch and channel dimensions.
    ndim = x.ndim - 2
    size_old = torch.as_tensor(x.shape)
    size_new = torch.as_tensor(size).ravel().expand(ndim)
    size_new = torch.cat((size_old[:2], size_new))
    if size_old.equal(size_new):
        return x

    # Indexing. Non-zero starting indices into the input are equivalent to
    # cropping, and those into the output are equivalent to padding.
    overlap = torch.minimum(size_old, size_new)
    a_old = size_old.sub(size_new).clamp(min=0) // 2
    a_new = size_new.sub(size_old).clamp(min=0) // 2
    ind_old = tuple(map(slice, a_old, a_old + overlap))
    ind_new = tuple(map(slice, a_new, a_new + overlap))

    out = torch.full(size_new.tolist(), fill, dtype=x.dtype, device=x.device)
    out[ind_new] = x[ind_old]
    return out


def barycenter(x, grid=None, eps=1e-6):
    """Compute the center of mass in N dimensions.

    Clamps negative values to zero for differentiability.

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        Input tensor of spatial `N`-element size.
    grid : (N, *size) torch.Tensor, optional
       Coordinate grid.
    eps : float, optional
        Epsilon, to avoid dividing by zero.

    Returns
    -------
    (B, C, N) torch.Tensor
        Barycenter.

    """
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    size = x.shape[2:]

    # Grid.
    if grid is None:
        grid = kt.transform.grid(size, device=x.device)

    # Added dimension indexing space.
    x = x.clamp(min=0).unsqueeze(2)

    # Sum over space.
    dim = tuple(range(3, len(size) + 3))
    return (grid * x).sum(dim) / x.sum(dim).clamp(min=eps)


def quantile(x, q, dim=None, keepdim=False):
    """Compute quantiles.

    This function produces the same numerical results as `torch.quantile` but
    differs in two ways. First, it does not limit the size of the input.
    Second, it will not add a leading quantile dimensions to the output when
    `q` is a 1D tensor. Instead, it will replace the reduction dimension.

    Remove once https://github.com/pytorch/pytorch/issues/64947 is fixed.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    q : torch.Tensor or float or sequence of float
        Quantiles in the range [0, 1].
    dim : int, optional
        Dimension to reduce. None flattens the input before computation.
    keepdim : bool, optional
        Retain a singleton dimension when `q` has only one value.

    Returns
    -------
    torch.Tensor
        Quantiles.

    """
    # Inputs.
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    q = torch.as_tensor(q, dtype=torch.get_default_dtype()).ravel()
    if q.lt(0).any() or q.gt(1).any():
        raise ValueError(f'quantile {q} is outside range [0, 1]')

    # Dimensions. Ensure output shapes for scalar `q` follow `torch.quantile`.
    if dim is None:
        x = x.view(-1, *[1] * (x.ndim - 1)) if keepdim else x.flatten()
        dim = 0

    size = [1] * x.ndim
    size[dim] = -1
    q = q.view(size)
    x, _ = x.sort(dim)

    # Indices.
    n = torch.tensor(x.size(dim))
    ind = q * (n - 1)
    ind = ind.to(x.device)
    ind_0 = ind.to(torch.int64)
    ind_1 = torch.minimum(ind_0 + 1, n - 1)

    # Quantiles.
    val_0 = x.index_select(dim, index=ind_0.flatten())
    val_1 = x.index_select(dim, index=ind_1.flatten())
    out = torch.lerp(val_0, val_1, weight=ind - ind_0)

    return out if keepdim else out.squeeze(dim)


def normalize_minmax(x, dim=None):
    """Min-max normalize into [0, 1], avoiding divisions by zero.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int or sequence of int, optional
        Dimensions to reduce. None means all, treating the input as a single
        image. To normalize batches or channels separately, exclude them.

    Returns
    -------
    torch.Tensor
        Normalized tensor.

    """
    x = torch.as_tensor(x)
    x = x - x.amin(dim, keepdim=True)

    # Avoid division by zero.
    amax = x.amax(dim, keepdim=True)
    return x / torch.where(amax > 0, amax, 1)


def normalize_quantile(x, low=0.01, high=0.99, dim=None):
    """Min-max normalize between quantiles.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    low : float, optional
        Lower quantile. We will clip values below this threshold.
    high : float, optional
        Upper quantile. We will clip values above this threshold.
    dim : int or sequence of int, optional
        Dimensions to reduce. None means all, treating the input as a single
        image. To normalize batches or channels separately, exclude them.

    Returns
    -------
    torch.Tensor
        Normalized tensor.

    """
    # Arguments.
    x = torch.as_tensor(x, dtype=torch.get_default_dtype())
    if dim is None:
        dim = range(x.ndim)
    dim = torch.as_tensor(dim).ravel().tolist()

    # Flatten normalization dimensions and move to left.
    ind = tuple(range(len(dim)))
    x = x.movedim(dim, ind)
    y = x.reshape(-1, *x.shape[len(dim):])

    # Clamp and restore shape.
    lim = quantile(y, q=torch.as_tensor((low, high)), dim=0)
    x = y.clamp(*lim).view_as(x).movedim(ind, dim)

    return normalize_minmax(x, dim=dim)
