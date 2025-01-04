"""Utility module."""


import functools
import torch
import katy as kt


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
    if size_old.eq(size_new).all():
        return x

    # Indexing. Non-zero starting indices into the input are equivalent to
    # cropping, and those into the output are equivalent to padding.
    overlap = torch.minimum(size_old, size_new)
    a_old = size_old.sub(size_new).clamp(min=0) // 2
    a_new = size_new.sub(size_old).clamp(min=0) // 2
    ind_old = tuple(map(slice, a_old, a_old + overlap))
    ind_new = tuple(map(slice, a_new, a_new + overlap))

    out = torch.full(size_new.tolist(), fill, device=x.device)
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
