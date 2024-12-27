"""Utility module."""


import functools
import torch


def chance(prob, size=1, device=None, generator=None):
    """Return True with given probability.

    Parameters
    ----------
    prob : float
        Probability of returning True, in the range [0, 1].
    size : torch.Size
        Output shape.
    device : torch.device, optional
        Device of the returned tensor.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    torch.Tensor
        True with probability `prob`.

    """
    if prob < 0 or prob > 1:
        raise ValueError(f'probability {prob} is not in range [0, 1]')

    size = torch.as_tensor(size).ravel()
    return torch.rand(*size, device=device, generator=generator) < prob


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


def batched(f):
    """Add batch support to a function that processes multi-channel tensors.

    Parameters
    ----------
    f : callable
        Callable taking a `torch.Tensor` of shape `(C, *size)` as its first
        argument and returning individually stackable outputs.

    Returns
    -------
    function
        Function wrapping `f`, with the added keyword argument `batched=True`.
        If `batched` is False, it behaves like `f`. If `batched` is True, it
        will take inputs `(B, C, *size)` inputs and return `(B, ...)` outputs.

    """
    @functools.wraps(f)
    def wrapper(x, *args, batched=True, **kwargs):
        # Unchanged behavior.
        if not batched:
            return f(x, *args, **kwargs)

        # Batch processing.
        out = [f(batch, *args, **kwargs) for batch in x]
        if isinstance(out[0], torch.Tensor):
            return torch.stack(out)

        # Function returned several outputs.
        return tuple(torch.stack(o) for o in zip(*out))

    return wrapper


def channels(shared):
    """Make a function treat the channels of a tensor independently.

    Parameters
    ----------
    f : callable
        Callable taking a `torch.Tensor` of shape `(C, *size)` as its first
        argument and returning separately concatenable outputs.
    shared : bool
        Default value of the `shared` keyword argument of the output function.

    Returns
    -------
    function
        Function wrapping `f`, with the added keyword argument `shared`. If
        `shared` is True, the function behaves like `f`. If `shared` is False,
        it will process each channel separately.

    """
    def wrapper(f):

        @functools.wraps(f)
        def process(x, *args, shared=shared, **kwargs):
            # Default shared-channel behavior.
            if shared:
                return f(x, *args, **kwargs)

            # Separate processing.
            out = [f(c, *args, **kwargs) for c in x.split(1)]
            if isinstance(out[0], torch.Tensor):
                return torch.cat(out)

            # Function returned several outputs.
            return tuple(torch.cat(o) for o in zip(*out))

        return process

    return wrapper


def randomized(f):
    """Randomize the application of a function that modifies a tensor.

    Parameters
    ----------
    f : callable
        Callable with one or more positional arguments.

    Returns
    -------
    function
        Function wrapping `f`, with the added keyword argument `prob=1`,
        controlling the chance of `f` being applied. When `f` is not applied,
        the function will return the the first argument, unmodified.

    """
    @functools.wraps(f)
    def wrapper(x, *args, prob=1, **kwargs):
        generator = kwargs.get('generator')
        device = None if generator is None else generator.device

        if chance(prob, device=device, generator=generator):
            return f(x, *args, **kwargs)

        return x

    return wrapper
