"""Indexing and slicing."""


def sub2ind(size, *sub):
    """Convert N-dimensional subscripts to linear 1D indices.

    Indices into the right-most dimension change most rapidly.

    Parameters
    ----------
    size : sequence of int
        N-element tensor shape.
    *sub : sequence of int or sequence of torch.Tensor
        Subscripts for each dimension.

    Returns
    -------
    int or torch.Tensor
        Linear index.

    """
    *size, mult = size
    *sub, ind = sub
    for i, s in zip(sub[::-1], size[::-1]):
        ind += i * mult
        mult *= s

    return ind


def ind2sub(size, ind):
    """Convert linear 1D indices to N-dimensional subscripts.

    Indices into the right-most dimension change most rapidly.

    Parameters
    ----------
    size : sequence of int
        N-element tensor shape.
    ind : int or torch.Tensor
        Linear index.

    Returns
    -------
    tuple of int or tuple of torch.Tensor
        Subscripts for each dimension.

    """
    sub = []
    for s in size[::-1]:
        sub.append(ind % s)
        ind = ind // s

    return tuple(sub[::-1])
