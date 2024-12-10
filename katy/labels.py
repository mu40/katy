"""Label manipulation and image synthesis."""


import katy as kt
import os
import torch


def to_image(label_map, channels=1, generator=None):
    """Synthesize gray-scale images from a discrete-valued label map.

    Parameters
    ----------
    label_map : (B, 1, *size) torch.Tensor
        Discrete, positive-valued label map.
    channels : int, optional
        Number of output channels.
    generator : torch.Generator, optional
        Pseudo-random number generator.

    Returns
    -------
    (B, channels, *size) torch.Tensor
        Synthetic image.

    """
    label_map = torch.as_tensor(label_map, dtype=torch.int64)
    if label_map.ndim < 3 or label_map.size(1) != 1:
        raise ValueError(f'label size {label_map.shape} is not (B, 1, *size)')

    dev = dict(device=label_map.device)
    channels = torch.as_tensor(channels, **dev)
    if channels.numel() > 1 or channels.lt(1):
        raise ValueError(f'channel count {channels} is not a positive scalar')

    # Dimensions.
    dim = label_map.ndim - 2
    labels = label_map.max() + 1
    batch = label_map.size(0)

    # Lookup table.
    lut = torch.rand(
        size=(batch, channels, labels),
        dtype=torch.get_default_dtype(),
        generator=generator,
        **dev,
    )

    # Indices into LUT.
    off_c = torch.arange(channels, **dev) * labels
    off_c = off_c.view(1, channels, *[1] * dim)
    off_b = torch.arange(batch, **dev) * channels * labels
    off_b = off_b.view(-1, 1, *[1] * dim)
    ind = label_map + off_b + off_c

    return lut.view(-1)[ind]


def one_hot(x, labels):
    """Convert label map to one-hot encoding, ignoring negative values.

    Parameters
    ----------
    x : (B, 1, *size) torch.Tensor
        Label map of `torch.int64` values in [0, labels).
    labels : int
        Number of labels or output channels.

    Returns
    -------
    (B, labels, *size) torch.Tensor
        One-hot encoding.

    """
    if x.ndim < 2 or x.size(1) != 1:
        raise ValueError(f'label map size {x.shape} is not (B, 1, ...)')
    if x.max() >= labels:
        raise ValueError(f'highest label {x.max()} is not less than {labels}')

    # Convert negative values to an additional class and remove it later.
    x = torch.where(x < 0, labels, x)
    x = torch.nn.functional.one_hot(x, num_classes=labels + 1)
    x = x[..., :labels]

    # Replace the singleton channel dimension.
    return x.squeeze(1).movedim(-1, 1)


def rebase(x, labels, mapping=None, unknown=-1):
    """Convert numeric label maps to contiguous indices.

    The labels represented by the output indices will correspond to the input
    labels, sorted by ascending numerical value.

    Parameters
    ----------
    x : torch.Tensor
        Discrete-valued label map.
    labels : os.PathLike or torch.Tensor or sequence of int
        All possible input label values.
    mapping : dict or os.PathLike, optional
        Key-value mapping. Labels missing from the keys will be removed,
        labels sharing the same value will be merged prior to conversion.
    unknown : int, optional
        Output value for labels missing from the `mapping` keys. Set this to
        a negative value to remove labels when calling `one_hot`.

    Returns
    -------
    tuple
        Rebased input tensor, mapping from indices to label values.

    """
    x = torch.as_tensor(x, dtype=torch.int64)
    if not isinstance(unknown, int):
        raise ValueError(f'value {unknown} is not a Python integer')

    # Possible input labels.
    if isinstance(labels, (str, os.PathLike)):
        labels = kt.io.load(labels)
    labels = sorted(map(int, labels))

    # Mapping from old to new labels. Make all keys Python integers, because
    # JSON stores keys as strings, PyTorch tensors are not hashable, and
    # torch.uint8 scalars are interpreted as boolean indices.
    if mapping is None:
        mapping = {x: x for x in labels}
    if isinstance(mapping, (str,  os.PathLike)):
        mapping = {int(k): v for k, v in kt.io.load(mapping).items()}

    # Conversion between new labels, indices. Order by old label value.
    new_labels = tuple(mapping[k] for k in mapping if k in labels)
    new_to_ind = {new: i for i, new in enumerate(new_labels)}
    ind_to_new = {i: new for new, i in new_to_ind.items()}

    # Lookup table.
    highest = max(labels)
    lut = torch.zeros(highest + 1, dtype=torch.int64, device=x.device)
    for old in labels:
        new = mapping.get(old)
        lut[old] = new_to_ind.get(new, unknown)

    return lut[x], ind_to_new
