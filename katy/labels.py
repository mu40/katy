"""Label manipulation and image synthesis."""

import katy as kt
import os
import torch


def to_image(x, channels=1, *, generator=None):
    """Synthesize gray-scale images from a discrete-valued label map.

    Parameters
    ----------
    x : (B, 1, *size) torch.Tensor
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
    x = torch.as_tensor(x, dtype=torch.int64)
    ndim = x.ndim - 2
    if ndim < 1 or x.size(1) != 1:
        raise ValueError(f'label map size {x.shape} is not (B, 1, *size)')

    dev = dict(device=x.device)
    channels = torch.as_tensor(channels)
    if channels.numel() > 1 or channels.lt(1):
        raise ValueError(f'channel count {channels} is not a positive scalar')

    # Lookup table.
    batch = x.size(0)
    labels = x.max() + 1
    lut = torch.rand(batch, channels, labels, generator=generator, **dev)

    # Indices into LUT. Keep channels on CPU until here.
    channels = channels.to(**dev)
    off_c = torch.arange(channels, **dev) * labels
    off_c = off_c.view(1, channels, *[1] * ndim)
    off_b = torch.arange(batch, **dev) * channels * labels
    off_b = off_b.view(-1, 1, *[1] * ndim)
    ind = x + off_b + off_c

    return lut.view(-1)[ind]


def to_rgb(x, colors, labels=None, dim=1):
    """Convert label maps to RGB color tensors.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Discrete or one-hot label map.
    colors : os.PathLike or dict
        FreeSurfer color lookup table.
    labels : os.PathLike or sequence of int
        Labels corresponding to the C one-hot channels. Required if `C > 1.`
    dim : int, optional
        Output channel dimension.

    Returns
    -------
    (B, ...) torch.Tensor
        RGB values in [0, 1], with 3 channels along dimension `dim`.

    """
    # Conversion to discrete labels.
    x = torch.as_tensor(x)
    if x.size(1) > 1 and labels is None:
        raise ValueError(f'need labels for one-hot map {x.shape}')
    if x.size(1) > 1:
        x = collapse(x, labels)

    # Lookup table.
    if not isinstance(colors, dict):
        colors = kt.io.read_colors(colors)
    colors = {k: v['color'] for k, v in colors.items()}

    # Lookup table.
    size = max(colors) + 1
    lut = torch.zeros(size, 3, device=x.device)
    for i, color in colors.items():
        lut[i] = torch.as_tensor(color)

    # Do not cast to integers before `argmax`.
    return lut[x.to(torch.int64)].squeeze(1).movedim(-1, dim) / 255


def remap(x, mapping=None, unknown=None):
    """Remap the values of a discrete label map.

    Parameters
    ----------
    x : torch.Tensor
        Label map of non-negative values.
    mapping : dict or os.PathLike, optional
        Labels missing from keys will become `unknown`. None returns the input.
    unknown : int, optional
        Value for labels missing from `mapping` keys. None means identity.

    Returns
    -------
    torch.Tensor
        Relabeled label map.

    """
    x = torch.as_tensor(x)
    if mapping is None:
        return x

    # Type.
    dtype = x.dtype
    x = x.to(torch.int64)

    # Mapping from old to new labels. Make all keys Python integers, because
    # JSON stores keys as strings, PyTorch tensors are not hashable, and
    # torch.uint8 scalars are interpreted as boolean indices.
    if isinstance(mapping, (str, os.PathLike)):
        mapping = kt.io.load(mapping)
    mapping = {int(k): v for k, v in mapping.items()}

    # Lookup table.
    size = max(max(mapping), x.max()) + 1
    if unknown is None:
        lut = torch.arange(size)
    else:
        lut = torch.full([size], fill_value=unknown)

    lut[list(mapping)] = torch.tensor(list(mapping.values()))
    return lut.to(x.device)[x].to(dtype)


def one_hot(x, labels):
    """One-hot encode a discrete label map.

    Parameters
    ----------
    x : (B, 1, *size) torch.Tensor
        Label map of non-negative values.
    labels : os.PathLike or sequence of int
        Unique integer labels to one-hot encode in order. Unspecified labels
        will be mapped to the first output channel.

    Returns
    -------
    (B, C, *size) torch.Tensor
        One-hot probability map, where `C` is `len(labels)`.

    """
    x = torch.as_tensor(x, dtype=torch.int64)
    if x.ndim < 2 or x.size(1) != 1:
        raise ValueError(f'label map {x.shape} is not (B, 1, ...)')

    if isinstance(labels, (str, os.PathLike)):
        labels = kt.io.load(labels)
    if not isinstance(labels, torch.Tensor):
        labels = list(map(int, labels))

    labels = torch.as_tensor(labels).ravel()
    if len(labels.unique(sorted=False)) != len(labels):
        raise ValueError(f'label values {labels} are not unique')

    # Lookup table.
    size = max(x.max(), labels.max()) + 1
    lut = torch.zeros(size, dtype=torch.int64)
    lut[labels] = torch.arange(len(labels))

    # Replace the singleton channel dimension.
    x = lut.to(x.device)[x]
    x = torch.nn.functional.one_hot(x, num_classes=len(labels))
    return x.squeeze(1).movedim(-1, 1).to(torch.get_default_dtype())


def collapse(x, labels):
    """Convert a one-hot map to discrete labels.

    Parameters
    ----------
    x : (B, C, *size) torch.Tensor
        One-hot probability map.
    labels : os.PathLike or sequence of int
        Label values that correspond to the C one-hot channels.

    Returns
    -------
    (B, 1, *size) torch.Tensor
        Discrete label map.

    """
    x = torch.as_tensor(x)
    if x.ndim < 2 or x.size(1) == 1:
        raise ValueError(f'one-hot map {x.shape} is not (B, C, ...), C > 1')

    if isinstance(labels, (str, os.PathLike)):
        labels = kt.io.load(labels)
    if not isinstance(labels, torch.Tensor):
        labels = list(map(int, labels))

    labels = torch.as_tensor(labels, device=x.device).ravel()
    if len(labels) != x.size(1):
        raise ValueError(f'{x.size(1)} channels != {len(labels)} labels')

    return labels[x.argmax(dim=1, keepdim=True)]
