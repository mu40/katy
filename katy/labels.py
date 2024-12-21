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
    channels = torch.as_tensor(channels)
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

    # Indices into LUT. Keep channels on CPU until here.
    channels = channels.to(**dev)
    off_c = torch.arange(channels, **dev) * labels
    off_c = off_c.view(1, channels, *[1] * dim)
    off_b = torch.arange(batch, **dev) * channels * labels
    off_b = off_b.view(-1, 1, *[1] * dim)
    ind = label_map + off_b + off_c

    return lut.view(-1)[ind]


def to_rgb(x, colors, mapping=None):
    """Convert label maps to RGB color tensors.

    Parameters
    ----------
    x : (B, C, ...) torch.Tensor
        Discrete-valued, index, or one-hot label map. The function assumes
        non-index label values if `C` is 1 and `mapping` is None.
    colors : os.PathLike or dict
        FreeSurfer color lookup table.
    mapping : dict, optional
        Mapping from indices to labels. Labels can be names of type `str` or
        values of type `int` in the color table. Required if `C > 1`.

    Returns
    -------
    (B, 3, ...) torch.Tensor
        RGB values in [0, 1].

    """
    # Convert one-hot map to indices.
    x = torch.as_tensor(x)
    if x.size(1) > 1 and mapping is None:
        raise ValueError(f'need mapping for one-hot map {x.shape}')
    if x.size(1) > 1:
        x = x.argmax(dim=1, keepdim=True)

    # Lookup table.
    if not isinstance(colors, dict):
        colors = kt.io.read_color_table(colors)

    # Actual labels.
    if mapping is None:
        to_color = {k: v['color'] for k, v in colors.items()}

    # Indices.
    else:
        # Mapping from label name or value to color.
        ntc = {v['name']: v['color'] for v in colors.values()}
        vtc = {k: v['color'] for k, v in colors.items()}
        value = next(iter(mapping.values()))
        trans = ntc if isinstance(value, str) else vtc

        # Mapping from index to color.
        to_color = {int(k): trans[v] for k, v in mapping.items()}

    # Lookup table.
    highest = max(to_color)
    lut = torch.zeros(highest + 1, 3, device=x.device)
    for i, color in to_color.items():
        lut[i] = torch.as_tensor(color)

    # Do not cast to integers before `argmax`.
    return lut[x.to(torch.int64)].squeeze(1).movedim(-1, 1) / 255


def label_to_index(labels, mapping=None, unknown=0, return_inverse=False):
    """Construct a mapping from label values to contiguous indices.

    Indices will reflect original labels in ascending order.

    Parameters
    ----------
    labels : os.PathLike or torch.Tensor or sequence of int
        All possible label values.
    mapping : dict or os.PathLike, optional
        Label translation applied before to indexing. Keys with the same value
        will be merged, labels missing from the keys set to `unknown`.
    unknown : int, optional
        Output index for labels missing from `mapping` keys. Set this to
        a negative value to remove labels when one-hotting.
    return_inverse : bool, optional
        Return an inverse. Will map to remapped labels if `mapping` provided.

    Returns
    -------
    dict
        Mapping from original labels to indices.
    dict, optional
        Return an inverse from indices to original or remapped labels, if
        `mapping` provided. Only if `return_inverse` is True.

    """
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
    if not isinstance(mapping, dict):
        mapping = kt.io.load(mapping)
    mapping = {int(k): v for k, v in mapping.items()}

    # Conversion between new labels, indices. Order by old label value.
    new_labels = tuple(mapping[k] for k in mapping if k in labels)
    new_to_ind = {new: i for i, new in enumerate(new_labels)}
    ind_to_new = {i: new for i, new in enumerate(new_labels)}

    old_to_ind = {}
    for old in labels:
        new = mapping.get(old)
        old_to_ind[old] = new_to_ind.get(new, unknown)

    return (old_to_ind, ind_to_new) if return_inverse else old_to_ind


def remap(x, mapping):
    """Remap the values of a discrete-valued label map.

    Parameters
    ----------
    x : torch.Tensor
        Label map.
    mapping : dict or os.PathLike
        Mapping from old to new labels, all convertible to `int`. Must include
        all possible label map values as keys.

    Returns
    -------
    torch.Tensor
        Remapped label map.

    """
    x = torch.as_tensor(x, dtype=torch.int64)
    if not isinstance(mapping, dict):
        mapping = kt.io.load(mapping)

    # Make all keys Python integers, because JSON stores keys as strings.
    mapping = {int(k): int(v) for k, v in mapping.items()}

    # Lookup table.
    highest = max(mapping)
    lut = torch.zeros(highest + 1, dtype=torch.int64, device=x.device)
    lut[list(mapping)] = torch.tensor(list(mapping.values()))
    return lut[x]


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
