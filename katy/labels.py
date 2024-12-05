"""Label manipulation and image synthesis."""


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


def rebase(labels, unknown=0, device=None):
    """Build a lookup table (LUT) from labels to contiguous indices.

    Instead of directly rebasing a `label_map`, the function returns a `lut`
    tensor for efficient rebasing via indexing, that is, via:

        `lut[label_map.ravel()].view_as(label_map)`

    Parameters
    ----------
    labels : sequence of int or dict or torch.Tensor
        All possible label values or a mapping from them to new output labels,
        enabling you to merge labels before converting to indices.
    unknown : int, optional
        Output value for missing input labels.
    device : torch.device, optional
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        Lookup table.

    """
    if not isinstance(unknown, int):
        raise ValueError(f'value {unknown} is not a Python integer')

    if not isinstance(labels, dict):
        labels = {x: x for x in labels}

    # Prepare output.
    max_label = int(max(labels))
    lut = torch.full(size=(max_label + 1,), fill_value=unknown)

    # Conversion to indices.
    out_labels = sorted(labels.values())
    out_to_ind = {label: i for i, label in enumerate(out_labels)}

    # Python scalars to prevent torch.uint8 interpretation as boolean indices.
    for inp, out in labels.items():
        lut[int(inp)] = out_to_ind[out]

    return lut.to(device=device)
