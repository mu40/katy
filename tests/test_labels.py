"""Tests for label-manipulation and image-synthesis module."""


import torch
import pytest
import katy as kt


def test_to_image_unchanged():
    """Test if image synthesis leaves input unchanged."""
    inp = torch.ones(1, 1, 4, 4)

    orig = inp.clone()
    kt.labels.to_image(inp)
    assert inp.eq(orig).all()


def test_to_image_illegal_inputs():
    """Test image synthesis with illegal inputs."""
    # Input should only have one channel.
    x = torch.ones(1, 2, 4, 4)
    with pytest.raises(ValueError):
        kt.labels.to_image(x)

    # Channels should be a positive scalar.
    x = torch.ones(1, 1, 4, 4)
    with pytest.raises(ValueError):
        kt.labels.to_image(x, channels=0)


def test_to_image_dtype():
    """Test image synthesis output data type."""
    x = torch.ones(1, 1, 4)
    assert kt.labels.to_image(x).dtype == torch.get_default_dtype()


def test_to_image_shape():
    """Test image synthesis output shape."""
    batch = 3
    space = (4, 5, 6)

    for dim in (2, 3):
        for channels in (1, 2):
            inp = torch.ones(batch, 1, *space[:dim])
            out = kt.labels.to_image(inp, channels)
            assert out.shape == (batch, channels, *space[:dim])


def test_to_image_variability():
    """Test if synthesis differs across batch and channels."""
    inp = torch.randint(100, size=(2, 1, 10, 10))
    out = kt.labels.to_image(inp, channels=3)

    # In each batch, each channels should differ.
    for batch in out:
        assert batch[0].ne(batch[1]).any()

    # Batches should differ.
    assert out[0].ne(out[1]).any()


def test_to_rgb_properties():
    """Test trivial shape and data type of RGB conversion."""
    colors = {0: {'name': 'label', 'color': (0, 0, 0)}}
    mapping = {0: 'label'}

    # Discrete-valued labels in 1D.
    inp = torch.zeros(1, 1, 2)
    out = kt.labels.to_rgb(inp, colors)
    assert out.shape == (inp.shape[0], 3, *inp.shape[2:])
    assert out.dtype == torch.get_default_dtype()

    # Index labels in 2D.
    inp = torch.zeros(2, 1, 4, 4)
    out = kt.labels.to_rgb(inp, colors, mapping, dim=-1)
    assert out.shape == (inp.shape[0], *inp.shape[2:], 3)
    assert out.dtype == torch.get_default_dtype()

    # One-hot labels in 3D.
    inp = torch.zeros(1, 4, 2, 2, 2)
    out = kt.labels.to_rgb(inp, colors, mapping)
    assert out.shape == (inp.shape[0], 3, *inp.shape[2:])
    assert out.dtype == torch.get_default_dtype()


def test_to_rgb_illegal_arguments():
    """Test if RGB conversion of one-hot map requires a mapping."""
    colors = {0: {'name': 'label', 'color': (0, 0, 0)}}
    x = torch.zeros(1, 2, 3, 3)

    with pytest.raises(ValueError):
        kt.labels.to_rgb(x, colors, mapping=None)


def test_to_rgb_labels(tmp_path):
    """Test conversion of labels to RGB, with colors from file."""
    # Colors.
    lut = (
        '# ID  Label  R   G   B   A\n'
        '# ------------------------\n'
        '  1   Red    255   0   0 0\n'
        '  2   Green  0   255   0 0\n'
        '  3   Blue   0     0 255 0\n'
    )

    # Save color map.
    path = tmp_path / 'lut.txt'
    with open(path, mode='w') as f:
        f.write(lut)

    # Label map: batch, channels, space.
    x = torch.ones(1, 1, 2, 2)
    x[..., 1] = 2
    rgb = kt.labels.to_rgb(x, colors=path)

    # Expect red in first half of last dimension only.
    assert rgb[:, 0, :, 0].eq(1).all()
    assert rgb[:, 0, :, 1].eq(0).all()

    # Expect green in second half of last dimension only.
    assert rgb[:, 1, :, 0].eq(0).all()
    assert rgb[:, 1, :, 1].eq(1).all()

    # Expect no blue.
    assert rgb[:, 2, ...].eq(0).all()


def test_to_rgb_indices():
    """Test converting rebased labels to RGB, with index-to-name mapping."""
    # Colors.
    r = {'name': 'red', 'color': (255, 0, 0)}
    g = {'name': 'green', 'color': (0, 255, 0)}
    colors = {9: r, 7: g}

    # Mapping of indices to label names.
    mapping = {0: 'red', 1: 'green'}

    # Index iabel map: batch, channels, space.
    x = torch.zeros(1, 1, 2, 2)
    x[..., 1] = 1
    rgb = kt.labels.to_rgb(x, colors, mapping)

    # Expect red in first half of last dimension only.
    assert rgb[:, 0, :, 0].eq(1).all()
    assert rgb[:, 0, :, 1].eq(0).all()

    # Expect green in second half of last dimension only.
    assert rgb[:, 1, :, 0].eq(0).all()
    assert rgb[:, 1, :, 1].eq(1).all()

    # Expect no blue.
    assert rgb[:, 2, ...].eq(0).all()


def test_to_rgb_one_hot():
    """Test converting one-hot labels to RGB, with index-to-value mapping."""
    # Colors.
    r = {'name': 'red', 'color': (255, 0, 0)}
    b = {'name': 'blue', 'color': (0, 0, 255)}
    colors = {9: r, 7: b}

    # Mapping of indices to label values.
    mapping = {0: 9, 1: 7}

    # One-hot label map: batch, channels, space.
    x = torch.zeros(3, 2, 4)
    x[:, 0, :2] = 1
    x[:, 1, :] = 1 - x[:, 0, :]
    rgb = kt.labels.to_rgb(x, colors, mapping)

    # Expect red in first half of space only.
    assert rgb[:, 0, :2].eq(1).all()
    assert rgb[:, 0, 2:].eq(0).all()

    # Expect no green.
    assert rgb[:, 1, :].eq(0).all()

    # Expect green in second half of space only
    assert rgb[:, 2, :2].eq(0).all()
    assert rgb[:, 2, 2:].eq(1).all()


def test_map_index():
    """Test creating a label-index mapping."""
    labels = (6, 5, 4, 1)

    lab_to_ind = kt.labels.map_index(labels)
    assert lab_to_ind == {1: 0, 4: 1, 5: 2, 6: 3}

    ind_to_lab = kt.labels.map_index(labels, invert=True)
    assert ind_to_lab == {0: 1, 1: 4, 2: 5, 3: 6}


def test_map_index_mapping():
    """Test creating a label-index lookup with prior re-mapping."""
    # Input labels, all possible labels, mapping.
    labels = (5, 4, 1, 0)
    mapping = {4: 2, 5: 1, 6: 0}

    # Expect old->new->index, unknown values X defaulting to index 0:
    # 0>X>0, 1>X>0, 4>2>1, 5>1>0.
    old_to_ind = kt.labels.map_index(labels, mapping=mapping)
    assert old_to_ind == {0: 0, 1: 0, 4: 1, 5: 0}

    ind_to_new = kt.labels.map_index(labels, mapping, invert=True)
    assert ind_to_new == {0: 1, 1: 2}

    # Set an explicit index for unknown values: 0>X>-2, 1>X>-2, 4>2>1, 5>1>0.
    old_to_ind = kt.labels.map_index(labels, mapping, unknown=-2)
    assert old_to_ind == {0: -2, 1: -2, 4: 1, 5: 0}

    ind_to_new = kt.labels.map_index(labels, mapping, unknown=-2, invert=True)
    assert ind_to_new == {0: 1, 1: 2}


def test_map_index_disk(tmp_path):
    """Test creating a label-index mapping from files."""
    # Tensors are not JSON serializable.
    labels = list(range(3))
    mapping = {i: i for i in labels}

    for ext in ('json', 'pickle', 'pt'):
        f_labels = tmp_path / f'labels.{ext}'
        f_mapping = tmp_path / f'mapping.{ext}'
        kt.io.save(labels, f_labels)
        kt.io.save(mapping, f_mapping)

        assert kt.labels.map_index(f_labels) == mapping
        assert kt.labels.map_index(f_labels, invert=True) == mapping
        assert kt.labels.map_index(f_labels, f_mapping) == mapping
        assert kt.labels.map_index(f_labels, f_mapping, invert=True) == mapping


def test_map_index_strings():
    """Test label-to-index lookup using string-type label values."""
    # JSON stores dictionary keys as strings. Cannot serialize tensors.
    labels = list(range(3))
    mapping = {i: i for i in labels}

    # Expect mapping keys to be cast to int.
    strings = {str(k): v for k, v in mapping.items()}
    assert kt.labels.map_index(labels, strings) == mapping
    assert kt.labels.map_index(labels, strings, invert=True) == mapping

    # Expect labels to be cast to int.
    strings = [str(i) for i in labels]
    assert kt.labels.map_index(strings, mapping) == mapping
    assert kt.labels.map_index(strings, mapping, invert=True) == mapping


def test_map_index_illegal_arguments():
    """Test rebasing labels with illegal input arguments."""
    # Input label map should have one channel.
    x = torch.arange(3)

    # The unknown value must be an `int`.
    with pytest.raises(ValueError):
        kt.labels.map_index(labels=x, unknown=0.1)

    with pytest.raises(ValueError):
        kt.labels.map_index(labels=x, unknown=torch.tensor(1))


def test_rebase_types():
    """Test remapping input and output types."""
    # Input does not have to include all the possible labels.
    x = (5, 6, 6, 7)
    mapping = {5: 1, 6: 1, 7: 2}

    for inp in (tuple(x), list(x), torch.tensor(x)):
        out = kt.labels.rebase(inp, labels=set(x), mapping=mapping)
        assert out.dtype == torch.int64
        assert out.tolist() == [0, 0, 0, 1]


def test_rebase_disk(tmp_path):
    """Test remapping labels with mapping in file."""
    # Tensors are not JSON serializable.
    x = (5, 6, 6, 7)
    mapping = {f: f for f in x}

    for ext in ('json', 'pickle', 'pt'):
        f_mapping = tmp_path / f'mapping.{ext}'
        kt.io.save(mapping, f_mapping)

        out = kt.labels.rebase(x, f_mapping)
        assert out.tolist() == [0, 1, 1, 2]


def test_one_hot_unchanged():
    """Test if one-hotting leaves input unchanged."""
    inp = torch.zeros(1, 1, 4, dtype=torch.int64)

    orig = inp.clone()
    kt.labels.one_hot(inp, depth=1)
    assert inp.eq(orig).all()


def test_one_hot_illegal_inputs():
    """Test if one-hotting with illegal input values."""
    # Input label map should have one channel.
    x = torch.ones(1, 2, 3, dtype=torch.int64)
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, depth=2)

    # Highest label should be one less than the number of labels.
    x = torch.ones(1, 1, 3, dtype=torch.int64)
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, depth=1)


def test_one_hot_values():
    """Test explicit result of one-hot encoding."""
    depth = 3
    inp = torch.tensor((*range(depth), -1))
    inp = inp.view(1, 1, inp.numel())
    out = kt.labels.one_hot(inp, depth)

    # The number of output channels should equal the non-negative labels.
    assert out.shape == (1, depth, inp.numel())

    # Index value i should lead to activation of channel i only.
    assert out[0, :, :-1].eq(torch.eye(depth)).all()

    # Negative values should not have any activation.
    assert out[0, :, -1].eq(0).all()
