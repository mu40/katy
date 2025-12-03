"""Unit tests for label-manipulation and image-synthesis module."""

import katy as kt
import pytest
import torch


def test_to_image_unchanged():
    """Test if image synthesis leaves input unchanged."""
    inp = torch.ones(1, 1, 4, 4)
    orig = inp.clone()
    kt.labels.to_image(inp)
    assert inp.equal(orig)


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


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('channels', [2, 3])
def test_to_image_shape(dim, channels):
    """Test image synthesis output shape."""
    batch = 3
    space = [4] * dim

    inp = torch.ones(batch, 1, *space)
    out = kt.labels.to_image(inp, channels)
    assert out.shape == (batch, channels, *space)


def test_to_image_variability():
    """Test if synthesis differs across batch and channels."""
    x = torch.arange(50).view(2, 1, 5, 5)
    x = kt.labels.to_image(x, channels=2)
    assert not x[0].equal(x[1])
    assert not x[:, 0].equal(x[:, 1])


def test_to_rgb_trivial():
    """Test trivial shape and data type of RGB conversion."""
    colors = {0: {'name': 'unknown', 'color': (0, 0, 0)}}
    labels = (0, 1, 2, 3)

    # Discrete labels in 1D.
    inp = torch.zeros(1, 1, 2)
    out = kt.labels.to_rgb(inp, colors)
    assert out.shape == (inp.shape[0], 3, *inp.shape[2:])
    assert out.dtype == torch.get_default_dtype()

    # One-hot labels in 3D.
    inp = torch.zeros(1, 4, 2, 2, 2)
    out = kt.labels.to_rgb(inp, colors, labels)
    assert out.shape == (inp.shape[0], 3, *inp.shape[2:])
    assert out.dtype == torch.get_default_dtype()


def test_to_rgb_illegal_arguments():
    """Test if RGB conversion of one-hot map requires a label list."""
    colors = {0: {'name': 'unknown', 'color': (0, 0, 0)}}
    x = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError):
        kt.labels.to_rgb(x, colors, labels=None)


def test_to_rgb_labels(tmp_path):
    """Test conversion of labels to RGB, with colors from file."""
    # Save color map.
    lut = (
        '# ID  Label  R   G   B   A\n'
        '# ------------------------\n'
        '  1   Red    255   0   0 0\n'
        '  2   Green  0   255   0 0\n'
        '  3   Blue   0     0 255 0\n'
    )
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


def test_to_rgb_one_hot():
    """Test converting one-hot map to RGB."""
    # Colors.
    r = {'name': 'red', 'color': (255, 0, 0)}
    b = {'name': 'blue', 'color': (0, 0, 255)}
    colors = {9: r, 7: b}

    # Label values corresponding to channels.
    labels = (9, 7)

    # One-hot label map: batch, channels, space.
    x = torch.zeros(3, 2, 4)
    x[:, 0, :2] = 1
    x[:, 1, :] = 1 - x[:, 0, :]
    rgb = kt.labels.to_rgb(x, colors, labels)

    # Expect red in first half of space only.
    assert rgb[:, 0, :2].eq(1).all()
    assert rgb[:, 0, 2:].eq(0).all()

    # Expect no green.
    assert rgb[:, 1, :].eq(0).all()

    # Expect green in second half of space only
    assert rgb[:, 2, :2].eq(0).all()
    assert rgb[:, 2, 2:].eq(1).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64])
def test_remap_dtype(dtype):
    """Test if remapping maintains the input data type."""
    x = torch.tensor((5, 6, 6, 7), dtype=dtype)
    assert kt.labels.remap(x, mapping={5: 5}).dtype == dtype


def test_remap_memory():
    """Test remapping label map with mapping in memory."""
    x = (5, 6, 6, 7)
    mapping = {7: 1, 6: 2, 4: 3}

    # Expect unspecified labels to remain unchanged.
    inp = tuple(x)
    out = kt.labels.remap(inp, mapping, unknown=None)
    assert out.tolist() == [5, 2, 2, 1]

    # Expect unspecified labels become specific value.
    inp = list(x)
    out = kt.labels.remap(inp, mapping=mapping, unknown=9)
    assert out.tolist() == [9, 2, 2, 1]


@pytest.mark.parametrize('ext', ['json', 'pickle', 'pt'])
def test_remap_disk(tmp_path, ext):
    """Test remapping labels with mapping in file."""
    # Tensors are not JSON serializable.
    x = (5, 6, 6, 7)
    mapping = {7: 1, 6: 2}
    unknown = -1

    f = tmp_path / f'mapping.{ext}'
    kt.io.save(mapping, f)

    out = kt.labels.remap(x, f, unknown)
    assert out.tolist() == [-1, 2, 2, 1]


def test_one_hot_unchanged():
    """Test if one-hotting leaves input unchanged."""
    inp = torch.zeros(1, 1, 4)
    orig = inp.clone()
    kt.labels.one_hot(inp, labels=(1, 2, 3))
    assert inp.equal(orig)


def test_one_hot_dtype():
    """Test if one-hotting outputs standard floating point type."""
    inp = torch.zeros(1, 1, 4, dtype=torch.int64)
    out = kt.labels.one_hot(inp, labels=[1, 2])
    assert out.dtype == torch.get_default_dtype()


def test_one_hot_illegal_inputs():
    """Test if one-hotting with illegal input values."""
    # Input label map should have one channel.
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, labels=[1, 4])

    # Label values should be unique.
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, labels=[1, 1])


def test_one_hot_values():
    """Test explicit result of one-hot encoding."""
    inp = torch.tensor([3, 2, 1]).view(1, 1, -1)
    x = [2, 1]

    # Expect `len(labels)` channels, unspecified values in first channel.
    for labels in (tuple(x), list(x), torch.tensor(x), {i: None for i in x}):
        out = kt.labels.one_hot(inp, labels)
        assert out.shape == (1, len(labels), inp.numel())
        assert out[0, 0, :].tolist() == [1, 1, 0]
        assert out[0, 1, :].tolist() == [0, 0, 1]


@pytest.mark.parametrize('ext', ['json', 'pickle', 'pt'])
def test_one_hot_disk(tmp_path, ext):
    """Test remapping labels with mapping in file."""
    inp = torch.tensor([0, 1, 1, 2]).view(1, 1, -1)
    x = (0, 1, 2)

    f = tmp_path / f'labels.{ext}'
    for labels in (tuple(x), list(x), {i: None for i in x}):
        kt.io.save(labels, f)
        out = kt.labels.one_hot(inp, labels=f)
        assert out[0, 0, :].tolist() == [1, 0, 0, 0]
        assert out[0, 1, :].tolist() == [0, 1, 1, 0]
        assert out[0, 2, :].tolist() == [0, 0, 0, 1]


def test_collapse_unchanged():
    """Test if collapsing one-hot maps leaves input unchanged."""
    inp = torch.ones(1, 3, 4)
    orig = inp.clone()
    kt.labels.collapse(inp, labels=(1, 2, 3))
    assert inp.equal(orig)


def test_collapse_illegal_inputs():
    """Test collapsing one-hot map with illegal input values."""
    # Input map should have more than one channel.
    x = torch.ones(1, 1, 3)
    with pytest.raises(ValueError):
        kt.labels.collapse(x, labels=[1, 4])

    # Channel count should match number of labels.
    x = torch.ones(1, 3, 1)
    with pytest.raises(ValueError):
        kt.labels.collapse(x, labels=[1, 4])


def test_collapse_memory():
    """Test collapsing one-hot map with various label types in memory."""
    inp = torch.tensor((
        (0, 1, 0, 0),  # Channel 0.
        (1, 0, 1, 0),  # Channel 1.
        (0, 0, 0, 1),  # Channel 2.
    )).unsqueeze(0)
    x = (5, 4, 3)

    for labels in (tuple(x), list(x), torch.tensor(x), {i: 'hi' for i in x}):
        out = kt.labels.collapse(inp, labels)
        assert out.shape == (inp.shape[0], 1, *inp.shape[2:])
        assert out[0, 0].tolist() == [4, 5, 4, 3]


@pytest.mark.parametrize('ext', ['json', 'pickle', 'pt'])
def test_collapse_disk(tmp_path, ext):
    """Test collapsing one-hot map with various label types in file."""
    inp = torch.tensor((
        (1, 1, 0, 0),  # Channel 0.
        (0, 0, 1, 1),  # Channel 1.
    )).unsqueeze(0)
    x = (0, 7)

    f = tmp_path / f'labels.{ext}'
    for labels in (tuple(x), list(x), {i: 'hi' for i in x}):
        kt.io.save(labels, f)
        out = kt.labels.collapse(inp, labels=f)
        assert out[0, 0].tolist() == [0, 0, 7, 7]
