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


def test_one_hot_unchanged():
    """Test if one-hotting leaves input unchanged."""
    inp = torch.zeros(1, 1, 4, dtype=torch.int64)

    orig = inp.clone()
    kt.labels.one_hot(inp, labels=1)
    assert inp.eq(orig).all()


def test_one_hot_illegal_inputs():
    """Test if one-hotting with illegal input values."""
    # Input label map should have one channel.
    x = torch.ones(1, 2, 3, dtype=torch.int64)
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, labels=2)

    # Highest label should be one less than the number of labels.
    x = torch.ones(1, 1, 3, dtype=torch.int64)
    with pytest.raises(ValueError):
        kt.labels.one_hot(x, labels=1)


def test_one_hot_values():
    """Test explicit result of one-hot encoding."""
    labels = 3
    inp = torch.tensor((*range(labels), -1))
    inp = inp.view(1, 1, inp.numel())
    out = kt.labels.one_hot(inp, labels)

    # The number of output channels should equal the non-negative labels.
    assert out.shape == (1, labels, inp.numel())

    # Index value i should lead to activation of channel i only.
    assert out[0, :, :-1].eq(torch.eye(labels)).all()

    # Negative values should not have any activation.
    assert out[0, :, -1].eq(0).all()


def test_rebase_labels():
    """Test rebasing labels of various type."""
    # Input does not have to include all the possible labels.
    inp = (1, 4, 4, 4, 5)
    labels = (6, 1, 4, 5)

    for dtype in (list, tuple, torch.tensor):
        ind, ind_to_inp = kt.labels.rebase(dtype(inp), labels)

        # Expect a `dict` mapping indices back to original labels.
        assert isinstance(ind_to_inp, dict)
        assert sorted(ind_to_inp.values()) == sorted(labels)
        assert ind.dtype == torch.int64
        assert tuple(ind_to_inp[i.item()] for i in ind) == inp


def test_rebase_mapping():
    """Test rebasing labels using unsorted input mapping."""
    # Input labels, all possible labels, mapping.
    inp = (1, 4, 4, 4, 5)
    labels = (0, 1, 4, 5)
    mapping = {1: 2, 4: 1}

    new_to_ind = {new: i for i, new in enumerate(sorted(mapping.values()))}
    remapped = [mapping.get(old) for old in inp]

    # Default value for labels missing in mapping keys should be -1.
    unknown = -1
    indices = [new_to_ind.get(new, unknown) for new in remapped]
    out, _ = kt.labels.rebase(inp, labels, mapping)
    assert out.tolist() == indices

    # Set explicit default value.
    unknown = 0
    indices = [new_to_ind.get(new, unknown) for new in remapped]
    out, _ = kt.labels.rebase(inp, labels, mapping, unknown=unknown)
    assert out.tolist() == indices


def test_rebase_illegal_arguments():
    """Test rebasing labels with illegal input arguments."""
    # Input label map should have one channel.
    x = torch.arange(3)

    # The unknown value must be an `int`.
    with pytest.raises(ValueError):
        kt.labels.rebase(x, labels=x, unknown=0.1)

    with pytest.raises(ValueError):
        kt.labels.rebase(x, labels=x, unknown=torch.tensor(1))

    # The input labels must not be a `dict`.
    with pytest.raises(ValueError):
        kt.labels.rebase(x, labels={i: i for i in x})


def test_rebase_disk(tmp_path):
    """Test rebasing labels with inputs stored on disk."""
    # Tensors are not JSON serializable.
    inp = torch.arange(3)
    labels = inp.unique().tolist()
    mapping = {i: i for i in labels}

    for ext in ('json', 'pickle', 'pt'):
        f_lab = tmp_path / f'labels.{ext}'
        f_map = tmp_path / f'mapping.{ext}'

        kt.io.save(labels, f_lab)
        kt.io.save(mapping, f_map)

        out, lut = kt.labels.rebase(inp, labels=f_lab)
        assert out.eq(inp).all()
        assert lut == mapping

        out, lut = kt.labels.rebase(inp, labels=f_lab, mapping=f_map)
        assert out.eq(inp).all()
        assert lut == mapping
