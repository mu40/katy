"""Tests for label-manipulation and image-synthesis module."""


import torch
import pytest
import kathryn as kt


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
