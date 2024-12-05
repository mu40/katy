"""Tests for index module."""


import torch
import pytest
import katy as kt


def test_save_load_generic(tmp_path):
    """Test saving and restoring data of generic Python types."""
    # JSON saves tuples as lists.
    ext = ('json', 'pickle', 'pt')
    inp = [1, 0.3, 'apple', [2], {'key': False, 'list': []}, None]

    for e in ext:
        path = tmp_path / f'testfile.{e}'

        kt.io.save(inp, path)
        assert kt.io.load(path) == inp


def test_save_load_tensor(tmp_path):
    """Test saving and restoring a tensor."""
    # JSON saves tuples as lists.
    path = tmp_path / 'testfile.pt'
    inp = torch.rand(8)

    kt.io.save(inp, path)
    assert kt.io.load(path).allclose(inp)


def test_save_illegal_extension(tmp_path):
    """Test saving to a file with an illegal extension."""
    path = tmp_path / 'testfile.csv'

    with pytest.raises(ValueError):
        kt.io.save(data=None, path=path)


def test_load_illegal_extension(tmp_path):
    """Test loading from a file with an illegal extension."""
    path = tmp_path / 'testfile.csv'

    with pytest.raises(ValueError):
        kt.io.load(path=path)
