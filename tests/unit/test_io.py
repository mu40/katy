"""Unit tests for index module."""

import katy as kt
import pathlib
import pytest
import torch


@pytest.mark.parametrize('ext', ['json', 'pickle', 'pt'])
def test_save_load_generic(tmp_path, ext):
    """Test saving and restoring data of generic Python types."""
    # JSON saves tuples as lists.
    path = tmp_path / f'testfile.{ext}'
    x = [1, 0.3, 'apple', [2], {'key': False, 'list': []}, None]

    kt.io.save(x, path)
    assert kt.io.load(path) == x
    assert path.is_file()


def test_save_load_tensor(tmp_path):
    """Test saving and restoring a tensor."""
    # JSON saves tuples as lists.
    path = tmp_path / 'testfile.pt'
    x = torch.arange(4)

    kt.io.save(x, path)
    assert kt.io.load(path).equal(x)
    assert path.is_file()


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


@pytest.mark.parametrize('dtype', [str, pathlib.Path])
def test_read_colors(tmp_path, dtype):
    """Test reading a FreeSurfer color lookup table."""
    lut = (
        '#No. Label Name:                R   G   B   A\n'
        '\n'
        '0   Unknown                     0   0   0   0\n'
        '1   Left-Cerebral-Exterior      70  130 180 0\n'
    )

    path = dtype(tmp_path / 'lut.txt')
    with open(path, mode='w') as f:
        f.write(lut)

    lut = kt.io.read_colors(path)
    assert isinstance(lut, dict)
    assert lut[0]['name'] == 'Unknown'
    assert lut[0]['color'] == (0, 0, 0)
    assert lut[1]['name'] == 'Left-Cerebral-Exterior'
    assert lut[1]['color'] == (70, 130, 180)


def test_default_colors():
    """Test reading the default color lookup table."""
    lut = kt.io.default_colors()
    assert isinstance(lut, dict)
    assert lut[0]['name'] == 'Unknown'
    assert lut[0]['color'] == (0, 0, 0)
    assert lut[3]['name'] == 'Left-Cerebral-Cortex'
    assert lut[3]['color'] == (205, 62, 78)
