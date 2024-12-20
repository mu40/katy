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
        assert path.is_file()


def test_save_load_tensor(tmp_path):
    """Test saving and restoring a tensor."""
    # JSON saves tuples as lists.
    path = tmp_path / 'testfile.pt'
    inp = torch.rand(8)

    kt.io.save(inp, path)
    assert kt.io.load(path).allclose(inp)
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


def test_read_color_table(tmp_path):
    """Test reading a FreeSurfer color table."""
    lut = (
        '#No. Label Name:                            R   G   B   A\n'
        '\n'
        '0   Unknown                                 0   0   0   0\n'
        '1   Left-Cerebral-Exterior                  70  130 180 0\n'
    )

    # Test data.
    path = tmp_path / 'lut.txt'
    with open(path, mode='w') as f:
        f.write(lut)

    lut = kt.io.read_color_table(path)
    assert isinstance(lut, dict)
    assert tuple(lut) == (0, 1)
    assert tuple(lut[0]) == ('name', 'color')

    assert lut[0]['name'] == 'Unknown'
    assert lut[0]['color'] == (0, 0, 0)
    assert lut[1]['name'] == 'Left-Cerebral-Exterior'
    assert lut[1]['color'] == (70, 130, 180)
