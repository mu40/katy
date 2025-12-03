"""Unit tests for index module."""

import katy as kt
import pytest
import torch


@pytest.mark.parametrize('ind', [1, 6, 23, 24, 29])
def test_ind2sub_sub2ind_python(ind):
    """Test circular conversion for Python integers."""
    size = (2, 3, 5)
    sub = kt.index.ind2sub(size, ind)
    out = kt.index.sub2ind(size, *sub)
    assert out == ind


def test_ind2sub_sub2ind_torch():
    """Test circular conversion for PyTorch tensors."""
    size = torch.tensor((2, 3, 5))
    ind = torch.tensor((0, 1, 6, 23, 24, 29))

    sub = kt.index.ind2sub(size, ind)
    out = kt.index.sub2ind(size, *sub)
    assert all(out == ind)


def test_sub2ind_unchanged():
    """Test if converting subscripts to integers leaves inputs unchanged."""
    size = torch.Size((2, 3))
    inp = tuple(torch.tensor(s) for s in (1, 2))

    orig = tuple(i.clone() for i in inp)
    kt.index.sub2ind(size, *inp)
    assert inp == orig


def test_ind2sub_unchanged():
    """Test if converting integers to subscripts leaves inputs unchanged."""
    size = torch.Size((2, 3))
    inp = torch.tensor(5)

    orig = inp.clone()
    kt.index.ind2sub(size, inp)
    assert inp == orig
