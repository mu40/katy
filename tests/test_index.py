"""Tests for index module."""


import torch
import kathryn as kt


def test_ind2sub_sub2ind_python():
    """Test circular conversion for Python integers."""
    size = (2, 3, 5)
    ind = (0, 1, 2, 5, 6, 23, 24, 25, 28, 29)

    for inp in ind:
        sub = kt.index.ind2sub(size, inp)
        out = kt.index.sub2ind(size, *sub)
        assert out == inp


def test_ind2sub_sub2ind_torch():
    """Test circular conversion for PyTorch tensors."""
    size = torch.tensor((2, 3, 5))
    inp = torch.tensor((0, 1, 2, 5, 6, 23, 24, 25, 28, 29))

    sub = kt.index.ind2sub(size, inp)
    out = kt.index.sub2ind(size, *sub)
    assert all(out == inp)


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
