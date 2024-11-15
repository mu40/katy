"""Tests for index module."""


import torch
import kathryn as kt


def test_ind2sub_sub2ind_python():
    size = (2, 3, 5)
    ind = (0, 1, 2, 5, 6, 23, 24, 25, 28, 29)

    for inp in ind:
        sub = kt.index.ind2sub(size, inp)
        out = kt.index.sub2ind(size, *sub)
        assert out == inp


def test_ind2sub_sub2ind_torch():
    size = torch.Size((2, 3, 5))
    inp = torch.tensor((0, 1, 2, 5, 6, 23, 24, 25, 28, 29))

    sub = kt.index.ind2sub(size, inp)
    out = kt.index.sub2ind(size, *sub)
    assert torch.all(out == inp)
