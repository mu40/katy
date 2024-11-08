"""Tests for transform module."""


import torch
import kathryn as kt


def test_compose_transform_size():
    angle = torch.randn(1)
    assert kt.transform.compose_rotation(angle).shape == (2, 2)

    angle = torch.randn(3)
    assert kt.transform.compose_rotation(angle).shape == (3, 3)

    angle = torch.randn(7, 9, 1)
    assert kt.transform.compose_rotation(angle).shape == (7, 9, 2, 2)

    angle = torch.randn(1, 1, 3)
    assert kt.transform.compose_rotation(angle).shape == (1, 1, 3, 3)


def test_index_to_torch():
    size = (17, 32, 63)
    dim = len(size)

    # Default normalized PyTorch grid.
    eye = torch.eye(dim, dim + 1).unsqueeze(0)
    y = torch.nn.functional.affine_grid(eye, size=(1, 1, *size))

    # Standard index coordinates.
    x = (torch.arange(s, dtype=y.dtype) for s in size)
    x = torch.meshgrid(*x, indexing='ij')
    x = torch.stack(x).view(dim, -1)

    # Convert indices to Pytorch coordinates.
    mat = kt.transform.index_to_torch(size)
    x = mat[:-1, :-1] @ x + mat[:-1, -1:]
    x = x.movedim(0, -1).view(1, *size, dim)

    assert y.allclose(x, atol=1e-6)


def test_torch_to_index():
    size = (17, 32, 63)

    for f in (True, False):
        fw = kt.transform.index_to_torch(size, align_corners=f)
        bw = kt.transform.torch_to_index(size, align_corners=f)
        assert fw.inverse().allclose(bw)
