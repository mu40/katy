"""Unit tests for detecting changes to PyTorch."""

import pytest
import torch


def test_quantile():
    """Test if `torch.quantile` has been fixed for large tensors."""
    x = torch.zeros(2 ** 24 + 1)
    with pytest.raises(RuntimeError):
        x.quantile(q=0)
