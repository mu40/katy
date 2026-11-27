# Katy

A toolbox for deep learning in medical image analysis with PyTorch.
It is in early development and may change without notice.

## Setup

To use the toolbox as is, install the latest version:

```sh
pip install https://github.com/mu40/katy/archive/main.zip
```

### Development environment

To choose a specific PyTorch build, such as `cpu` or `cu130`, install it first:

```sh
pip install torch -i https://download.pytorch.org/whl/cpu
```

In a repository clone, install the package with the `dev` group:

```sh
pip install --group dev -e .
```

## Checks

In a development environment, install pre-commit hooks with:

```sh
prek install
```

Run all checks on all files:

```sh
prek run -a
```

## Development

- Write unit tests for new code
- Depend on Python's standard library and PyTorch
- Focus support on the latest stable PyTorch version
- Lint, spell-check, and test on commit

## Attribution

If you find this work useful, please cite the associated [tutorial paper](https://arxiv.org/abs/2507.13458):

```bibtex
@article{hoffmann2025domain,
  title={Domain-randomized deep learning for neuroimage analysis},
  author={Hoffmann, Malte},
  journal={IEEE Signal Processing Magazine},
  volume={42},
  number={4},
  pages={78--90},
  year={2025}
}
```
