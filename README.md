# Katy

Katy is a toolbox for deep learning in medical image analysis with PyTorch.
It is in early development, and breaking changes may occur without warning.


## Ideas

- Specialized image and transform types
- Global, thread-safe batch dimension control


## Setup

Install a local virtual environment for development, with pre-commit hooks.

```sh
./setup.sh
```


## Development

- Write unit tests for new code
- Depend on Python's standard library and PyTorch
- Focus support on the latest stable PyTorch version
- Lint, spell-check, and test on commit


## Run checks

```sh
pytest -xsv
ruff check --watch
shellcheck hooks/* setup.sh
typos
```


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
