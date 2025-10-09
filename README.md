# Katy

Katy is a PyTorch-first toolbox for deep learning in medical image analysis.
The library is in an early developmental stage.
Breaking changes may occur without warning.


## Ideas

- Specialized image and transform types
- Global, thread-safe batch dimension control


## Setup

Install a local virtual environment for development and testing, with pre-commit hooks for linting.

```sh
./setup.sh
```


## Development

- Write unit tests for new code
- Depend on Python's standard library and PyTorch
- Focus support on the latest stable PyTorch
- Lint, spell-check, and test on commit


## Run checks

```sh
pytest -xsv
ruff check --watch
typos
```


## Attribution

If you find this work useful, please cite the [paper it was developed for](https://arxiv.org/abs/2507.13458):

```bibtex
@article{hoffmann2025domain,
  title={Domain-randomized deep learning for neuroimage analysis},
  author={Hoffmann, Malte},
  journal={arXiv preprint arXiv:2507.13458},
  year={2025}
}
```
