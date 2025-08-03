# Katy

Katy is a PyTorch-first, PyTorch-only toolbox for deep learning in medical imaging.
The library is in an early developmental stage.
Breaking changes may occur without warning.


## Todo

- Specialized image and transform types
- Global, thread-safe batch dimension control


## Setup

Install a local virtual environment for development and testing, with pre-commit hooks for linting.

```sh
./setup.sh
```


## Development principles

- Write unit tests for any new code
- Depend only on Python's standard library and PyTorch
- Support the latest stable PyTorch only
- Lint, spell-check, and test every commit


## Experimental practices

- Keep configuration separate from code
- Use intermediate data representations
- Explore in notebooks, process with scripts


## Run checks

```sh
pytest
ruff check --watch
typos
```


## Attribution

If you find this work useful, please cite the [paper it was developed for](https://doi.org/10.1162/imag_a_00197):

```bibtex
@article{hoffmann2024anatomy,
  title={{Anatomy-aware and acquisition-agnostic joint registration with SynthMorph}},
  author={Hoffmann, Malte and Hoopes, Andrew and Greve, Douglas N and Fischl, Bruce and Dalca, Adrian V},
  journal={Imaging Neuroscience},
  volume={2},
  pages={1--33},
  year={2024}
}
```
