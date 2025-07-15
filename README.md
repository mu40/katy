# Katy

Katy is a PyTorch-first, PyTorch-only toolbox for deep learning in medical imaging.
The library is in an early developmental stage.
Breaking changes may occur without warning.


## Todo

- Specialized image and transform types
- Global, thread-safe batch dimension control


## Development principles

- Write unit tests for any new code
- Depend only on Python's standard library and PyTorch
- Support the latest stable PyTorch only
- Lint, spell-check, and test every commit


## Experimental practices

- Keep configuration separate from code
- Use intermediate data representations
- Explore in notebooks, process with scripts


## Install tools

```sh
pip install -U pytest ruff typos
```


## Run checks

```sh
pytest tests/
ruff check --watch
typos
```


## Install pre-commit hook

```sh
cp hooks/pre-commit .git/hooks/pre-commit
```
