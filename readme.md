# Kathryn


## Todo

- Perlin
- Augmentation
- Specialized data types
- Compose with `index_to_torch` in `transform`, and/or remove `interpolate`?


## Principles

- Write unit tests
- Use existing libraries
- Keep configuration separate from code
- Use intermediate data representations
- Lint, spell-check, and test automatically
- Explore with notebooks, process with scripts


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
