#!/bin/sh

# Setup up virtual Python environment for linting and testing.

ENV='.venv'


# Virtual environment.
if [ ! -d "$ENV" ]; then
    py=$(find /usr/bin/ /usr/local/bin/ -regex '.*/python[0-9.]*' |
        sort -V | tail -n1)
    "$py" -m venv "$ENV"
    . "$ENV/bin/activate"

    # Packages.
    pip install -U pip setuptools wheel
    pip install pytest ruff shellcheck-py typos
    pip install -i https://download.pytorch.org/whl/cpu torch
fi


# Hooks.
cp -v hooks/* .git/hooks
if f=$(command -v commit-msg.py 2>/dev/null); then
    ln -vsfn "$f" .git/hooks/commit-msg
fi


# Environment manager.
cat >.envrc <<EOF
[ -d "$ENV" ] && . "$ENV/bin/activate"
export PYTHONPATH="$(dirname "$(realpath "$0")")"
EOF
