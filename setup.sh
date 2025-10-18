#!/bin/sh

# Set up a virtual Python environment for development.

set -e
ENV='.venv'


if [ ! -f .gitignore ]; then
    echo "ERROR: not in the top-level repository directory"
    exit 1
fi


# Virtual environment.
if [ ! -d "$ENV" ]; then
    py=$(find /usr/bin/ /usr/local/bin/ -regex '.*/python[0-9.]*' |
        sort -V | tail -n1)
    "$py" -m venv "$ENV"
    . "$ENV/bin/activate"

    # Packages.
    pip install -U pip setuptools
    pip install pytest ruff shellcheck-py typos
    pip install -i https://download.pytorch.org/whl/cpu torch
fi


# Git hooks.
if [ -d .git/hooks ]; then
    cp -v hooks/* .git/hooks
    if f=$(command -v commit-msg.py 2>/dev/null); then
        ln -vsfn "$f" .git/hooks/commit-msg
    fi
fi


# Environment manager.
cat >.envrc <<EOF
[ -d "$ENV" ] && . "$ENV/bin/activate"
export PYTHONPATH="$PWD"
EOF
