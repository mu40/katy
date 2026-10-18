#!/bin/sh

# Set up a virtual Python environment for development.

set -e
cd "$(dirname "$0")"
venv_dir='.venv'


# Arguments.
if [ $# -ne 1 ]; then
    echo "Usage: $(basename "$0") PYTORCH_PLATFORM"
    echo "Platforms: cpu or cu<N> (https://pytorch.org/get-started)"
    exit 1
fi
platform="$1"


# Virtual environment.
if ! "$venv_dir/bin/python" -c 'import pip' 2>/dev/null; then
    python=$(
        find /usr/bin/ /usr/local/bin/ -name 'python*' |
        grep 'python[0-9.]*$' |
        sort -V |
        tail -n1
    )
    "$python" -m venv --clear "$venv_dir"
    . "$venv_dir/bin/activate"

    # Packages.
    pip install -U pip
    pip install -i "https://download.pytorch.org/whl/$platform" torch
    pip install -e '.[dev]'
fi


# Git hooks. Relative symlinks, so edits apply without rerunning setup.
if [ -d .git/hooks ]; then
    for f in hooks/*; do
        ln -sfn "../../$f" ".git/hooks/${f#hooks/}"
    done
    if f=$(command -v commit-msg.py 2>/dev/null); then
        ln -sfn "$f" .git/hooks/commit-msg
    fi
fi


# Environment manager.
cat >.envrc <<EOF
venv_dir='$venv_dir'
if [ -d "\$venv_dir/bin" ]; then
    export VIRTUAL_ENV="\$PWD/\$venv_dir"
    PATH_add "\$VIRTUAL_ENV/bin"
fi
EOF
