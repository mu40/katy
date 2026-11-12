#!/bin/sh

# Set up a virtual Python environment for development.

set -e
cd "$(dirname "$0")"
venv_dir='.venv'


# Arguments.
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $(basename "$0") PLATFORM [PYTHON]"
    echo "Platform: cpu or cu<N> (https://pytorch.org/get-started)"
    echo "Python: command or path (default python3)"
    exit 1
fi
platform="$1"
python="${2:-python3}"


# Virtual environment.
if ! "$venv_dir/bin/python" -c 'import pip' 2>/dev/null; then
    "$python" -m venv --clear "$venv_dir"
    "$venv_dir/bin/python" -m pip install -U pip
fi
. "$venv_dir/bin/activate"


# Packages.
pip install -i "https://download.pytorch.org/whl/$platform" torch
pip install -e . --group dev


# Git hooks. Relative symlinks, so edits apply without rerunning setup.
if [ -d .git/hooks ]; then
    for f in hooks/*; do
        ln -sfn "../../$f" ".git/hooks/${f#hooks/}"
    done
fi


# Environment manager.
cat >.envrc <<EOF
venv_dir='$venv_dir'
if [ -d "\$venv_dir/bin" ]; then
    export VIRTUAL_ENV="\$PWD/\$venv_dir"
    PATH_add "\$VIRTUAL_ENV/bin"
fi
EOF
