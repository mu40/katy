"""System tests for package install."""

import pathlib
import subprocess
import sys


def test_install_and_import(tmp_path):
    """Test package install and import with PIP."""
    # Paths.
    venv = tmp_path
    pip = venv / 'bin' / 'pip'
    python = venv / 'bin' / 'python'

    # Repository.
    repo = None
    for d in pathlib.Path(__file__).resolve().parents:
        if (d / 'pyproject.toml').exists():
            repo = d
            break

    assert repo, f'no pyproject.toml upstream of {__file__}'

    # Helper. Run outside of repository to be able to install.
    def run(*f):
        p = subprocess.run(
            f,
            cwd=venv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={'PIP_CACHE_DIR': str(venv / 'cache')},
        )
        assert not p.returncode
        print(p.stdout)

    # Virtual environment.
    run(sys.executable, '-m', 'venv', venv)

    # Upgrade. Needed to find latest CPU version.
    run(pip, 'install', '-U', 'pip', 'setuptools')

    # CPU-only PyTorch. Avoid lengthy GPU install.
    run(pip, 'install', 'torch', '-i', 'https://download.pytorch.org/whl/cpu')

    # Install, import.
    run(pip, 'install', repo)
    run(python, '-c', 'import katy')
