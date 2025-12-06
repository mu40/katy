"""System tests for package install."""

import pathlib
import pytest
import shutil
import subprocess
import venv


@pytest.fixture(scope='module')
def package_install(tmp_path_factory):
    """Install package with PIP once per module and return Python function."""
    # Repository.
    for repo in pathlib.Path(__file__).resolve().parents:
        if (repo / 'pyproject.toml').exists():
            break

    else:
        pytest.fail(f'no pyproject.toml upstream of {__file__}')

    # Paths. Need `tmp_path_factory` fixture since `tmp_path` function-scoped.
    d = tmp_path_factory.mktemp('scratch')
    copy = d / 'repo'
    env = d / 'env'
    pip = env / 'bin' / 'pip'
    python = env / 'bin' / 'python'

    # Clean copy without pre-built wheel.
    ignore = (repo / '.gitignore').read_text().split()
    ignore = shutil.ignore_patterns('.git', *ignore)
    shutil.copytree(repo, copy, ignore=ignore)

    # Helper. Run outside of repository to be able to install. Pass empty
    # environment to avoid paths, variables leaking from local repository.
    def run(*f):
        p = subprocess.run(f, cwd=env, env={})
        assert p.returncode == 0

    # Virtual environment. Upgrade for finding latest CPU-only PyTorch.
    # CPU-only PyTorch avoids lengthy GPU install.
    venv.EnvBuilder(with_pip=True).create(env)
    run(pip, 'install', '-U', 'pip', 'setuptools')
    run(pip, 'install', 'torch', '-i', 'https://download.pytorch.org/whl/cpu')
    run(pip, 'install', copy)

    return lambda f: run(python, '-c', f)


def test_package_import(package_install):
    """Test importing installed package."""
    package_install(
        'import katy\n'
        'assert "site-packages" in katy.__file__\n'
    )


def test_default_colors(package_install):
    """Test reading the default color lookup table."""
    package_install(
        'import katy\n'
        'lut = katy.io.default_colors()\n'
        'assert lut\n'
    )
