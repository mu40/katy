"""Tests for scripts."""

import pathlib
import pytest
import shutil
import subprocess


@pytest.fixture(scope='module')
def clone_repo(tmp_path_factory):
    """Copy repository once per module and return subprocess runner."""
    # Repository.
    for repo in pathlib.Path(__file__).resolve().parents:
        if (repo / 'pyproject.toml').exists():
            break

    else:
        pytest.fail(f'no pyproject.toml upstream of {__file__}')

    # Clean copy: `tmp_path_factory` fixture asince `tmp_path` function-scoped.
    copy = tmp_path_factory.mktemp('scratch') / 'copy'
    ignore = (repo / '.gitignore').read_text().split()
    ignore = shutil.ignore_patterns('.git', *ignore)
    shutil.copytree(repo, copy, ignore=ignore)

    # Helper. Run inside cloned repository. Pass empty environment to avoid
    # paths, variables leaking from local repository.
    def run(*f):
        p = subprocess.run(f, cwd=copy, env={})
        assert p.returncode == 0

    return run


def test_setup(clone_repo):
    """Test setting up a development environment."""
    clone_repo('./setup.sh')
