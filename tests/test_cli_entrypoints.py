"""
Entrypoint smoke tests.

Each console_script declared in [project.scripts] must be importable and
runnable from an installed environment without any PYTHONPATH manipulation.
This guards against packaging regressions where a top-level package (e.g.
`quality`, added in 81661af) exists in the repo but is missing from the
setuptools package discovery config, so the editable install works only
when the repo root happens to already be on sys.path.

Tests invoke the entrypoints via subprocess with cwd set to a tmp_path
outside the repo, so a missing package registration cannot be masked by
the current working directory being on sys.path.
"""

import os
import shutil
import subprocess

import pytest

ENTRYPOINTS = ["nn-pipeline", "noema-gate", "noema-audit", "noema-evidence"]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_entrypoint_help_exits_zero(entrypoint, tmp_path):
    exe = shutil.which(entrypoint)
    assert exe is not None, f"{entrypoint} console_script not found on PATH"

    # Deliberately drop any inherited PYTHONPATH so the resolved package
    # discovery (not an ambient sys.path hack) is what makes this pass.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [exe, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"{entrypoint} --help failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
