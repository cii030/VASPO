"""Check source archives include rollout code and exclude runtime artifacts."""

from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_tree(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ['audit_submission.py', 'package_submission.py']:
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    files = {
        'README.md': '# VASPO\n',
        '.gitignore': 'dist/\n',
        'verl/workers/rollout/worker.py': 'value = 1\n',
        'rollout/generated.json': '{}\n',
        'outputs/run.json': '{}\n',
        '.env': 'LOCAL_SETTING=private\n',
        '.git/config': 'local git metadata\n',
        '.venv/local.py': '# local environment\n',
        'verl/__pycache__/worker.pyc': 'cache\n',
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


def test_archive_retains_rollout_source_and_is_deterministic(source_tree):
    command = [sys.executable, str(source_tree / 'scripts/package_submission.py')]
    subprocess.run(command, check=True, capture_output=True)
    archive = source_tree / 'dist/vaspo-source.zip'
    first = archive.read_bytes()
    with zipfile.ZipFile(archive) as packed:
        names = set(packed.namelist())
    assert names == {
        'vaspo/README.md',
        'vaspo/.gitignore',
        'vaspo/scripts/audit_submission.py',
        'vaspo/scripts/package_submission.py',
        'vaspo/verl/workers/rollout/worker.py',
    }
    subprocess.run(command, check=True, capture_output=True)
    assert archive.read_bytes() == first


def test_audit_checks_nested_rollout_implementation(source_tree):
    # Assemble a synthetic marker without putting a credential literal in source.
    worker = source_tree / 'verl/workers/rollout/worker.py'
    worker.write_text('value = "' + 'ghp_' + 'x' * 24 + '"\n')
    audit = runpy.run_path(str(source_tree / 'scripts/audit_submission.py'))['audit']
    findings = audit(source_tree)['findings']
    assert findings == [{
        'file': 'verl/workers/rollout/worker.py',
        'line': 1,
        'category': 'credential_literal',
    }]
    result = subprocess.run(
        [sys.executable, str(source_tree / 'scripts/package_submission.py')],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (source_tree / 'dist/vaspo-source.zip').exists()
