"""Check packaged source for direct identifiers and credential literals.

Copyright and license attribution is intentionally preserved. This heuristic
check does not certify anonymity against inference from research content.
"""
from pathlib import Path
import re

EXCLUDED_PARTS = {'__pycache__', 'outputs', 'checkpoints', 'swanlog', 'wandb', 'tensorboard', 'dist', 'build', 'venv'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.orig', '.bkp', '.log', '.pt', '.pth', '.safetensors', '.parquet'}
PATTERNS = {
    'personal_absolute_path': rb'(?:/(?:Users|home)/[^/\s"\']+|[A-Za-z]:\\Users\\[^\\\s"\']+)',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'credential_literal': rb'\b(?:olp_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b',
    'email_address': rb'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|cn|edu|hk|org|net|io)\b',
    'experiment_account_url': rb'https?://(?:wandb\.ai|swanlab\.cn)/[^\s"\']+',
}


def source_files(root):
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if not path.is_file() or any(part.startswith('.') or part in EXCLUDED_PARTS for part in rel.parts):
            continue
        # Only root-level rollout/ holds runtime data. verl/workers/rollout/
        # and other nested rollout directories contain implementation code.
        if rel.parts[0] == 'rollout':
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path


def audit(root):
    findings = []
    count = 0
    for path in source_files(root):
        data = path.read_bytes()
        count += 1
        for category, pattern in PATTERNS.items():
            for match in re.finditer(pattern, data):
                # Never expose a matched credential or identifier in the report.
                findings.append({'file': path.relative_to(root).as_posix(),
                                 'line': data[:match.start()].count(b'\n') + 1,
                                 'category': category})
    return {'source_files_checked': count, 'findings': findings}


if __name__ == '__main__':
    import json
    import sys
    result = audit(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
    sys.exit(bool(result['findings']))
