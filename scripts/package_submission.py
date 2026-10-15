"""Create a deterministic source archive without runtime artifacts."""
from pathlib import Path
import zipfile
import runpy

root = Path(__file__).resolve().parents[1]
audit_module = runpy.run_path(str(root / 'scripts/audit_submission.py'))
result = audit_module['audit'](root)
if result['findings']:
    raise SystemExit('Privacy preflight failed; run scripts/audit_submission.py for redacted locations.')
archive = root / 'dist' / 'vaspo-source.zip'
archive.parent.mkdir(parents=True, exist_ok=True)
count = 0
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as out:
    # Share the audit's source selection so all packaged code is checked,
    # including verl/workers/rollout/. Include the public ignore rules too.
    paths = list(audit_module['source_files'](root))
    if (root / '.gitignore').is_file():
        paths.append(root / '.gitignore')
    for path in sorted(paths):
        rel = path.relative_to(root)
        entry = zipfile.ZipInfo('vaspo/' + rel.as_posix(), date_time=(2026, 10, 13, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = (0o100755 if path.suffix == '.sh' else 0o100644) << 16
        out.writestr(entry, path.read_bytes())
        count += 1
print(f'{archive}: {count} source files')
