#!/usr/bin/env python3
"""
fix_numpy_aliases.py

Conservative fixer for NumPy 2.x removed type aliases.

- Scans all .py files under current directory (recursively).
- Makes a .bak backup copy for each file it modifies.
- Replaces common removed aliases with safe alternatives:
    np.int        -> int
    np.float      -> float
    np.bool       -> bool
    np.object     -> object
    np.str        -> str
    np.complex    -> complex
    np.long       -> int
    np.unicode    -> str
    .astype(np.int)   -> .astype(int)
    .astype(np.float) -> .astype(float)
    dtype=np.int      -> dtype=int
    dtype=np.float    -> dtype=float
- Does NOT change explicit dtypes (np.int32, np.int64, np.float32, etc).
- Writes backups filename.py.bak next to modified files.

Usage: python fix_numpy_aliases.py
"""
import re
import shutil
from pathlib import Path

# conservative replacements
REPLACEMENTS = [
    # astype(...)
    (r"\.astype\(\s*np\.int\s*\)", 'astype(int)'),
    (r"\.astype\(\s*np\.float\s*\)", 'astype(float)'),
    (r"\.astype\(\s*np\.bool\s*\)", 'astype(bool)'),
    (r"\.astype\(\s*np\.object\s*\)", 'astype(object)'),
    (r"\.astype\(\s*np\.str\s*\)", 'astype(str)'),
    (r"\.astype\(\s*np\.complex\s*\)", 'astype(complex)'),

    # bare aliases -> builtins
    (r"\bnp\.int\b", 'int'),
    (r"\bnp\.float\b", 'float'),
    (r"\bnp\.bool\b", 'bool'),
    (r"\bnp\.object\b", 'object'),
    (r"\bnp\.str\b", 'str'),
    (r"\bnp\.complex\b", 'complex'),
    (r"\bnp\.long\b", 'int'),
    (r"\bnp\.unicode\b", 'str'),

    # dtype= uses
    (r"\bdtype\s*=\s*np\.int\b", 'dtype=int'),
    (r"\bdtype\s*=\s*np\.float\b", 'dtype=float'),
]

REPLACEMENTS = [(re.compile(p), r) for p, r in REPLACEMENTS]


def process_file(p: Path):
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        return False, 'read_error'

    new_text = text
    for pattern, repl in REPLACEMENTS:
        new_text = pattern.sub(repl, new_text)

    if new_text != text:
        bak = p.with_suffix(p.suffix + '.bak')
        shutil.copy2(p, bak)
        p.write_text(new_text, encoding='utf-8')
        return True, 'modified'
    return False, 'unchanged'


def main():
    repo_root = Path('.').resolve()
    py_files = [p for p in repo_root.rglob('*.py') if '/.git/' not in str(p) and 'site-packages' not in str(p)]
    modified = []
    for f in py_files:
        changed, status = process_file(f)
        if changed:
            modified.append(str(f))
            print(f'Modified: {f} ({status})')
    print('---')
    print(f'Total py files scanned: {len(py_files)}')
    print(f'Total modified: {len(modified)}')
    if len(modified) > 0:
        print('Backups written as *.py.bak next to each modified file.')

if __name__ == '__main__':
    main()
