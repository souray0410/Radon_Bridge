"""Read-only relocation of immutable historical artifacts; never rewrite JSON."""
import hashlib
import os
from pathlib import Path

STUDY = '2026_09_06_14_05_08'
SOURCE = Path('/data/mengh/RadonBridge')
ARCHIVE = Path('/backup/mengh/RadonBridge/archive') / STUDY

def resolve(path):
    p = Path(path)
    # Existing paths retain precedence until verified retirement is complete.
    if p.exists(): return p
    try: relative = p.relative_to(SOURCE)
    except ValueError: return p
    current_prefix=Path('runs')/STUDY
    if relative.is_relative_to(current_prefix):
        return ARCHIVE/'current'/relative.relative_to(current_prefix)
    root = Path(os.environ.get('RB_HISTORY_ARCHIVE', str(ARCHIVE / 'history')))
    return root / relative

def sha256(path):
    digest = hashlib.sha256()
    with resolve(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    return digest.hexdigest()

def relocate(value):
    """Runtime copy only; hashes, basis provenance, and archived originals stay intact."""
    if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list): return [relocate(v) for v in value]
    if isinstance(value, str) and value.startswith(str(SOURCE) + '/'): return str(resolve(value))
    return value
