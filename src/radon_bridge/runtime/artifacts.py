"""Resolve artifact paths without embedding a user's server layout or changing records."""
import hashlib
import json
import os
from pathlib import Path


def project_root():
    override = os.environ.get("RB_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise ValueError("RB_PROJECT_ROOT must be absolute")
        return root.resolve()
    for root in Path(__file__).resolve().parents:
        if (root / "project.json").is_file():
            return root
    raise ValueError("Set RB_PROJECT_ROOT when running outside an editable project")


def _rooted(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else project_root() / p).resolve()


def _mapping():
    filename = os.environ.get("RB_ARTIFACT_MAP")
    if not filename:
        return []
    config = json.loads(_rooted(filename).read_text())
    if config.get("schema") != "artifact_path_map_v1":
        raise ValueError("Unsupported artifact map")
    rows = []
    for row in config["mappings"]:
        source = Path(row["source"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("Historical source prefixes must be absolute")
        rows.append((source, _rooted(row["target"])))
    if len({str(a) for a, _ in rows}) != len(rows):
        raise ValueError("Duplicate historical source prefix")
    return sorted(rows, key=lambda row: len(row[0].parts), reverse=True)


def resolve(path):
    p = _rooted(path)
    if p.exists():
        return p
    for source, target in _mapping():
        if p.is_relative_to(source):
            return target / p.relative_to(source)
    return p


def sha256(path):
    digest = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def relocate(value):
    """Create a runtime copy; only explicitly mapped absolute paths change."""
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        p = Path(value)
        if any(p.is_relative_to(source) for source, _ in _mapping()):
            return str(resolve(p))
    return value
