"""Compatibility imports for pre-2026-09-09 code and serialized Python objects.

New code imports radon_bridge. Aliases share the same module/class objects;
legacy imports must not instantiate a second implementation.
"""
import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / 'radon_bridge')]

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target
    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        return module
    def exec_module(self, module):
        module.__spec__ = self.canonical_spec
    def get_code(self, fullname):
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)
    def is_package(self, fullname):
        return importlib.util.find_spec(self.target).submodule_search_locations is not None

class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('radonbridge.'):
            return None
        canonical = 'radon_bridge.' + fullname[len('radonbridge.'):]
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical), origin=spec.origin,
            is_package=spec.submodule_search_locations is not None)

if not any(type(f).__module__ == __name__ and type(f).__name__ == '_AliasFinder'
           for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
