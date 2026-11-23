"""Scoped imports of the two licensed SegNeXt author source files."""
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'third_party'
def package(name,path):
    if name not in sys.modules:
        m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
    return sys.modules[name]

def load(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    try:spec.loader.exec_module(m)
    except BaseException:sys.modules.pop(name,None);raise
    return m

@contextmanager
def temporary_modules(mapping):
    previous={k:sys.modules.get(k)for k in mapping}
    sys.modules.update(mapping)
    try:yield
    finally:
        for k,value in previous.items():
            if value is None:sys.modules.pop(k,None)
            else:sys.modules[k]=value
