import importlib.util
from pathlib import Path

PY_FILES = [
    Path('ml') / 'predict.py',
    Path('ml') / 'train.py',
    Path('ml') / 'btc' / 'predict.py',
    Path('ml') / 'cl' / 'predict.py',
]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(str(path), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_python_scripts_importable():
    for path in PY_FILES:
        assert path.exists(), f"File not found: {path}"
        load_module(path)
