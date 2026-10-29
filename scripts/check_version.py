"""Fail a release if tag, Python, frontend, and native versions disagree."""
import json
import os
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parent.parent
python = tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
frontend = json.loads((root/'desktop/package.json').read_text())['version']
native = tomllib.loads((root/'desktop/src-tauri/Cargo.toml').read_text())['package']['version']
config = json.loads((root/'desktop/src-tauri/tauri.conf.json').read_text())['version']
tag = os.environ.get('RELEASE_TAG', 'v' + python)
assert {python, frontend, native, config, tag.removeprefix('v')} == {python}, 'Release version mismatch'
print(f'Package versions agree: {python}')
