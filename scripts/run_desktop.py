"""Run the built native app while keeping all development data in this checkout."""
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent.parent
if sys.platform=='darwin':
    binary=root/'desktop/src-tauri/target/release/bundle/macos/Photo Dedup.app/Contents/MacOS/photo-dedup'
elif sys.platform=='win32':
    binary=root/'desktop/src-tauri/target/release/photo-dedup.exe'
else:
    binary=root/'desktop/src-tauri/target/release/photo-dedup'
if not binary.exists():
    raise SystemExit('Build the desktop application first; see docs/RELEASING.md.')
data=root/'local-validation/app-data'
data.mkdir(parents=True,exist_ok=True)
print(f'Application data stays here: {data}',flush=True)
raise SystemExit(subprocess.call([str(binary)],cwd=root,env=dict(os.environ,PHOTO_DEDUP_DATA_DIR=str(data))))
