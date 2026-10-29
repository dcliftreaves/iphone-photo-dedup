#!/usr/bin/env python3
"""Build a relocatable Python engine with its image libraries and ExifTool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
EXIFTOOL_VERSION = '13.60'


def download(url, destination):
    if not destination.exists():
        request = urllib.request.Request(url, headers={'User-Agent':'Photo-Dedup-build/0.2'})
        partial = destination.with_suffix(destination.suffix + '.part')
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as target:
            shutil.copyfileobj(response, target)
        partial.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-exiftool', action='store_true', help='Development only: build without metadata tools')
    args = parser.parse_args()
    tools = ROOT / '.tools' / 'packaging'
    tools.mkdir(parents=True, exist_ok=True)
    extra = []
    if not args.skip_exiftool:
        if sys.platform == 'win32':
            archive = tools / f'exiftool-{EXIFTOOL_VERSION}_64.zip'
            download(f'https://downloads.sourceforge.net/project/exiftool/exiftool-{EXIFTOOL_VERSION}_64.zip', archive)
            folder = tools / 'windows-exiftool'
            if not folder.exists():
                with zipfile.ZipFile(archive) as source:
                    source.extractall(folder)
            executable = next(folder.rglob('exiftool(-k).exe'), None) or next(folder.rglob('exiftool.exe'))
            prepared = tools / 'exiftool'
            if prepared.exists():
                shutil.rmtree(prepared)
            prepared.mkdir()
            shutil.copy2(executable, prepared / 'exiftool.exe')
            shutil.copytree(executable.parent / 'exiftool_files', prepared / 'exiftool_files')
        else:
            archive = tools / f'exiftool-{EXIFTOOL_VERSION}.tar.gz'
            download(f'https://github.com/exiftool/exiftool/archive/refs/tags/{EXIFTOOL_VERSION}.tar.gz', archive)
            prepared = tools / f'exiftool-{EXIFTOOL_VERSION}'
            if not prepared.exists():
                with tarfile.open(archive) as source:
                    source.extractall(tools, filter='data')
            (prepared / 'exiftool').chmod(0o755)
        if sys.platform != 'win32':
            curated = tools / 'exiftool-runtime'
            if curated.exists():
                shutil.rmtree(curated)
            curated.mkdir()
            shutil.copy2(prepared / 'exiftool', curated / 'exiftool')
            shutil.copytree(prepared / 'lib', curated / 'lib')
            for name in ('README', 'Changes', 'html/faq.html'):
                if (prepared / name).is_file():
                    shutil.copy2(prepared / name, curated / Path(name).name)
            prepared = curated
        extra = ['--add-data', f'{prepared}{os.pathsep}tools/exiftool']
    os.environ.setdefault('PYINSTALLER_CONFIG_DIR', str(tools / 'pyinstaller-cache'))
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
                    '--name', 'photo-dedup-engine', '--distpath', str(ROOT/'dist'),
                    '--workpath', str(ROOT/'build'/'pyinstaller'), '--specpath', str(ROOT/'build'),
                    '--paths', str(ROOT), '--collect-all', 'pillow_heif',
                    '--collect-submodules', 'PIL', '--copy-metadata', 'pillow-heif',
                    '--copy-metadata', 'Pillow', '--copy-metadata', 'numpy',
                    '--add-data', f'{ROOT/"LICENSE"}{os.pathsep}licenses',
                    *(['--codesign-identity', os.environ['APPLE_SIGNING_IDENTITY']]
                      if sys.platform == 'darwin' and os.environ.get('APPLE_SIGNING_IDENTITY') else []),
                    *extra, str(ROOT/'scripts'/'worker_entry.py')], check=True, cwd=ROOT)
    target = ROOT / 'desktop' / 'src-tauri' / 'engine'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT/'dist'/'photo-dedup-engine', target)
    # Distribute source and notices alongside the tool; the third-party tree includes
    # its own license texts. PyInstaller collects native image library dependencies.
    (target/'BUILD-INFO.json').write_text(json.dumps(dict(engine='0.2.0', exiftool=EXIFTOOL_VERSION,
        python=sys.version, platform=sys.platform), indent=2))
    print(f'Engine ready: {target}')


if __name__ == '__main__':
    main()
