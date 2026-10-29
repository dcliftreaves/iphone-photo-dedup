"""Check that bundled native libraries support the advertised macOS minimum."""
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
if sys.platform != 'darwin':
    raise SystemExit(0)
minimum = json.loads((root/'desktop/src-tauri/tauri.conf.json').read_text())['bundle']['macOS']['minimumSystemVersion']
def version(value):
    return tuple((list(map(int,value.split('.'))) + [0,0])[:3])
failures=[]
checked=set()
for path in (root/'desktop/src-tauri/engine').rglob('*'):
    if not path.is_file() or path.resolve() in checked:
        continue
    checked.add(path.resolve())
    with path.open('rb') as stream:
        magic=stream.read(4)
    if magic not in (b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'):
        continue
    result=subprocess.run(['otool','-l',str(path)],capture_output=True,text=True,check=True)
    for required in re.findall(r'(?:LC_BUILD_VERSION[\s\S]{0,100}?minos|LC_VERSION_MIN_MACOSX[\s\S]{0,100}?version)\s+(\d+(?:\.\d+){0,2})',result.stdout):
        if version(required)>version(minimum):
            failures.append(f'{path.name} needs macOS {required}')
if failures:
    raise SystemExit('Bundled runtime exceeds macOS '+minimum+':\n'+'\n'.join(sorted(set(failures)))+
                     '\nBuild with a portable Python distribution, not a host-specific Homebrew interpreter.')
print('Bundled native libraries support the advertised macOS minimum: '+minimum)
