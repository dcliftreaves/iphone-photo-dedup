"""Create a styled local DMG without Finder automation or remote publication."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import json
import os
import platform

from ds_store import DSStore

root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--app',type=Path,default=root/'desktop/src-tauri/target/release/bundle/macos/Photo Dedup.app')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
if sys.platform!='darwin':raise SystemExit('DMG packaging requires macOS')
version=json.loads((root/'desktop/package.json').read_text())['version']
output=args.output or root/'dist'/f'Photo-Dedup-{version}-macos-{platform.machine()}.dmg'
if output.exists():raise SystemExit(f'Refusing to overwrite {output}')
subprocess.run(['codesign','--verify','--deep','--strict',str(args.app)],check=True)
output.parent.mkdir(parents=True,exist_ok=True)
validation=root/'local-validation'
validation.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='dmg-stage-',dir=validation) as directory:
    stage=Path(directory)
    subprocess.run(['ditto',str(args.app),str(stage/'Photo Dedup.app')],check=True)
    (stage/'Applications').symlink_to('/Applications',target_is_directory=True)
    with DSStore.open(str(stage/'.DS_Store'),'w+') as store:
        store['Photo Dedup.app']['Iloc']=(170,155)
        store['Applications']['Iloc']=(450,155)
        store['.']['bwsp']={'ShowToolbar':False,'ShowStatusBar':False,'ShowSidebar':False,
                            'ShowTabView':False,'ContainerShowSidebar':False,
                            'WindowBounds':'{{120, 120}, {640, 340}}','PreviewPaneVisibility':False}
        store['.']['icvp']={'viewOptionsVersion':1,'backgroundType':1,'backgroundColorRed':.97,
                           'backgroundColorGreen':.98,'backgroundColorBlue':.96,'iconSize':100.,
                           'textSize':14.,'arrangeBy':'none','gridSpacing':100.,'gridOffsetX':0.,
                           'gridOffsetY':0.,'showIconPreview':True,'showItemInfo':False,
                           'labelOnBottom':True}
        store['.']['vstl']=('type',b'icnv')
    subprocess.run(['hdiutil','create','-volname','Photo Dedup · Drag to Applications','-srcfolder',str(stage),
                    '-format','UDZO',str(output)],check=True)
if os.environ.get('SIGNED_RELEASE', '').lower() == 'true':
    identity=os.environ.get('APPLE_SIGNING_IDENTITY')
    if not identity:raise SystemExit('Missing Developer ID identity for the signed installer')
    subprocess.run(['codesign','--sign',identity,'--timestamp',str(output)],check=True)
subprocess.run(['hdiutil','verify',str(output)],check=True)
with output.open('rb') as stream:
    digest=hashlib.file_digest(stream,'sha256').hexdigest()
output.with_suffix(output.suffix+'.sha256').write_text(f'{digest}  {output.name}\n')
print(f'Local installer ready: {output}')
print('Signed installer prepared.' if os.environ.get('SIGNED_RELEASE', '').lower() == 'true' else
      'Development build: public distribution still requires Developer ID signing and notarization.')
