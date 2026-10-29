"""Attach verified installers to a draft GitHub release, only inside approved CI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',required=True)
    args=parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS')!='true':
        raise SystemExit('Release uploads are CI-only. Obtain approval before pushing source or starting release workflows.')
    root=Path(__file__).resolve().parent.parent
    repo=os.environ['GITHUB_REPOSITORY']
    tag=os.environ['RELEASE_TAG']
    version=json.loads((root/'desktop/package.json').read_text())['version']
    if tag!='v'+version:
        raise SystemExit('Release tag does not match package version')
    if 'apple' in args.target:
        artifacts=list((root/'dist').glob('Photo-Dedup-*.dmg'))
    else:
        artifacts=list((root/'desktop/src-tauri/target'/args.target/'release/bundle/nsis').glob('*.exe'))
    if len(artifacts)!=1:
        raise SystemExit('Expected exactly one platform installer before upload')
    notes=Path(os.environ['RUNNER_TEMP'])/'release-notes.txt'
    notes.write_text('Local photo duplicate review. Download the installer for your platform. '
                     'This draft prerelease must pass signing and clean-machine review before publication. '
                     'No source photos are included. See README and docs/USER_GUIDE.md for usage.')
    for attempt in range(5):
        result=subprocess.run(['gh','release','view',tag,'--repo',repo,'--json','isDraft'],capture_output=True,text=True)
        if result.returncode==0:
            if not json.loads(result.stdout)['isDraft']:
                raise SystemExit('Refusing to modify an already published release')
            break
        created=subprocess.run(['gh','release','create',tag,'--repo',repo,'--target',os.environ['GITHUB_SHA'],
                                '--draft','--prerelease','--title',f'Photo Dedup {tag}',
                                '--notes-file',str(notes)],capture_output=True,text=True)
        if created.returncode==0:
            break
        if attempt==4:
            raise SystemExit('Could not create/find the draft release. Check workflow token permissions.')
        time.sleep(attempt+1)
    uploads=[]
    for artifact in artifacts:
        with artifact.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        checksum=artifact.with_suffix(artifact.suffix+'.sha256')
        checksum.write_text(f'{digest}  {artifact.name}\n')
        uploads.extend([str(artifact),str(checksum)])
    subprocess.run(['gh','release','upload',tag,*uploads,'--repo',repo,'--clobber'],check=True)
    print(f'Uploaded installer and checksum to draft release {repo} / {tag}')


if __name__=='__main__':
    main()
