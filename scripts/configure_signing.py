"""Prepare runner-only signing settings; credential values are never printed."""
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
signed = os.environ.get('SIGNED_RELEASE', '').lower() == 'true'
config = {'bundle': {}}


def command(args, **kwargs):
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise SystemExit('Platform certificate setup failed. Check protected release secrets and certificate permissions.')
    return result.stdout.strip()


if sys.platform == 'darwin':
    if signed:
        required = ['APPLE_CERTIFICATE', 'APPLE_CERTIFICATE_PASSWORD', 'APPLE_SIGNING_IDENTITY',
                    'APPLE_API_ISSUER', 'APPLE_API_KEY', 'APPLE_API_KEY_CONTENT']
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise SystemExit('Signed macOS release needs configured secrets: ' + ', '.join(missing))
        temp = Path(os.environ['RUNNER_TEMP'])
        keychain = temp / 'photo-dedup-signing.keychain-db'
        keychain_password = secrets.token_urlsafe(32)
        certificate = temp / 'photo-dedup-certificate.p12'
        certificate.write_bytes(base64.b64decode(os.environ['APPLE_CERTIFICATE']))
        certificate.chmod(0o600)
        command(['security','create-keychain','-p',keychain_password,str(keychain)])
        command(['security','set-keychain-settings','-lut','21600',str(keychain)])
        command(['security','unlock-keychain','-p',keychain_password,str(keychain)])
        try:
            command(['security','import',str(certificate),'-k',str(keychain),'-P',os.environ['APPLE_CERTIFICATE_PASSWORD'],
                     '-T','/usr/bin/codesign','-T','/usr/bin/security'])
        finally:
            certificate.unlink(missing_ok=True)
        command(['security','set-key-partition-list','-S','apple-tool:,apple:,codesign:', '-s','-k',keychain_password,str(keychain)])
        existing = command(['security','list-keychains','-d','user']).splitlines()
        command(['security','list-keychains','-d','user','-s',str(keychain),*[line.strip().strip('"') for line in existing]])
        key_path = temp / 'AuthKey.p8'
        key_path.write_text(os.environ['APPLE_API_KEY_CONTENT'])
        key_path.chmod(0o600)
        with Path(os.environ['GITHUB_ENV']).open('a') as target:
            target.write(f'APPLE_API_KEY_PATH={key_path}\n')
        config['bundle']['macOS'] = {'signingIdentity': os.environ['APPLE_SIGNING_IDENTITY']}
    else:
        config['bundle']['macOS'] = {'signingIdentity': '-'}
elif sys.platform == 'win32' and signed:
    required = ['WINDOWS_CERTIFICATE_BASE64','WINDOWS_CERTIFICATE_PASSWORD']
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit('Signed Windows release needs configured secrets: '+', '.join(missing))
    certificate = Path(os.environ['RUNNER_TEMP'])/'photo-dedup-certificate.pfx'
    certificate.write_bytes(base64.b64decode(os.environ['WINDOWS_CERTIFICATE_BASE64']))
    env = dict(os.environ, PHOTO_DEDUP_PFX_PATH=str(certificate))
    try:
        thumbprint = command(['pwsh','-NoProfile','-NonInteractive','-Command',
          "$password=ConvertTo-SecureString $env:WINDOWS_CERTIFICATE_PASSWORD -AsPlainText -Force; "
          "$cert=Import-PfxCertificate -FilePath $env:PHOTO_DEDUP_PFX_PATH -CertStoreLocation Cert:\\CurrentUser\\My -Password $password; "
          "$cert.Thumbprint"], env=env).splitlines()[-1]
    finally:
        certificate.unlink(missing_ok=True)
    config['bundle']['windows'] = {'certificateThumbprint':thumbprint,'digestAlgorithm':'sha256',
                                  'timestampUrl':'http://timestamp.digicert.com'}
(root/'desktop/release-config.json').write_text(json.dumps(config))
print('Configured signed release' if signed else 'Configured local-test / draft signing only')
