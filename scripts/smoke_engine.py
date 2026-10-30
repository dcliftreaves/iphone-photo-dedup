"""Prove that the frozen engine works from an unrelated directory without Python on PATH."""
import json
import argparse
import os
from pathlib import Path
from photo_dedup import __version__
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path)
args = parser.parse_args()
binary = args.binary or root/'desktop/src-tauri/engine'/('photo-dedup-engine.exe' if os.name == 'nt' else 'photo-dedup-engine')
binary = binary.resolve()
validation = root/'local-validation'
validation.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='packaged-smoke-', dir=validation) as scratch:
    env = dict(os.environ)
    env['PATH'] = os.environ.get('SystemRoot', r'C:\Windows') + r'\System32' if os.name == 'nt' else '/usr/bin:/bin'
    worker = subprocess.Popen([str(binary), '--data-dir', str(Path(scratch)/'data')], cwd=scratch,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    count = 0
    def call(method, params=None):
        global count
        count += 1
        worker.stdin.write(json.dumps(dict(version=1, id=count, method=method, params=params or {}))+'\n')
        worker.stdin.flush()
        response = json.loads(worker.stdout.readline())
        assert not response.get('error'), response
        return response['result']
    try:
        health = call('health')
        assert health['exiftool'], health
        assert health['version'] == __version__, health
        demo = call('demo')
        preparation = call('prepare_sources', {'roots':demo['roots']})
        assert preparation['same_filesystem'] is True, preparation
        assert preparation['hardlink_support_verified'] is False, preparation
        job = call('start_scan', demo)
        end = time.monotonic() + 60
        while time.monotonic() < end:
            status = call('job', {'job_id':job['job_id']})
            if status['state'] != 'running':
                break
            time.sleep(.1)
        assert status['state'] == 'complete', status
        groups = call('groups', {'session_id':job['session_id']})
        assert groups['total'] == 3, groups
        record = call('group', {'session_id':job['session_id'], 'group_id':groups['items'][0]['id']})['records'][0]
        assert call('preview', {'session_id':job['session_id'],'file_id':record['id']})['image']
        plan = call('export_plan', {'session_id':job['session_id']})
        assert plan['mode'] == 'hardlink' and plan['ignored'] == 1, plan
        from synthetic_dng import make_dng
        raw_folder = Path(scratch) / 'raw-inputs'
        raw_folder.mkdir()
        make_dng(raw_folder / 'original.dng')
        make_dng(raw_folder / 'rewritten.dng', 'different metadata')
        raw_job = call('start_scan', dict(roots=[str(raw_folder)], deep_scan=True))
        end = time.monotonic() + 60
        while time.monotonic() < end:
            status = call('job', dict(job_id=raw_job['job_id']))
            if status['state'] != 'running':
                break
            time.sleep(.1)
        assert status['state'] == 'complete', status
        raw_session = raw_job['session_id']
        raw_groups = call('groups', dict(session_id=raw_session))
        assert raw_groups['total'] == 1, raw_groups
        raw_group = call('group', dict(session_id=raw_session, group_id=raw_groups['items'][0]['id']))
        assert any(r['confidence'] == 'raw_samples' for r in raw_group['records']), raw_group
        assert call('preview', dict(session_id=raw_session, file_id=raw_group['recommendation']))['image']
        call('decide', dict(session_id=raw_session, group_id=raw_group['id'], action='keep_selected', selected=[raw_group['recommendation']]))
        output = Path(scratch) / 'canonical'
        exported = call('start_export', dict(session_id=raw_session, destination=str(output), template='{kind}/{year}/{stem}{ext}'))
        end = time.monotonic() + 60
        while time.monotonic() < end:
            status = call('job', dict(job_id=exported['job_id']))
            if status['state'] != 'running':
                break
            time.sleep(.1)
        assert status['state'] == 'complete', status
        assert call('cleanup_review', dict(session_id=raw_session))['total'] == 1
        assert (output / 'source-cleanup.csv').is_file()
        assert list((output / 'curated' / 'raw' / 'Unknown-year').glob('*.dng'))
        print('Bundled engine: metadata, image/RAW decoding, sample evidence, custom hard-link export, and saved cleanup review passed.')
    finally:
        worker.stdin.close()
        worker.wait(timeout=70)
