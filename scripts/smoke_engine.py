"""Prove that the frozen engine works from an unrelated directory without Python on PATH."""
import json
import argparse
import os
from pathlib import Path
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
        demo = call('demo')
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
        print('Bundled engine: metadata tool, scan, matching, SQLite sessions, and image preview passed.')
    finally:
        worker.stdin.close()
        worker.wait(timeout=70)
