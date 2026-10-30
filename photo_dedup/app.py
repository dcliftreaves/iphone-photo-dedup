"""Persistent application service shared by desktop, browser preview, and CLI."""
from collections import Counter
from concurrent.futures import CancelledError
from datetime import datetime, timezone
import base64
import io
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import uuid

from PIL import Image, ImageDraw, ImageOps

from . import __version__
from .export import export_manifest, plan_export, cleanup_records
from .media import exiftool_path, normalized_image, open_normalized
from .layout import DEFAULT_TEMPLATE
from .volumes import local_volumes, test_folder
from .review import apply_decisions, record_id
from .scanner import scan


def now():
    return datetime.now(timezone.utc).isoformat()


class AppService:
    METHODS = {'health', 'sessions', 'session', 'start_scan', 'restart_scan', 'job',
               'cancel', 'groups', 'group', 'preview', 'decide', 'undo', 'export_plan',
               'start_export', 'demo', 'issues', 'prepare_sources', 'local_volumes', 'test_folder', 'cleanup_review'}

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_lock = (self.data_dir / 'application.lock').open('a+b')
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                self.file_lock.seek(0)
                self.file_lock.write(b'0')
                self.file_lock.flush()
                self.file_lock.seek(0)
                msvcrt.locking(self.file_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file_lock.close()
            raise ValueError('This workspace is already open in another Photo Dedup instance')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.data_dir / 'sessions.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS cleanup_snapshots (session TEXT PRIMARY KEY, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS records (session TEXT, id TEXT, group_id TEXT, status TEXT,
            data TEXT NOT NULL, PRIMARY KEY(session,id));
          CREATE INDEX IF NOT EXISTS record_groups ON records(session,group_id);
          CREATE TABLE IF NOT EXISTS decisions (session TEXT, group_id TEXT, data TEXT,
            PRIMARY KEY(session,group_id));
          CREATE TABLE IF NOT EXISTS history (seq INTEGER PRIMARY KEY AUTOINCREMENT,
            session TEXT, group_id TEXT, previous TEXT);
        ''')
        self.db.commit()
        self.jobs = {}
        self.threads = []
        # A process can exit during a job. Make that state explicit on the next launch.
        for session in self.sessions():
            if session['state'] in ('scanning', 'exporting'):
                pending = session.pop('pending_export', None)
                recovered = False
                if pending:
                    staging = Path(pending['staging'])
                    destination = Path(pending['destination'])
                    marker = staging / '.photo-dedup-staging.json'
                    # Only remove a marked, owned staging directory from a saved
                    # export. Never infer ownership from a generic temporary name.
                    if (not staging.is_symlink() and staging.parent == destination.parent and
                            staging.name.startswith('.' + destination.name + '-') and marker.is_file()):
                        try:
                            identity = json.loads(marker.read_text())
                            if identity == dict(schema_version=1, destination=str(destination.resolve())):
                                shutil.rmtree(staging)
                                recovered = True
                        except (OSError, ValueError):
                            pass
                session.update(state='ready' if session.get('manifest') else 'interrupted',
                               error=('An interrupted partial export was removed. Source files were not changed.' if recovered else
                                      'The application closed during processing. Source files were not changed.'))
                self._save_session(session)

    def call(self, method, params=None):
        if method not in self.METHODS:
            raise ValueError('Unknown application method')
        return getattr(self, method)(**(params or {}))

    def _save_session(self, session):
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO sessions VALUES (?,?)',
                            (session['id'], json.dumps(session)))

    def sessions(self):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM sessions ORDER BY rowid DESC')]

    def session(self, session_id):
        with self.lock:
            row = self.db.execute('SELECT data FROM sessions WHERE id=?', (session_id,)).fetchone()
        if not row:
            raise ValueError('This scan session was not found')
        return json.loads(row[0])

    def health(self):
        return dict(protocol_version=1, version=__version__, exiftool=bool(exiftool_path()),
                    data_dir=str(self.data_dir), source_policy='read-only')

    def prepare_sources(self, roots):
        """Inspect source locations without writing probes or scanning their files."""
        if not isinstance(roots, list) or not roots:
            raise ValueError('Choose at least one import folder')
        paths = sorted({Path(p).expanduser().resolve() for p in roots})
        devices = set()
        for path in paths:
            if not path.is_dir():
                raise ValueError(f'Folder not found: {path}')
            devices.add(path.stat().st_dev)
        return dict(same_filesystem=len(devices) == 1, folders=len(paths),
                    hardlink_support_verified=False)

    def local_volumes(self):
        return local_volumes()

    def test_folder(self, path):
        return test_folder(path)

    def _active(self):
        return any(j['state'] == 'running' for j in self.jobs.values())

    def _new_job(self, session_id, kind):
        with self.lock:
            if self._active():
                raise ValueError('Another job is running. Finish or cancel it first.')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = dict(id=job_id, session_id=session_id, kind=kind, state='running',
                                     done=0, total=0, phase='Preparing', cached=0, event=threading.Event())
            return job_id

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError('Job is no longer active. Reopen its saved session.')
            return {k: v for k, v in self.jobs[job_id].items() if k != 'event'}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise ValueError('Job not found')
            job['event'].set()
            job['phase'] = 'Stopping safely after current files'
        return self.job(job_id)

    def _launch(self, job_id, task):
        def work():
            job = self.jobs[job_id]
            try:
                result = task(job)
                with self.lock:
                    job.update(state='complete', result=result, phase='Complete')
            except CancelledError as error:
                with self.lock:
                    job.update(state='cancelled', error=str(error))
                session = self.session(job['session_id'])
                session.pop('pending_export', None)
                session.update(state='ready' if job['kind'] == 'export' else 'cancelled', error=str(error))
                self._save_session(session)
            except Exception as error:
                with self.lock:
                    job.update(state='error', error=str(error))
                session = self.session(job['session_id'])
                session.pop('pending_export', None)
                session.update(state='ready' if job['kind'] == 'export' else 'error', error=str(error))
                self._save_session(session)
        thread = threading.Thread(target=work, name=f'dedup-{job_id}', daemon=True)
        self.threads.append(thread)
        thread.start()

    def start_scan(self, roots, name=None, workers=4, threshold=.02, full_images=False, deep_scan=False):
        if not isinstance(roots, list) or not roots:
            raise ValueError('Choose at least one folder')
        paths = sorted({Path(p).expanduser().resolve() for p in roots})
        for path in paths:
            if not path.is_dir():
                raise ValueError(f'Folder not found: {path}')
            if path.suffix.lower() == '.photoslibrary':
                raise ValueError('Export originals from Apple Photos first; Photos libraries are not scanned directly.')
            if path == self.data_dir or (self.data_dir in path.parents and path != self.data_dir / 'demo-photos'):
                raise ValueError('Application data cannot be used as a source folder')
        if not isinstance(workers, int) or not 1 <= workers <= 32 or not 0 <= threshold <= 1:
            raise ValueError('Invalid scan settings')
        session_id = uuid.uuid4().hex
        job_id = self._new_job(session_id, 'scan')
        session = dict(id=session_id, name=(name or paths[0].name)[:120], created_at=now(),
                       roots=[str(p) for p in paths], state='scanning', job_id=job_id,
                       settings=dict(workers=workers, threshold=threshold, full_images=bool(full_images), deep_scan=bool(deep_scan)),
                       counts={}, groups=0, reviewed=0)
        self._save_session(session)
        self._launch_scan(job_id, session)
        return dict(session_id=session_id, job_id=job_id)

    def restart_scan(self, session_id):
        session = self.session(session_id)
        return self.start_scan(session['roots'], name=session['name'], **session['settings'])

    def _launch_scan(self, job_id, session):
        def task(job):
            def progress(done, total, hits):
                with self.lock:
                    job.update(done=done, total=total, cached=hits,
                               phase='Comparing matches' if done == total else 'Reading files')
            manifest = scan([Path(p) for p in session['roots']], cache_path=self.data_dir / 'inspection.sqlite3',
                            excludes=[] if session['roots'] == [str(self.data_dir / 'demo-photos')] else [self.data_dir],
                            progress=progress, cancel_event=job['event'], **session['settings'])
            if job['event'].is_set():
                raise CancelledError('Scan cancelled. Restart to reuse cached inspections.')
            group_counts = Counter(r.get('group_id') for r in manifest['records'] if r.get('group_id'))
            with self.lock, self.db:
                for record in manifest['records']:
                    record['id'] = record_id(record)
                    self.db.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                                    (session['id'], record['id'], record.get('group_id'), record['status'], json.dumps(record)))
            folder = self.data_dir / 'scans' / session['id']
            folder.mkdir(parents=True, exist_ok=True)
            manifest_file = folder / 'manifest.json'
            manifest_file.write_text(json.dumps(manifest, indent=2))
            session.update(state='ready', manifest=str(manifest_file),
                           counts=dict(Counter(r['status'] for r in manifest['records'])),
                           groups=sum(count > 1 for count in group_counts.values()),
                           warnings=sum(bool(r.get('warnings')) for r in manifest['records']),
                           cache_hits=manifest['cache_hits'])
            self._save_session(session)
            return session
        self._launch(job_id, task)

    def _manifest(self, session_id):
        session = self.session(session_id)
        if not session.get('manifest'):
            raise ValueError('This scan has no completed results yet')
        return json.loads(Path(session['manifest']).read_text())

    def _decisions(self, session_id):
        with self.lock:
            return {group: json.loads(data) for group, data in self.db.execute(
                'SELECT group_id,data FROM decisions WHERE session=?', (session_id,))}

    def _records(self, session_id, group_id=None):
        self.session(session_id)
        query = 'SELECT data FROM records WHERE session=?'
        args = [session_id]
        if group_id is not None:
            query += ' AND group_id=?'
            args.append(group_id)
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute(query, args)]

    def groups(self, session_id, filter='all', query='', offset=0, limit=60):
        if filter not in {'all', 'exact', 'visual', 'metadata', 'unreviewed', 'reviewed', 'unique'}:
            raise ValueError('Unknown results filter')
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError('Invalid result page')
        decisions = self._decisions(session_id)
        grouped = {}
        for record in self._records(session_id):
            if record.get('group_id') and record['status'] != 'sidecar':
                grouped.setdefault(record['group_id'], []).append(record)
        results = []
        for group, records in grouped.items():
            unique = len(records) == 1
            if unique != (filter == 'unique'):
                continue
            kind = ('metadata' if any(r.get('confidence') == 'metadata_only' for r in records) else
                    'visual' if any(r.get('confidence') in {'visual_similarity', 'raw_samples', 'raw_render_similarity', 'deep_visual_similarity'} for r in records) else 'exact')
            decision = decisions.get(group)
            if filter in {'exact', 'visual', 'metadata'} and kind != filter:
                continue
            if filter == 'reviewed' and not decision or filter == 'unreviewed' and decision:
                continue
            if query and not any(query.casefold() in r['path'].casefold() for r in records):
                continue
            first = next((r for r in records if r['status'] == 'keep'), records[0])
            results.append(dict(id=group, kind='unique' if unique else kind, count=len(records),
                                title=Path(first['path']).name, preview_id=first['id'],
                                size=sum(r['size'] for r in records), decision=decision,
                                has_image=first.get('kind') in {'image', 'raw'}, media_kind=first.get('kind')))
        results.sort(key=lambda g: (-g['size'], g['title'], g['id']))
        return dict(items=results[offset:offset + limit], total=len(results), offset=offset)

    def group(self, session_id, group_id):
        records = self._records(session_id, group_id)
        if not records:
            raise ValueError('Group not found')
        decisions = self._decisions(session_id)
        records.sort(key=lambda r: (r['status'] != 'keep', r['path']))
        decision = decisions.get(group_id)
        selected = decision['selected'] if decision else [r['id'] for r in records if r['status'] != 'exact_duplicate']
        return dict(id=group_id, records=records, decision=decision, selected=selected,
                    recommendation=next((r['id'] for r in records if r['status'] == 'keep'), records[0]['id']))

    def _record(self, session_id, file_id):
        with self.lock:
            row = self.db.execute('SELECT data FROM records WHERE session=? AND id=?', (session_id, file_id)).fetchone()
        if not row:
            raise ValueError('File is not part of this scan')
        return json.loads(row[0])

    def preview(self, session_id, file_id, size=640):
        if size not in (160, 640, 1600):
            raise ValueError('Unsupported preview size')
        record = self._record(session_id, file_id)
        if record.get('kind') not in {'image', 'raw'}:
            return dict(image=None, message='Preview is not available for this format. Original files remain intact.')
        source = Path(record['path'])
        if source.is_symlink() or not source.is_file() or source.stat().st_size != record['size'] or source.stat().st_mtime_ns != record['mtime_ns']:
            raise ValueError('File changed since scanning; scan again before reviewing it')
        folder = self.data_dir / 'previews' / session_id
        folder.mkdir(parents=True, exist_ok=True)
        cached = folder / f'{file_id}-{size}.png'
        if not cached.exists():
            result = open_normalized(source, half_size=size <= 640, raw=record.get("kind") == "raw")
            result.thumbnail((size, size), Image.Resampling.LANCZOS)
            result.save(cached, format='PNG')
        return dict(image='data:image/png;base64,' + base64.b64encode(cached.read_bytes()).decode(), message=None)

    def decide(self, session_id, group_id, action, selected=None):
        if self.session(session_id)['state'] != 'ready':
            raise ValueError('Wait for the current operation before changing review choices')
        group = self.group(session_id, group_id)
        if action not in {'keep_selected', 'keep_all', 'not_duplicates', 'reset'}:
            raise ValueError('Unknown review action')
        if action in {'keep_all', 'not_duplicates'}:
            selected = [r['id'] for r in group['records']]
        if action != 'reset':
            allowed = {r['id'] for r in group['records']}
            if not isinstance(selected, list) or not selected or not set(selected) <= allowed:
                raise ValueError('Keep at least one existing file')
        previous = group['decision']
        with self.lock, self.db:
            self.db.execute('INSERT INTO history(session,group_id,previous) VALUES (?,?,?)',
                            (session_id, group_id, json.dumps(previous) if previous else None))
            if action == 'reset':
                self.db.execute('DELETE FROM decisions WHERE session=? AND group_id=?', (session_id, group_id))
            else:
                decision = dict(action=action, selected=sorted(set(selected)), updated_at=now())
                self.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?)',
                                (session_id, group_id, json.dumps(decision)))
        self._update_review_count(session_id)
        return self.group(session_id, group_id)

    def _update_review_count(self, session_id):
        session = self.session(session_id)
        session['reviewed'] = len(self._decisions(session_id))
        self._save_session(session)

    def undo(self, session_id):
        if self.session(session_id)['state'] != 'ready':
            raise ValueError('Wait for the current operation before undoing choices')
        with self.lock, self.db:
            row = self.db.execute('SELECT seq,group_id,previous FROM history WHERE session=? ORDER BY seq DESC LIMIT 1',
                                  (session_id,)).fetchone()
            if not row:
                return dict(undone=False)
            seq, group, previous = row
            if previous:
                self.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?)', (session_id, group, previous))
            else:
                self.db.execute('DELETE FROM decisions WHERE session=? AND group_id=?', (session_id, group))
            self.db.execute('DELETE FROM history WHERE seq=?', (seq,))
        self._update_review_count(session_id)
        return dict(undone=True, group_id=group)

    def issues(self, session_id, offset=0, limit=100):
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError('Invalid result page')
        records = [r for r in self._records(session_id)
                   if r['status'] in {'ignored', 'error'} or r.get('warnings')]
        return dict(items=records[offset:offset + limit], total=len(records))

    def export_plan(self, session_id, destination=None, mode='hardlink', template=DEFAULT_TEMPLATE):
        if mode not in {'copy', 'hardlink'}:
            raise ValueError('Unknown export mode')
        manifest = apply_decisions(self._manifest(session_id), self._decisions(session_id))
        manifest["layout"] = dict(template=template)
        plan = plan_export(manifest)
        total = sum(r['size'] for r in manifest['records'] if r['path'] in plan)
        session = self.session(session_id)
        result = dict(files=len(plan), bytes=total, omitted=sum(r['status'] not in {'ignored', 'error'} and
                      not r['export_selected'] for r in manifest['records']), unresolved=max(0, session['groups'] - session['reviewed']),
                      mode=mode, destination=destination, records=[dict(source=k, destination=str(v)) for k, v in plan.items()])
        result['ignored'] = sum(r['status'] == 'ignored' for r in manifest['records'])
        result['errors'] = sum(r['status'] == 'error' for r in manifest['records'])
        result['warnings'] = sum(bool(r.get('warnings')) for r in manifest['records'])
        if destination:
            path = Path(destination).expanduser().resolve()
            if path.exists():
                raise ValueError('Choose a new export folder; this destination already exists')
            for root in manifest['roots']:
                source = Path(root)
                if path == source or path in source.parents or source in path.parents:
                    raise ValueError('Choose an export destination outside the source folders')
            if path == self.data_dir or self.data_dir in path.parents or path in self.data_dir.parents:
                raise ValueError('Choose an export destination outside application data')
            parent = path.parent
            while not parent.exists():
                parent = parent.parent
            if mode == 'hardlink':
                for record in manifest['records']:
                    if record['path'] in plan and record['device'] != parent.stat().st_dev:
                        raise ValueError('Hard links require all retained imports and the output on one filesystem. '
                                         'Gather the imports on the output volume, or explicitly choose independent copies.')
            free = shutil.disk_usage(parent).free
            result['free_bytes'] = free
            if mode == 'copy' and total > free:
                raise ValueError('There is not enough free space for independent copies at this destination')
        return result

    def start_export(self, session_id, destination, mode='hardlink', template=DEFAULT_TEMPLATE):
        approved_plan = self.export_plan(session_id, destination, mode, template)
        manifest = apply_decisions(self._manifest(session_id), self._decisions(session_id))
        manifest['layout'] = dict(template=template)
        job_id = self._new_job(session_id, 'export')
        session = self.session(session_id)
        session.update(state='exporting', job_id=job_id)
        self._save_session(session)
        def task(job):
            plan = plan_export(manifest)
            job.update(total=len(plan), phase='Verifying and exporting')
            def check_cancel():
                if job['event'].is_set():
                    raise CancelledError('Export cancelled. The partial export was removed; originals are unchanged.')
            def progress(path):
                with self.lock:
                    job['done'] += 1
                check_cancel()
            def staging_ready(path):
                session['pending_export'] = dict(staging=str(path), destination=str(Path(destination).expanduser().absolute()))
                self._save_session(session)
            export_manifest(manifest, Path(destination).expanduser(), mode=mode,
                            progress=progress, check_cancel=check_cancel, staging_ready=staging_ready)
            with self.lock, self.db:
                self.db.execute('INSERT OR REPLACE INTO cleanup_snapshots VALUES (?,?)',
                                (session_id, json.dumps(self._cleanup_records(manifest, plan))))
            session.update(state='ready', last_export=str(Path(destination).expanduser().resolve()),
                           last_export_mode=mode, last_export_template=template)
            session.pop('error', None)
            session.pop('pending_export', None)
            self._save_session(session)
            return dict(destination=session['last_export'], files=len(plan), mode=mode,
                        omitted=approved_plan['omitted'], ignored=approved_plan['ignored'],
                        errors=approved_plan['errors'], warnings=approved_plan['warnings'])
        self._launch(job_id, task)
        return dict(job_id=job_id, session_id=session_id)

    _cleanup_records = staticmethod(cleanup_records)

    def cleanup_review(self, session_id, offset=0, limit=100):
        session = self.session(session_id)
        if not session.get('last_export'):
            raise ValueError('Complete an export before reviewing source cleanup')
        with self.lock:
            row = self.db.execute('SELECT data FROM cleanup_snapshots WHERE session=?', (session_id,)).fetchone()
        records = json.loads(row[0]) if row else None
        if records is None:
            raise ValueError('This older export has no saved cleanup evidence. Export again before using cleanup review.')
        omitted = [r for r in records if r['disposition'] == 'not_retained']
        roots = []
        for root in session['roots']:
            members = [r for r in records if Path(r['path']) == Path(root) or Path(root) in Path(r['path']).parents]
            roots.append(dict(path=root, files=len(members), at_risk=sum(r['disposition'] == 'not_retained' for r in members)))
        offset, limit = max(0, int(offset)), min(100, max(1, int(limit)))
        return dict(roots=roots, items=omitted[offset:offset + limit], total=len(omitted),
                    retained=sum(r['disposition'] == 'retained' for r in records),
                    identical_retained=sum(r['disposition'] == 'identical_retained' for r in records),
                    message='Saved export evidence, not a fresh disk audit. Files added or changed after the scan are not covered. '
                            'Verify the library and backup before deleting anything. Ignored directories may contain uncounted files.')

    def demo(self):
        folder = self.data_dir / 'demo-photos'
        if not folder.exists():
            folder.mkdir()
            # Synthetic landscapes are demo fixtures, never personal media.
            for index, (sky, water) in enumerate([((193, 217, 218), (60, 101, 102)),
                                                ((234, 209, 174), (171, 111, 75)),
                                                ((206, 220, 201), (91, 115, 83))], start=1):
                im = Image.new('RGB', (960, 640), sky)
                draw = ImageDraw.Draw(im)
                draw.ellipse((660, 75, 760, 175), fill=(249, 237, 205))
                draw.polygon([(0, 390), (280, 140), (470, 350), (660, 220), (960, 390)], fill=(82, 105, 103))
                draw.polygon([(0, 420), (190, 300), (420, 440), (780, 290), (960, 400), (960, 640), (0, 640)], fill=water)
                draw.polygon([(0, 525), (400, 460), (960, 520), (960, 640), (0, 640)], fill=tuple(min(255, c + 20) for c in water))
                for sub in ('Camera originals', 'Previous imports'):
                    (folder / sub).mkdir(exist_ok=True)
                original = folder / 'Camera originals' / f'IMG_{index:04}.png'
                im.save(original)
                if index == 1:
                    shutil.copy2(original, folder / 'Previous imports' / 'IMG_0001 copy.png')
                else:
                    im.save(folder / 'Previous imports' / f'IMG_{index:04}.jpg', quality=95)
            (folder / 'readme.txt').write_text('Synthetic demo images. Your own files have not been accessed.\n')
        return dict(roots=[str(folder)], name='Mixed import history · demo')

    def close(self):
        for job in self.jobs.values():
            job['event'].set()
        for thread in self.threads:
            thread.join(timeout=65)
        with self.lock:
            self.db.close()
        self.file_lock.close()
