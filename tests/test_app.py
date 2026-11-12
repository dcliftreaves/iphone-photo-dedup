from concurrent.futures import CancelledError
import copy
import json
from pathlib import Path
import shutil
import time

from PIL import Image
import pytest

from photo_dedup.app import AppService
from photo_dedup.export import export_manifest, plan_export
from photo_dedup.review import apply_decisions, record_id
from photo_dedup.scanner import scan


@pytest.fixture(autouse=True)
def no_exiftool(monkeypatch):
    monkeypatch.setattr('photo_dedup.scanner.exiftool_path', lambda: None)


@pytest.fixture
def app(tmp_path):
    service = AppService(tmp_path / 'app')
    yield service
    service.close()


def wait(app, job_id):
    end = time.monotonic() + 20
    while time.monotonic() < end:
        result = app.job(job_id)
        if result['state'] != 'running':
            return result
        time.sleep(.01)
    raise AssertionError('Job timed out')


def ready_demo(app):
    demo = app.demo()
    result = app.start_scan(**demo)
    job = wait(app, result['job_id'])
    assert job['state'] == 'complete', job
    return result['session_id']


def test_demo_scan_groups_and_paged_results(app):
    session = ready_demo(app)
    saved = app.session(session)
    assert saved['state'] == 'ready'
    assert saved['groups'] == 3
    assert saved['counts']['exact_duplicate'] == 1
    assert saved['counts']['review'] == 2
    groups = app.groups(session, limit=1)
    assert groups['total'] == 3 and len(groups['items']) == 1
    assert app.groups(session, filter='exact')['total'] == 1
    assert app.groups(session, filter='visual')['total'] == 2
    assert len(app.issues(session)['items']) == 1
    for summary in app.groups(session)['items']:
        group = app.group(session, summary['id'])
        assert len(group['records']) == 2
        assert app.preview(session, group['records'][0]['id'])['image'].startswith('data:image/png;base64,')
    with pytest.raises(ValueError, match='part of this scan'):
        app.preview(session, 'unknown')


def test_review_changes_export_plan_and_undo_restores_it(app, tmp_path):
    session = ready_demo(app)
    visual = app.groups(session, filter='visual')['items'][0]
    group = app.group(session, visual['id'])
    initial = app.export_plan(session)
    assert initial['files'] == 5 and initial['omitted'] == 1
    app.decide(session, visual['id'], 'keep_selected', [group['recommendation']])
    changed = app.export_plan(session)
    assert changed['files'] == 4 and changed['omitted'] == 2
    assert app.session(session)['reviewed'] == 1
    assert app.undo(session)['undone']
    assert app.export_plan(session)['files'] == 5
    assert not app.undo(session)['undone']
    app.decide(session, visual['id'], 'not_duplicates')
    assert app.group(session, visual['id'])['decision']['action'] == 'not_duplicates'
    app.decide(session, visual['id'], 'reset')
    assert app.group(session, visual['id'])['decision'] is None
    with pytest.raises(ValueError, match='at least one'):
        app.decide(session, visual['id'], 'keep_selected', [])
    with pytest.raises(ValueError, match='at least one'):
        app.decide(session, visual['id'], 'keep_selected', ['outside-the-group'])


def test_review_is_persistent_and_workspace_has_one_owner(tmp_path):
    path = tmp_path / 'app'
    app = AppService(path)
    session = ready_demo(app)
    group = app.groups(session)['items'][0]
    app.decide(session, group['id'], 'keep_all')
    with pytest.raises(ValueError, match='already open'):
        AppService(path)
    app.close()
    restored = AppService(path)
    try:
        assert restored.session(session)['reviewed'] == 1
        assert restored.group(session, group['id'])['decision']['action'] == 'keep_all'
        assert restored.undo(session)['undone']
    finally:
        restored.close()


def test_export_uses_choices_and_retains_complete_audit(app, tmp_path):
    session = ready_demo(app)
    groups = app.groups(session)['items']
    for summary in groups:
        group = app.group(session, summary['id'])
        app.decide(session, summary['id'], 'keep_selected', [group['recommendation']])
    destination = tmp_path / 'organized'
    plan = app.export_plan(session, str(destination))
    assert plan['files'] == 3 and plan['unresolved'] == 0
    job = app.start_export(session, str(destination))
    result = wait(app, job['job_id'])
    assert result['state'] == 'complete', result
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert len(manifest['records']) == 7
    assert sum('export_path' in r for r in manifest['records']) == 3
    assert app.session(session)['last_export'] == str(destination)
    with pytest.raises(ValueError, match='already exists'):
        app.export_plan(session, str(destination))
    with pytest.raises(ValueError, match='outside the source'):
        app.export_plan(session, str(Path(app.demo()['roots'][0]) / 'nested'))


def test_source_change_between_review_and_export_blocks_export(app, tmp_path):
    session = ready_demo(app)
    group = app.group(session, app.groups(session)['items'][0]['id'])
    record = group['records'][0]
    Image.new('RGB', (100, 100), 'purple').save(record['path'])
    with pytest.raises(ValueError, match='changed since scanning'):
        app.preview(session, record['id'])
    job = app.start_export(session, str(tmp_path / 'out'))
    result = wait(app, job['job_id'])
    assert result['state'] == 'error'
    assert 'changed since scan' in result['error']
    assert not (tmp_path / 'out').exists()
    assert app.session(session)['state'] == 'ready'


def test_unknown_rpc_and_invalid_requests_are_rejected(app):
    with pytest.raises(ValueError, match='Unknown application method'):
        app.call('_manifest', {})
    with pytest.raises(ValueError, match='Choose at least one'):
        app.start_scan([])
    with pytest.raises(ValueError, match='Application data'):
        app.start_scan([str(app.data_dir)])
    session = ready_demo(app)
    with pytest.raises(ValueError, match='Unknown results filter'):
        app.groups(session, filter='invalid')
    with pytest.raises(ValueError, match='Invalid result page'):
        app.groups(session, limit=-1)


def test_cancelled_scan_can_restart_using_cache(app, monkeypatch):
    from photo_dedup import app as module
    actual_scan = module.scan
    def cancellable(*args, **kwargs):
        while not kwargs['cancel_event'].wait(.01):
            pass
        raise CancelledError('Cancelled safely')
    monkeypatch.setattr(module, 'scan', cancellable)
    demo = app.demo()
    started = app.start_scan(**demo)
    app.cancel(started['job_id'])
    assert wait(app, started['job_id'])['state'] == 'cancelled'
    assert app.session(started['session_id'])['state'] == 'cancelled'
    monkeypatch.setattr(module, 'scan', actual_scan)
    restart = app.restart_scan(started['session_id'])
    assert wait(app, restart['job_id'])['state'] == 'complete'


def test_sidecar_is_preserved_when_its_local_image_is_an_omitted_copy(tmp_path):
    root = tmp_path / 'inputs'
    (root/'a').mkdir(parents=True)
    (root/'b').mkdir()
    Image.new('RGB', (40, 40), 'red').save(root/'a'/'one.png')
    shutil.copy2(root/'a'/'one.png', root/'b'/'one.png')
    (root/'b'/'one.aae').write_text('<plist/>')
    manifest = apply_decisions(scan([root]), {})
    plan = plan_export(manifest)
    assert len(plan) == 2
    sidecar = str(root/'b'/'one.aae')
    assert plan[sidecar].with_suffix('.png') == plan[str(root/'a'/'one.png')]


def test_nested_photos_libraries_are_not_entered(tmp_path):
    library = tmp_path/'Photos.photoslibrary'
    library.mkdir()
    Image.new('RGB', (40,40), 'red').save(library/'original.png')
    result = scan([tmp_path])
    assert len(result['records']) == 1
    assert 'Apple Photos library' in result['records'][0]['reason']


def test_selected_alternative_is_in_curated_output_not_duplicates(app):
    session = ready_demo(app)
    summary = app.groups(session, filter='exact')['items'][0]
    group = app.group(session, summary['id'])
    alternate = next(r for r in group['records'] if r['status'] == 'exact_duplicate')
    app.decide(session, summary['id'], 'keep_selected', [alternate['id']])
    plan = app.export_plan(session)
    output = next(r for r in plan['records'] if r['source'] == alternate['path'])
    assert Path(output['destination']).parts[0] == 'curated'
    assert all(r['source'] != next(r['path'] for r in group['records'] if r['status'] == 'keep') for r in plan['records'])


def test_live_photo_and_raw_companions_cannot_be_silently_dropped():
    from photo_dedup.review import record_id
    def record(path, kind, status, group, related):
        return dict(path=path, kind=kind, status=status, group_id=group,
                    related_files=[dict(path=related,relationship='live_photo')])
    image = record('/photos/a.heic', 'image', 'review', 'images', '/photos/a.mov')
    original = dict(path='/other/b.heic', kind='image', status='keep', group_id='images', related_files=[])
    video = record('/photos/a.mov', 'video', 'keep', 'videos', '/photos/a.heic')
    manifest = dict(records=[image, original, video])
    result = apply_decisions(manifest, {'images':dict(action='keep_selected',selected=[record_id(original)])})
    selected = {r['path']:r['export_selected'] for r in result['records']}
    assert selected['/photos/a.heic'] and selected['/photos/a.mov']
    assert result['records'][0]['preserved_as_companion']


def test_crashed_export_only_removes_a_marked_owned_staging_directory(tmp_path):
    app = AppService(tmp_path/'app')
    session_id = ready_demo(app)
    session = app.session(session_id)
    destination = tmp_path/'Library'
    staging = tmp_path/'.Library-owned123'
    staging.mkdir()
    (staging/'.photo-dedup-staging.json').write_text(json.dumps(dict(schema_version=1,destination=str(destination))))
    (staging/'partial.png').write_bytes(b'partial generated output')
    unrelated = tmp_path/'.Library-unrelated'
    unrelated.mkdir()
    (unrelated/'keep.txt').write_text('not owned by this export')
    session.update(state='exporting',pending_export=dict(staging=str(staging),destination=str(destination)))
    app._save_session(session)
    app.close()
    restored=AppService(tmp_path/'app')
    try:
        assert not staging.exists()
        assert (unrelated/'keep.txt').exists()
        assert restored.session(session_id)['state']=='ready'
        assert 'partial export was removed' in restored.session(session_id)['error']
    finally:
        restored.close()


def test_source_preparation_checks_locations_without_writing_to_imports(app, tmp_path, monkeypatch):
    root=tmp_path/'imports'
    first=root/'Mac downloads'
    second=root/'Windows downloads'
    first.mkdir(parents=True)
    second.mkdir()
    before=list(root.rglob('*'))
    result=app.prepare_sources([str(first),str(second)])
    assert result['same_filesystem'] is True
    assert result['hardlink_support_verified'] is False
    assert list(root.rglob('*'))==before
    import os
    real_stat=Path.stat
    def other_device(path, *args, **kwargs):
        stat=real_stat(path,*args,**kwargs)
        if path==second:
            values=list(stat)
            values[2]=stat.st_dev+1
            return os.stat_result(values)
        return stat
    monkeypatch.setattr(Path,'stat',other_device)
    assert app.prepare_sources([str(first),str(second)])['same_filesystem'] is False


def test_same_volume_workflow_defaults_to_links_and_reports_unhandled_inputs(app, tmp_path):
    session=ready_demo(app)
    preview=app.export_plan(session)
    assert preview['mode']=='hardlink'
    assert preview['ignored']==1
    assert preview['errors']==0
    target=tmp_path/'Canonical library'
    started=app.start_export(session,str(target))
    result=wait(app,started['job_id'])
    assert result['state']=='complete',result
    assert result['result']['mode']=='hardlink'
    assert result['result']['ignored']==1
    exported=json.loads((target/'manifest.json').read_text())
    record=next(r for r in exported['records'] if r.get('export_path'))
    source=Path(record['path'])
    retained=target/record['export_path']
    assert source.stat().st_ino==retained.stat().st_ino
    expected=retained.read_bytes()
    # Simulate the user's explicit final cleanup of a synthetic source entry.
    source.unlink()
    assert retained.read_bytes()==expected
    assert app.session(session)['last_export_mode']=='hardlink'


def test_unsupported_hardlinks_explain_preparation_and_never_silently_copy(app, tmp_path, monkeypatch):
    import errno
    from photo_dedup.media import sha256
    session=ready_demo(app)
    source=next(Path(r['path']) for r in app._manifest(session)['records'] if r['status']=='keep')
    before=sha256(source)
    def unsupported(*args,**kwargs):
        raise OSError(errno.ENOTSUP,'Hard links not supported')
    monkeypatch.setattr('photo_dedup.export.os.link',unsupported)
    destination=tmp_path/'Canonical library'
    started=app.start_export(session,str(destination))
    result=wait(app,started['job_id'])
    assert result['state']=='error'
    assert 'hard-link-capable volume' in result['error']
    assert 'explicitly choose independent copies' in result['error']
    assert not destination.exists()
    assert not list(tmp_path.glob('.Canonical library-*'))
    assert sha256(source)==before
