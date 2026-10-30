"""Feature tests exercise real RAW decoding, export paths, disk probes and cleanup evidence."""
import base64
from copy import deepcopy
import errno
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

from photo_dedup.app import AppService
from photo_dedup.export import plan_export
from photo_dedup.layout import render_template, validate_template
from photo_dedup.matching import classify
from photo_dedup.media import inspect, visual_hash
from photo_dedup.scanner import scan
from photo_dedup.volumes import support, test_folder as probe


from scripts.synthetic_dng import make_dng


def test_real_raw_sample_evidence_and_preview(tmp_path):
    a,b,c = [tmp_path/name for name in ['IMG_0001.dng','renamed.dng','different.dng']]
    make_dng(a)
    make_dng(b,'metadata rewritten')
    make_dng(c,change=True)
    records = [dict(inspect(p),status='pending') for p in (a,b,c)]
    assert all(r.get('thumbnail') for r in records), [r['warnings'] for r in records]
    assert records[0]['sha256'] != records[1]['sha256']
    assert records[0]['raw_samples_sha256'] == records[1]['raw_samples_sha256']
    assert records[0]['raw_samples_sha256'] != records[2]['raw_samples_sha256']
    classify(records)
    assert len([r for r in records if r['confidence']=='raw_samples']) == 1
    assert len([r for r in records if r['status']=='keep']) == 2
    assert all(r['status'] != 'exact_duplicate' for r in records)


def test_raw_cache_and_unsupported_retention(tmp_path):
    root=tmp_path/'inputs';root.mkdir()
    make_dng(root/'photo.dng')
    (root/'unsupported.cr3').write_bytes(b'not a real RAW file')
    manifest=scan([root],cache_path=tmp_path/'cache.db',workers=2)
    raw=next(r for r in manifest['records'] if r['path'].endswith('photo.dng'))
    assert raw.get('raw_decoder')
    assert 'thumbnail' not in raw
    bad=next(r for r in manifest['records'] if r['path'].endswith('unsupported.cr3'))
    assert bad['status']=='keep' and any('RAW decoding unavailable' in w for w in bad['warnings'])
    again=scan([root],cache_path=tmp_path/'cache.db')
    assert again['cache_hits']==1  # unsupported files can be retried after decoder/access changes


def image_record(path, pixels):
    im=Image.fromarray(pixels).convert('RGBA')
    return dict(path=path,kind='image',extension='.png',status='pending',sha256=path,
                thumbnail=base64.b64encode(im.resize((128,128)).tobytes()).decode(),
                visual_hash=visual_hash(im),width=128,height=128,warnings=[],metadata={})


def test_deep_pass_finds_renames_without_date_and_skips_video():
    pixels=np.random.default_rng(4).integers(0,256,(128,128,3),dtype=np.uint8)
    original=[image_record('/a.png',pixels),image_record('/unrelated.png',pixels.copy())]
    regular=deepcopy(original);classify(regular)
    assert all(r['status']=='keep' for r in regular)
    video=dict(path='/movie.mp4',kind='video',extension='.mp4',status='pending',sha256='movie',warnings=[],metadata={})
    records=deepcopy(original)+[video];classify(records,deep_scan=True)
    assert sum(r['confidence']=='deep_visual_similarity' for r in records)==1
    assert video['status']=='keep'
    assert all(r['status']!='exact_duplicate' for r in records)


def test_deep_matches_point_directly_to_kept_representatives():
    pixels=np.full((128,128,3),100,dtype=np.uint8)
    records=[image_record(f'/{name}.png',pixels+delta) for name,delta in [('a',0),('b',3),('c',6)]]
    classify(records,threshold=.015,deep_scan=True)
    by_path={r['path']:r for r in records}
    for r in records:
        if r.get('matched_to'):
            assert by_path[r['matched_to']]['status']=='keep'


@pytest.mark.parametrize('template',['../{stem}{ext}','/{stem}{ext}','a//{stem}{ext}','{unknown}/{stem}{ext}','{stem.__class__}{ext}','{stem!r}{ext}','{stem:10}{ext}','C:/{stem}{ext}','{stem}','{ext}/{stem}{ext}','{stem}\\{ext}'])
def test_invalid_templates(template):
    with pytest.raises(ValueError):validate_template(template)


def test_templates_sanitize_values_and_preserve_sidecars():
    record=dict(path='/input/IMG_0001.JPG',model='../../Camera:evil',captured_at='2020:07:08 12:00:00',extension='.jpg',kind='image',status='keep')
    relative=render_template(record,'{year}/{month}/{camera}/{stem}{ext}')
    assert str(relative)=='2020/07/_.._Camera_evil/IMG_0001.jpg'
    record['captured_at']=None
    assert str(render_template(record)).endswith('Unknown-date/IMG_0001.jpg')
    sidecar=dict(path='/input/IMG_0001.xmp',extension='.xmp',kind='sidecar',status='sidecar',related_files=[dict(path=record['path'])])
    plan=plan_export(dict(records=[record,sidecar],layout=dict(template='{kind}/{stem}{ext}')))
    assert plan[sidecar['path']].parent==plan[record['path']].parent
    assert plan[sidecar['path']].stem==plan[record['path']].stem


def test_probe_cleans_and_denial_is_unknown(tmp_path,monkeypatch):
    assert probe(str(tmp_path))['verified'] is True
    assert list(tmp_path.iterdir())==[]
    def denied(*args,**kwargs):raise PermissionError(errno.EACCES,'denied')
    monkeypatch.setattr('photo_dedup.volumes.os.link',denied)
    result=probe(str(tmp_path))
    assert result['supported'] is None and result['verified'] is False
    assert list(tmp_path.iterdir())==[]
    assert support('APFS') is True and support('NTFS') is True
    assert support('exFAT') is False and support('smbfs') is None


def test_cleanup_distinguishes_identical_omissions_from_distinct_loss():
    records=[dict(path='/input/a.jpg',sha256='a',status='keep'),
             dict(path='/input/copy.jpg',sha256='a',status='exact_duplicate'),
             dict(path='/input/edited.jpg',sha256='b',status='review'),
             dict(path='/input/document.txt',status='ignored',reason='Unsupported file extension.')]
    result=AppService._cleanup_records(dict(records=records),{'/input/a.jpg':Path('curated/a.jpg')})
    assert [r['disposition'] for r in result]==['retained','identical_retained','not_retained','not_retained']


def test_app_raw_preview_custom_export_and_cleanup_persist(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr('photo_dedup.scanner.exiftool_path',lambda:None)
    inputs=tmp_path/'inputs';inputs.mkdir()
    make_dng(inputs/'original.dng')
    make_dng(inputs/'rewritten.dng','new metadata')
    (inputs/'notes.txt').write_text('Only at source')
    data=tmp_path/'app'
    def finish(app, job_id):
        end=time.monotonic()+20
        while time.monotonic()<end:
            job=app.job(job_id)
            if job['state']!='running':
                assert job['state']=='complete',job
                return job
            time.sleep(.01)
        raise AssertionError('Job timed out')
    with_app=AppService(data)
    try:
        started=with_app.start_scan([str(inputs)],deep_scan=True)
        finish(with_app,started['job_id']);session=started['session_id']
        summary=with_app.groups(session)['items'][0]
        assert summary['kind']=='visual' and summary['has_image']
        group=with_app.group(session,summary['id'])
        assert with_app.preview(session,group['recommendation'])['image'].startswith('data:image/png;base64,')
        with_app.decide(session,group['id'],'keep_selected',[group['recommendation']])
        template='{kind}/{year}/{stem}{ext}'
        preview=with_app.export_plan(session,template=template)
        exported=with_app.start_export(session,str(tmp_path/'output'),template=template)
        finish(with_app,exported['job_id'])
        assert all((tmp_path/'output'/r['destination']).is_file() for r in preview['records'])
        assert 'cleanup_records' not in with_app.session(session)  # history responses stay small
        review=with_app.cleanup_review(session)
        assert review['total']==2  # rewritten RAW has distinct file metadata; never call it safely discarded
        assert review['roots'][0]['at_risk']==2
    finally:with_app.close()
    reopened=AppService(data)
    try:assert reopened.cleanup_review(session)['total']==2
    finally:reopened.close()


def test_local_disk_discovery_reports_read_only_and_excludes_system_mounts(monkeypatch):
    import plistlib
    from photo_dedup.volumes import local_volumes
    monkeypatch.setattr('photo_dedup.volumes.sys.platform','darwin')
    infos={
        'disk1':dict(MountPoint='/System/Volumes/Data',FilesystemType='apfs',WritableVolume=True),
        'disk2':dict(MountPoint='/Volumes/ReadOnly',FilesystemType='apfs',WritableVolume=False),
        'disk3':dict(MountPoint='/System/Volumes/Preboot',FilesystemType='apfs',WritableVolume=True),
        'disk4':dict(MountPoint='/Volumes/Card',FilesystemType='exfat',WritableVolume=True),
    }
    def result(command,**kwargs):
        if command[1]=='list':return plistlib.dumps(dict(AllDisks=list(infos)))
        return plistlib.dumps(infos[command[-1]])
    monkeypatch.setattr('photo_dedup.volumes.subprocess.check_output',result)
    volumes=local_volumes()['volumes']
    assert len(volumes)==3
    assert next(v for v in volumes if v['path']=='/Volumes/ReadOnly')['read_only'] is True
    assert next(v for v in volumes if v['path']=='/Volumes/Card')['supported'] is False


def test_detected_media_kind_can_override_a_misleading_extension(tmp_path):
    from photo_dedup.media import open_normalized
    disguised_raw=tmp_path/'old-import.jpg'
    make_dng(disguised_raw)
    assert open_normalized(disguised_raw,raw=True).width > 0
    disguised_png=tmp_path/'old-import.dng'
    Image.new('RGB',(40,30)).save(disguised_png,format='PNG')
    assert open_normalized(disguised_png,raw=False).size == (40,30)
