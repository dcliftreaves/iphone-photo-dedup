import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from photo_dedup.cli import main
from photo_dedup.export import export_manifest, load_manifest
from photo_dedup.matching import classify
from photo_dedup.media import inspect, rmse, sha256
from photo_dedup.reports import write_reports
from photo_dedup.scanner import scan


@pytest.fixture(autouse=True)
def no_external_exiftool(monkeypatch):
    monkeypatch.setattr('photo_dedup.scanner.exiftool_path', lambda: None)


def image(path, color='red', **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (64, 48), color).save(path, **kwargs)
    return path


def test_unsigned_overflow_cannot_make_black_and_white_similar():
    a = np.zeros((32, 32, 4), dtype=np.uint8)
    b = np.full((32, 32, 4), 255, dtype=np.uint8)
    assert rmse(a, b) == 1.0
    assert rmse(a, a) == 0.0


def test_complete_manifest_exact_visual_distinct_sidecar_ignored_and_corrupt(tmp_path):
    root = tmp_path / 'inputs'
    first = image(root / 'original' / 'IMG_0001.png')
    image(root / 'derivative' / 'IMG_0001.jpg', quality=100)
    shutil.copy2(first, root / 'original' / 'same.png')
    image(root / 'different' / 'IMG_0001.png', color='blue')
    (root / 'original' / 'IMG_0001.aae').write_text('<plist/>')
    (root / 'notes.txt').write_text('not a photo')
    (root / 'bad.jpg').write_bytes(b'not a jpeg')
    (root / 'symlink.jpg').symlink_to(first)
    manifest = scan([root], workers=2)
    records = manifest['records']
    assert len(records) == 8
    assert sum(r['status'] == 'exact_duplicate' for r in records) == 1
    assert sum(r['status'] == 'review' for r in records) == 1
    assert sum(r['status'] == 'ignored' for r in records) == 2
    assert sum(r['status'] == 'sidecar' for r in records) == 1
    bad = next(r for r in records if r['path'].endswith('bad.jpg'))
    assert bad['warnings'] and bad['status'] == 'keep'
    sidecar = next(r for r in records if r['status'] == 'sidecar')
    assert sidecar['related_files']
    assert all('reason' in r and 'confidence' in r for r in records)


def test_cache_reuses_inspection_and_invalidates_changed_files_and_reclassifies(tmp_path):
    root = tmp_path / 'inputs'
    image(root / 'a' / 'IMG_0001.png')
    image(root / 'b' / 'IMG_0001.jpg', quality=100)
    cache = tmp_path / 'cache.sqlite3'
    first = scan([root], cache_path=cache)
    again = scan([root], cache_path=cache, threshold=0)
    assert first['cache_hits'] == 0
    assert again['cache_hits'] == 2
    assert not any(r['status'] == 'review' for r in again['records'])
    image(root / 'a' / 'IMG_0001.png', color='blue')
    changed = scan([root], cache_path=cache)
    assert changed['cache_hits'] == 1
    assert not any(r['status'] == 'review' for r in changed['records'])
    (root / 'a' / 'IMG_0001.png').unlink()
    assert len(scan([root], cache_path=cache)['records']) == 1


@pytest.mark.parametrize('mode', ['copy', 'hardlink'])
def test_export_preserves_every_media_file_and_rejects_reruns(tmp_path, mode):
    root = tmp_path / 'inputs'
    first = image(root / 'a' / 'IMG_0001.png')
    shutil.copy2(first, root / 'a' / 'same.png')
    image(root / 'b' / 'IMG_0001.jpg', quality=100)
    (root / 'a' / 'IMG_0001.aae').write_text('<plist/>')
    before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    manifest = scan([root])
    output = tmp_path / 'export'
    plan = export_manifest(manifest, output, mode=mode)
    assert len(plan) == 4
    assert len(list((output / 'duplicates').rglob('*.png'))) == 1
    assert len(list((output / 'review').rglob('*.jpg'))) == 1
    for record in manifest['records']:
        target = output / plan[record['path']]
        assert sha256(target) == record['sha256']
        if mode == 'hardlink':
            assert target.stat().st_ino == Path(record['path']).stat().st_ino
        if record['status'] == 'sidecar':
            assert target.with_suffix('.png').exists()
    assert {str(p): sha256(p) for p in root.rglob('*') if p.is_file()} == before
    assert load_manifest(output / 'manifest.json')['schema_version'] == 1
    with pytest.raises(ValueError, match='already exists'):
        export_manifest(manifest, output, mode=mode)


def test_changed_source_aborts_export_and_removes_partial_output(tmp_path):
    root = tmp_path / 'inputs'
    image(root / 'a.png')
    image(root / 'z.png', color='blue')
    manifest = scan([root])
    image(root / 'z.png', color='green')
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='changed since scan'):
        export_manifest(manifest, output, mode='copy')
    assert not output.exists()
    assert not list(tmp_path.glob('.export-*'))


def test_dry_run_does_not_create_report_cache_or_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'inputs'
    image(root / 'a.png')
    assert main(['scan', str(root), '--dry-run', '--quiet']) == 0
    assert not (tmp_path / 'results').exists()
    assert not (tmp_path / '.dedup-cache.sqlite3').exists()
    manifest = scan([root])
    assert len(export_manifest(manifest, tmp_path / 'out', dry_run=True)) == 1
    assert not (tmp_path / 'out').exists()


def test_export_rejects_output_inside_sources_and_symlink_replacement(tmp_path):
    root = tmp_path / 'inputs'
    source = image(root / 'a.png')
    manifest = scan([root])
    with pytest.raises(ValueError, match='contain one another'):
        export_manifest(manifest, root / 'out')
    source.rename(root / 'renamed.png')
    source.symlink_to(root / 'renamed.png')
    with pytest.raises(ValueError, match='symbolic link'):
        export_manifest(manifest, tmp_path / 'out')


def test_filename_collisions_and_companion_stems_remain_paired(tmp_path):
    root = tmp_path / 'inputs'
    image(root / 'a' / 'IMG_0001.png')
    image(root / 'b' / 'IMG_0001.png', color='blue')
    for folder in ['a', 'b']:
        (root / folder / 'IMG_0001.aae').write_text('<plist/>')
    manifest = scan([root])
    plan = export_manifest(manifest, tmp_path / 'out', mode='copy')
    assert len(set(plan.values())) == 4
    for path, target in plan.items():
        if path.endswith('.aae'):
            assert target.with_suffix('.png') == plan[str(Path(path).with_suffix('.png'))]


def test_misnamed_image_is_detected_without_changing_source(tmp_path):
    source = image(tmp_path / 'wrong.jpg', format='PNG')
    result = inspect(source)
    assert result['extension'] == '.png'
    assert source.exists()


def test_heic_decode_and_orientation(tmp_path):
    from pillow_heif import from_pillow
    path = tmp_path / 'image.heic'
    from_pillow(Image.new('RGB', (64, 48), 'red')).save(path)
    record = inspect(path)
    assert record['thumbnail'] and not record['warnings']
    assert (record['width'], record['height']) == (64, 48)
    exif = Image.Exif()
    exif[274] = 6
    rotated = image(tmp_path / 'rotated.jpg', exif=exif)
    assert inspect(rotated)['width'] == 48


def test_full_image_comparison_keeps_small_edits_for_review(tmp_path):
    root = tmp_path / 'inputs'
    image(root / 'a' / 'IMG_0001.png', color=(20, 20, 20))
    image(root / 'b' / 'IMG_0001.png', color=(21, 21, 21))
    manifest = scan([root], full_images=True)
    assert sum(r['status'] == 'review' for r in manifest['records']) == 1
    assert manifest['options']['comparison'] == 'full'


def media_record(path, kind='video', **extra):
    return dict(path=path, status='pending', sha256=hashlib.sha256(path.encode()).hexdigest(),
                kind=kind, extension='.mov', metadata={'Duration': 2}, warnings=[],
                capture_key='2020:01:01 00:00:00||', model='iPhone', width=100, height=100,
                **extra)


def test_metadata_only_video_matching_requires_equal_duration():
    records = [media_record('/a/a.mov'), media_record('/b/a.mov'), media_record('/c/a.mov')]
    records[2]['metadata']['Duration'] = 3
    classify(records)
    assert records[1]['status'] == 'review'
    assert records[1]['confidence'] == 'metadata_only'
    assert records[2]['status'] == 'keep'


def test_live_photo_ids_and_raw_pairs_are_relationships_not_duplicates():
    records = [media_record('/a/IMG_0001.mov', content_id='uuid'),
               media_record('/b/IMG_0002.heic', kind='image', content_id='uuid'),
               media_record('/b/IMG_0002.dng', kind='raw')]
    records[1]['extension'] = '.heic'
    records[2]['extension'] = '.dng'
    classify(records)
    assert all(r['status'] == 'keep' for r in records)
    assert any(r['relationship'] == 'live_photo' for r in records[0]['related_files'])
    assert any(r['relationship'] == 'raw_jpeg_pair' for r in records[1]['related_files'])


def test_invalid_cli_arguments_fail_cleanly(tmp_path, capsys):
    assert main(['scan', str(tmp_path / 'missing')]) == 2
    assert 'Not a valid input directory' in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(['scan', str(tmp_path), '--workers', '0'])
    assert error.value.code == 2


def test_overlapping_roots_are_not_scanned_twice_and_symlink_dirs_not_followed(tmp_path):
    root = tmp_path / 'inputs'
    image(root / 'sub' / 'a.png')
    (root / 'link').symlink_to(root / 'sub', target_is_directory=True)
    manifest = scan([root, root / 'sub'])
    assert len(manifest['records']) == 2
    assert sum(r['status'] == 'ignored' for r in manifest['records']) == 1


def test_csv_and_json_roundtrip_and_csv_formula_escaping(tmp_path):
    root = tmp_path / 'inputs'
    source = image(root / 'a.png')
    manifest = scan([root])
    manifest['records'][0]['reason'] = '=HYPERLINK("example")'
    output = tmp_path / 'report'
    output.mkdir()
    write_reports(manifest, output)
    with (output / 'files.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['reason'].startswith("'=")
    assert load_manifest(output / 'manifest.json')['records'][0]['path'] == str(source)
    payload = json.loads((output / 'manifest.json').read_text())
    payload['records'][0]['extension'] = '/../../escape'
    (output / 'manifest.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Invalid media extension'):
        load_manifest(output / 'manifest.json')


def test_renamed_exact_aliases_match_even_when_alias_is_sorted_later(tmp_path):
    root = tmp_path / 'inputs'
    first = image(root / 'a.png', color=(20, 20, 20))
    image(root / 'IMG_0001.png', color=(21, 21, 21))
    shutil.copy2(first, root / 'IMG_0001 1.png')
    manifest = scan([root])
    assert sum(r['status'] == 'exact_duplicate' for r in manifest['records']) == 1
    assert sum(r['status'] == 'review' for r in manifest['records']) == 1


def test_similarity_is_not_transitive(tmp_path):
    root = tmp_path / 'inputs'
    for folder, level in [('a', 20), ('b', 24), ('c', 28)]:
        image(root / folder / 'IMG_0001.png', color=(level, level, level))
    records = scan([root], threshold=.02)['records']
    assert records[0]['status'] == 'keep'
    assert records[1]['status'] == 'review'
    assert records[2]['status'] == 'keep'


def test_static_metadata_warnings_are_cached_but_failures_are_retried(tmp_path, monkeypatch):
    from photo_dedup import media
    root = tmp_path / 'inputs'
    image(root / 'a.png')
    cache = tmp_path / 'cache.sqlite3'
    monkeypatch.setattr(media, 'metadata', lambda *args: {'Warning': '[minor] A metadata issue'})
    assert scan([root], cache_path=cache)['cache_hits'] == 0
    second = scan([root], cache_path=cache)
    assert second['cache_hits'] == 1
    assert second['records'][0]['warnings']
    # A failure should not permanently prevent metadata acquisition on a future scan.
    image(root / 'b.png', color='blue')
    def failure(*args):
        raise OSError('temporary read failure')
    monkeypatch.setattr(media, 'metadata', failure)
    scan([root], cache_path=cache)
    assert scan([root], cache_path=cache)['cache_hits'] == 1


def test_multi_frame_images_not_classified_from_first_frame(tmp_path):
    root = tmp_path / 'inputs'
    root.mkdir()
    a = Image.new('RGB', (32, 32), 'red')
    for name, color in [('a.tiff', 'blue'), ('b.tiff', 'green')]:
        a.save(root / name, save_all=True, append_images=[Image.new('RGB', (32, 32), color)])
    records = scan([root])['records']
    assert all(r['status'] == 'keep' and r['multi_frame'] for r in records)


def test_export_interrupt_cleans_staging_without_touching_sources(tmp_path):
    root = tmp_path / 'inputs'
    source = image(root / 'a.png')
    before = sha256(source)
    manifest = scan([root])
    def interrupted(path):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        export_manifest(manifest, tmp_path / 'out', mode='copy', progress=interrupted)
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.out-*'))
    assert sha256(source) == before


def test_file_changed_during_inspection_is_not_cached(tmp_path, monkeypatch):
    from photo_dedup import media
    root = tmp_path / 'inputs'
    image(root / 'a.png')
    original = media.sha256
    def changed(path):
        digest = original(path)
        image(path, color='blue')
        return digest
    monkeypatch.setattr(media, 'sha256', changed)
    manifest = scan([root], cache_path=tmp_path / 'cache.sqlite3')
    assert manifest['records'][0]['status'] == 'error'
    assert 'changed during inspection' in manifest['records'][0]['reason']


def test_alpha_channel_participates_in_visual_comparison(tmp_path):
    root = tmp_path / 'inputs'
    for folder, alpha in [('a', 0), ('b', 255)]:
        path = root / folder / 'IMG_0001.png'
        path.parent.mkdir(parents=True)
        Image.new('RGBA', (32, 32), (255, 0, 0, alpha)).save(path)
    assert all(r['status'] == 'keep' for r in scan([root])['records'])
