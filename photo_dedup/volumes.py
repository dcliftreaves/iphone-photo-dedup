"""Read-only local volume discovery and an explicit, tiny hard-link probe."""
import errno
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile


def support(filesystem):
    fs = filesystem.casefold()
    if fs in {'apfs', 'hfs', 'hfs+', 'hfsplus', 'ntfs', 'ext2', 'ext3', 'ext4', 'btrfs', 'xfs'}:
        return True
    if fs in {'exfat', 'fat', 'fat32', 'msdos', 'vfat', 'msdos fat32'}:
        return False
    return None


def local_volumes():
    """Only local block devices; never traverse or probe network mounts."""
    results = []
    unavailable = 0
    if sys.platform == 'darwin':
        def diskutil(*args):
            return plistlib.loads(subprocess.check_output(['/usr/sbin/diskutil', *args], timeout=15, stderr=subprocess.DEVNULL))
        try:
            listing = diskutil('list', '-plist')
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ValueError('Could not read the local disk list. Reconnect your drive and try again, or test a chosen folder directly.') from error
        for identifier in listing.get('AllDisks', []):
            try:
                info = diskutil('info', '-plist', identifier)
            except (OSError, ValueError, subprocess.SubprocessError):
                # Volumes can disconnect between listing and inspection. Preserve
                # the remaining results without guessing this volume's capability.
                unavailable += 1
                continue
            mount = info.get('MountPoint')
            if not mount or info.get('NetworkMount', False):
                continue
            # Hide macOS implementation mounts; show the writable Data volume as /.
            if mount == '/System/Volumes/Data':
                mount = '/'
            elif mount.startswith('/System/Volumes/') or mount == '/' and not info.get('WritableVolume', info.get('Writable', False)):
                continue
            fs = info.get('FilesystemType', '')
            results.append(dict(path=mount, name=info.get('VolumeName') or mount, filesystem=fs,
                                supported=support(fs), read_only=not info.get('WritableVolume', info.get('Writable', False)),
                                evidence='Filesystem type reported by macOS; folder permissions are not tested.'))
    elif sys.platform == 'win32':
        import ctypes
        kernel = ctypes.windll.kernel32
        buffer = ctypes.create_unicode_buffer(512)
        kernel.GetLogicalDriveStringsW(len(buffer), buffer)
        for mount in buffer[:].split('\0'):
            if not mount or kernel.GetDriveTypeW(mount) not in (2, 3):
                continue
            name, fs = ctypes.create_unicode_buffer(261), ctypes.create_unicode_buffer(261)
            flags = ctypes.c_ulong()
            ok = kernel.GetVolumeInformationW(mount, name, len(name), None, None, ctypes.byref(flags), fs, len(fs))
            results.append(dict(path=mount, name=name.value or mount, filesystem=fs.value,
                                supported=bool(flags.value & 0x00400000) if ok else None,
                                read_only=bool(flags.value & 0x00080000) if ok else None,
                                evidence='Windows volume capability flags; folder permissions are not tested.'))
    elif Path('/proc/self/mountinfo').is_file():
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields, device = before.split(), after.split()
            if not device[1].startswith('/dev/'):
                continue
            mount = fields[4].replace('\\040', ' ')
            results.append(dict(path=mount, name=mount, filesystem=device[0], supported=support(device[0]),
                                read_only='ro' in fields[5].split(','),
                                evidence='Local mount filesystem type; folder permissions are not tested.'))
    return dict(volumes=sorted({r['path']: r for r in results}.values(), key=lambda r: r['path']),
                message=('Local mounted volumes only. No media was read and no test files were written.' +
                         (f' {unavailable} disk(s) could not be checked; they may have disconnected. Check again or test a folder directly.' if unavailable else '')))


def test_folder(path):
    """Called only from an explicit user action, never during discovery or scanning."""
    folder = Path(path).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('Choose an existing folder to test')
    try:
        with tempfile.TemporaryDirectory(prefix='.photo-dedup-link-test-', dir=folder) as temporary:
            original = Path(temporary) / 'original'
            linked = Path(temporary) / 'linked'
            original.write_bytes(b'Photo Dedup hard-link check\n')
            os.link(original, linked)
            verified = os.path.samefile(original, linked) and linked.read_bytes() == original.read_bytes()
            if not verified:
                raise ValueError('The filesystem did not report a shared file identity')
        return dict(path=str(folder), supported=True, verified=True,
                    message='Hard links work in this folder. Temporary test files were removed.')
    except OSError as error:
        unsupported = error.errno in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV}
        return dict(path=str(folder), supported=False if unsupported else None, verified=False,
                    message=('Hard links are unavailable here.' if unsupported else
                             'Could not verify hard links. Check folder access, available space, and OS permissions.') + f' {error}')
