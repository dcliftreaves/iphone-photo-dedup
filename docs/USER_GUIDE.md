# Using Photo Dedup

## Start with a small folder

Open Photo Dedup and choose a folder, or drag folders onto the desktop window.
You can compare multiple imports in one scan. Try the demo first if you want to
see the workflow without choosing your own photos.

Click **Find duplicates**. Reading files does not edit their contents. The scan
session is saved locally; cancel safely if you need to stop. A restarted scan
reuses cached inspections. Native application data is kept in your user profile;
development data in this checkout is under .photo-dedup or local-validation.

Apple Photos libraries are not entered. In Photos, export the unmodified
originals into ordinary folders before scanning.

## Understand the results

- **Identical:** the whole-file SHA-256 fingerprints match.
- **Similar:** normalized image previews look similar. Edits, detail, or metadata
  can still matter. Compare before choosing.
- **Possible:** capture metadata agrees but the media content has not been verified.
- **Unique:** no duplicate was confirmed for that file.

Search by filename or path. Open a group to compare files. Selecting a checkbox
means including that file in the new library. You must keep at least one file per
group. **Keep all** and **These aren't duplicates** both retain every group member,
with different review explanations. Closing the comparison leaves its choice as-is.

Zoom affects all previews together; scrolling a zoomed image synchronizes the
others. Preview regions are keyboard-focusable. Videos and RAW files currently
have no rendered preview: use the original-path action to find them in the OS file
manager. File details explain the evidence used, metadata warnings, and companions.

**Save choice** stores the decision automatically. **Undo** reverses the latest
saved choice. Recent scans can be reopened after quitting the app.

## Create an organized copy

Choose **Create library**, then select a parent folder or type a new destination.
The destination must not exist or be inside a source folder. Preview the export
before creating it. The preview shows selected files, omitted alternatives, byte
size, available space, and unresolved groups.

Independent copies are the default. They work across filesystems and can be
edited without changing source contents. Hard links are an advanced same-filesystem
option: editing an exported link also changes the source's contents.

Unreviewed identical byte copies use the suggested representative. Unreviewed
similar and possible matches stay included for later review. Explicit choices
control which alternatives are exported. AAE/XMP editing metadata is preserved;
associated local Live Photo and RAW/image companions can cause extra files to be
retained. The export preview and manifest record the actual selection.

Files are organized under curated/ and review/, with camera/date subfolders where
metadata is available. Explicitly kept review candidates go into curated/. Filename
collisions receive suffixes; sidecars follow their associated output stem when
possible. A complete manifest, CSV, and readable summary are included.

Every exported source is rechecked against its scan fingerprint. If a source has
changed, the export stops and explains that a new scan is needed. Cancelled/failed
exports remove their generated partial output. An abrupt process exit can leave a
marked staging folder; the next application launch recovers and removes owned,
marked partial output. Unrelated folders are never inferred to be temporary files.

## Important distinctions

Creating a library does not reclaim disk space from your source folders. Omitted
alternatives remain at their original locations. This app does not delete sources,
edit Apple Photos libraries, or automatically declare similar images safe to remove.

Metadata or decoder warnings do not disappear. Open **Skipped & warnings** to see
them, including unsupported extensions and entries that were not inspected.
