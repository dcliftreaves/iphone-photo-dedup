# Consolidating old media imports

Photo Dedup is for loose iPhone, smartphone, and SD-card media accumulated through
years of Mac and Windows imports. Its purpose is to build one canonical,
opinionated camera-and-capture-month directory structure without rewriting the
source media. The app builds the library; you inspect it and remove the old source
paths yourself afterward.

## First-run guide

The app opens a four-part guide the first time: why repeated imports need special
handling, preparing the media on one volume, reviewing and building the output,
and finishing the consolidation. Reopen it anytime through **Guide & help** or
the preparation link on the setup screen. Closing it only records that it was
shown; it does not certify that your files are ready or that cleanup is safe.

## 1. Gather the imports onto one working volume

Bring the old downloads onto one hard-link-capable filesystem. Keep each import
folder intact, including Live Photo MOVs, AAE/XMP editing metadata, and RAW/JPEG
companions. Keep a verified backup; two hard links on one disk are not a backup.

```text
Media volume/
  Imports/                   # select this as input
    Old Mac downloads/
    Windows downloads/
    SD-card transfers/
  Canonical library/         # a new folder created by the app
```

Select Imports or its individual import folders, not the common parent containing
the output. The output must be a new folder outside the input folders. It must be
on the same volume as every retained source for the efficient hard-link workflow.
One physical disk can have multiple filesystems; being on the same hardware is
not sufficient. The setup screen checks whether source folders share a filesystem;
that read-only check does not prove the volume supports creating hard links.
APFS, Mac OS Extended (HFS+) and NTFS support them; exFAT and FAT32 do not.
Open **Which disks support hard links?** for a read-only list of local disks.
The optional folder test creates and cleans a tiny temporary file and link, only
after you click the test button. Choose a folder through the native picker if
the OS needs you to grant access. A permission error means unverified support.

If the media is in Apple Photos, export unmodified originals to ordinary folders
first. The app does not enter or edit Photos library bundles. Copies left on other
disks are outside this scan and are not automatically consolidated or removed.

## 2. Scan and review repeated imports

Choose or drag the gathered folders into the desktop app, then **Scan imports**.
You can cancel and restart using cached inspections. Sessions and review choices
are saved locally.

- **Identical:** whole-file SHA-256 fingerprints match.
- **Similar:** normalized image previews look alike; edits, detail, or metadata
  can still matter. Review rather than assuming interchangeability.
- **Possible:** capture metadata agrees, but media content has not been verified.
- **Unique:** no duplicate was confirmed within the current matching rules.

HEIC/JPEG transitions and repeated downloads can change bytes or metadata. The
suggested representative is a heuristic, not certification of the original capture.
Videos use byte and metadata matching, without visual content verification.
Supported RAW files are decoded with LibRaw: identical active sensor samples and
similar generated sRGB renders are separate review evidence classes. Equal sensor
samples do not imply identical file metadata or rendering instructions. Unsupported
RAW variants remain intact with warnings. The optional deeper photo pass in Scan
settings finds content candidates beyond names and capture dates; it skips video
and is bounded, not exhaustive. Missing or changed metadata and unrelated filenames can
also cause equivalent files to remain separate. This is not an exhaustive,
general-purpose near-duplicate search.

Compare groups side by side and select files for the canonical library. Keep both
when uncertain. You must retain at least one file in every group. **Keep all** and
**These aren't duplicates** retain the whole group with different explanations.
**Save choice** persists the selection; **Undo** reverses the latest saved choice.

## 3. Build the canonical directory structure

Choose **Create library**, specify the new same-volume output folder, and preview
the plan. Hard links are the default for this consolidation workflow: the new
library gets new directory entries for retained media without another copy of its
file data. Think of it as a selected set of file copies that take no additional
media space: a new way to view the data you already have. Folders and reports use
a little additional space. It does not include every source file. Media is not re-encoded and embedded metadata is not rewritten.

**Hard links share contents:** editing a retained file through either path can
affect both. Removing an old source path leaves the retained library link intact.
A hard-linked file's data remains while another link exists.

**Independent copies** are an explicit fallback for different volumes or
filesystems without hard links. They need enough space for another copy of the
selected media and do not provide the same storage efficiency.

```text
Canonical library/
  curated/<camera>/<YYYY-MM>/...
  review/<camera>/<YYYY-MM>/...
  manifest.json
  files.csv
  summary.txt
```

The default organization is shown above. You can choose a preset or enter a
relative token template in the export screen, then preview actual paths. For
example, `{year}/{month}/{stem}{ext}` uses year/month folders. Supported tokens are
`{camera}`, `{capture_month}`, `{year}`, `{month}`, `{day}`, `{date}`, `{stem}`,
`{ext}`, and `{kind}`. `{ext}` includes the dot. Category folders stay fixed.

This is an opinionated organization, not a mirror of the old import folders.
Missing camera/date information uses Unknown / Unknown-date. Explicitly retained
review candidates go into curated/. Unresolved similar or possible matches remain
in review/; unreviewed exact byte copies use a suggested representative. Companion
and sidecar preservation may retain extra files; check the actual plan. Filename
collisions receive suffixes and associated sidecars follow the retained stem where
possible.

The preview lists selected files, omitted alternatives, unsupported entries, read
errors, warnings, and unresolved groups. Unsupported files and errors are recorded
but not exported. A completed export includes a manifest accounting for all
inventoried source entries and the selection made.

Each exported file is verified against its scan fingerprint. If a source changed,
scan again. Failed/cancelled exports remove owned partial output. On restart, the
app can recover marked staging folders left by an abrupt exit.

## 4. We don't delete anything

Building the output does not delete or move the old imports. The source cleanup
review links to your top-level input folders in Finder/Explorer and lists files
without a retained byte-identical counterpart. An omitted exact copy is accounted
for separately from an omitted visual alternative, unsupported file, or read error.
Reopen this review from **Review the last export and source cleanup** later.
The manifest and source-cleanup.csv also preserve it. This is saved scan/export
evidence, not a fresh disk audit: later additions/changes and files inside ignored
directories are not covered. To finish:

1. Open curated/ and review/, spot-check photos and videos, inspect camera/date
   placement, and verify the backup of the collection you intend to retain.
2. Read files.csv, summary.txt, and **Skipped & warnings**. Omitted alternatives,
   unsupported entries, and read errors may exist only in the old source folders.
   Preserve anything you still need before removing those folders.
3. After verifying the result, manually remove the old import paths using Finder
   or Explorer. The app does not perform this cleanup. Do not delete entire source
   trees blindly, and keep the canonical output folder separate from that cleanup.

Selected hard-linked media remains accessible through the canonical library after
old source links are removed. Space from unwanted copies is reclaimed only when
their last links are removed; items in Trash can still occupy space. Simply
creating the library does not itself reclaim source disk space.

## Local operation

There is no photo upload, account, or hosted processing. Native application state
uses the per-user app-data directory; development launches through
scripts/run_desktop.py keep it in this checkout under local-validation/app-data.
The demo uses synthetic media and is for learning the workflow.

Hard-link semantics are described in [Microsoft's filesystem documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions).
