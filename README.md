# iPhone Photo Dedup

An opinionated consolidator for old iPhone, smartphone, and SD-card media gathered
through years of Mac and Windows imports. Collect the loose import folders on one
working volume, review repeated downloads and conversions, and build a canonical
camera/month library using hard links without rewriting source file contents.

**Preparation and the final cleanup are part of the workflow.** Every retained
source and the output must share a hard-link-capable filesystem for efficient
consolidation. After inspecting the output and accounting for unexported files,
you manually remove the old import paths. The app does not do that deletion for you.

Start with the first-run guide in the app or [the full workflow](docs/USER_GUIDE.md).
This is not direct Apple Photos library editing or an exhaustive search for every
visually related image. Uncertain matches and metadata limitations remain visible.

## Desktop application

The Tauri desktop application bundles the Python engine, HEIC decoding, and
ExifTool. It provides folder selection and drag-and-drop, grouped review,
synchronized image comparison, saved decisions and undo, and verified export.
Hard links are the same-volume default. Independent copies are an explicit fallback
for other filesystems and require additional storage. The first-run guide covers
preparation, the output structure, and the user’s final source-cleanup step.

Download native installers from GitHub Releases after the release has been
published. Release automation is prepared in .github/workflows/release.yml;
local builds have not been published automatically.

To preview locally from source:

```sh
npm ci --prefix desktop
npm --prefix desktop run build
.venv/bin/photo-dedup ui --data-dir .photo-dedup
```

Set up the Python environment below first. For the packaged desktop build and
release process, see [RELEASING.md](docs/RELEASING.md). The application plan is in
[APP_PLAN.md](docs/APP_PLAN.md), and the end-user flow is in
[USER_GUIDE.md](docs/USER_GUIDE.md). Local verification and limitations are recorded
in [VERIFICATION.md](docs/VERIFICATION.md).

## CLI quick start

Python 3.10 or newer is required. On macOS, install Python and ExifTool if needed:

```sh
brew install python exiftool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
photo-dedup doctor
```

ExifTool is optional but strongly recommended for video and RAW metadata, detecting
incorrect file extensions, and Live Photo identifiers. Pillow supplies basic image
metadata without it. HEIC decoding is installed through `pillow-heif`.

Start with a report; scan does not copy or link your media:

```sh
photo-dedup scan "/Volumes/Photos/iPhone imports" "/Volumes/Photos/Old downloads" \
  --output results/first-scan
```

Read `summary.txt` for counts, open `files.csv` in a spreadsheet to filter decisions,
and keep `manifest.json` as the complete, versioned scan record. Every encountered
file appears, including unsupported files, unreadable files, and symbolic links.
Excluded paths are deliberately outside the inventory. Directory enumeration errors
stop the scan rather than presenting an incomplete library as a successful result.

Preview an export and then create it:

```sh
photo-dedup export results/first-scan/manifest.json \
  --output "/Volumes/Photos/Curated" --dry-run
photo-dedup export results/first-scan/manifest.json \
  --output "/Volumes/Photos/Curated"
```

The default export uses hard links and requires all exported sources and the output
to share a filesystem that supports hard links. For sources on different drives or
an independent copy, use `--mode copy` in both commands. Export verifies each file's
SHA-256 against the saved scan and verifies the resulting file. Changed sources
require a new scan. Existing destinations are never reused.

**Hard links share the same file contents. Editing an exported hard link changes
its source too.** Deleting one link only removes that directory entry. Export does
not itself edit file contents. Filesystem link counts and change times can change.

## What each decision means

| Manifest status | Meaning | Export destination |
| --- | --- | --- |
| `keep` | No duplicate confirmed; retained as a representative | `curated/` |
| `exact_duplicate` | SHA-256 matches another file byte for byte | `duplicates/exact/` |
| `review` / `visual_similarity` | Similar normalized image pixels; may be an edit | `review/` |
| `review` / `raw_samples` | Decoded active RAW samples and CFA layout agree; metadata/rendering instructions may differ | `review/` |
| `review` / `raw_render_similarity` | Similar generated RAW renders; samples may differ | `review/` |
| `review` / `deep_visual_similarity` | Content-indexed image similarity without requiring names/dates | `review/` |
| `review` / `metadata_only` | Capture metadata agrees, but content is unverified | `review/` |
| `sidecar` | Editing metadata preserved independently | Beside its local media when associated |
| `ignored` | Unsupported extension, symbolic link, or nonregular entry | Not exported; listed in reports |
| `error` | Unable to inspect this file reliably | Not exported; listed in reports |

All supported, readable media is exported, including exact duplicates and every
review candidate. No decision authorizes deletion. A file with a decode or metadata
warning is retained using the evidence available; inspect its `warnings` column.

Visual comparison applies EXIF orientation, converts embedded color profiles to
sRGB, retains alpha, and calculates normalized root mean squared error using
floating point. The default threshold is 0.02 on 128×128 previews. This catches
some format conversions but cannot prove equivalence or detect every small edit.
All visual matches need human review. Animated/multi-frame images get exact-byte
matching only; first-frame previews are not used to classify them as similar.

Candidates share capture time (including available subsecond/timezone metadata),
camera model, kind, and dimensions. Images without capture dates use normalized
filenames and dimensions as a fallback. Unrelated names without usable capture
metadata may remain separate in the normal pass. The optional `--deep-scan` indexes
perceptual image hashes and checks at most 200 candidates per image by RMSE, without
requiring names or dates. It skips video, crops and rotations can be missed, and it
is a suggestion pass rather than an exhaustive visual search. Video metadata candidates
must also have equal known durations; this does not verify their frames or audio.

Representative selection is deterministic: preferred format, shortest filename,
oldest filesystem modification time, then full path. The image preference starts
DNG, HEIF, HEIC, TIFF, PNG, JPEG. This is a recommendation for review, not proof of
which file is the original. Video codecs are not ranked as evidence of quality.
Exact matches are grouped first, including renamed aliases. Visual candidates are
compared directly to retained representatives, avoiding chains of weak similarities.

### Companions and supported formats

- Images: JPEG, PNG, HEIC/HEIF, TIFF, WebP.
- RAW: DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2. Preserved with byte matching and
  optional ExifTool metadata. LibRaw (via rawpy) decodes supported files, compares active
  sensor samples, and generates sRGB renders for capture-guided visual review. Unsupported
  camera/compression variants remain intact with a warning; no embedded-preview verification.
- Video: MOV, MP4, M4V, AVI, MTS. Byte matching and optional metadata comparison.
- Sidecars: AAE and XMP. Always preserved; same-directory, same-stem association.
- RAW/image companions are associated by local stem and both are retained by default.
  An explicit choice within a shared reviewed group can omit an alternate; sidecars
  and Live Photo components remain protected.
- Live Photo components are associated by content identifier when ExifTool provides
  one. Same-directory, same-stem image/video pairs without IDs are explicitly
  labeled `possible_live_photo`. Both components are retained.

The manifest's `related_files` lists these relationships. Sidecars follow a local
companion into its export directory, including when filename collisions require
renaming. Embedded HEIC auxiliary data and RAW bytes remain intact because files
are copied or linked without re-encoding.

## Useful options

```sh
# Fully read-only preview: no report directory and no cache reads or writes.
photo-dedup scan "/Volumes/Photos" --dry-run

# Reduce concurrent disk access on a spinning disk.
photo-dedup scan "/Volumes/Photos" --workers 1 --output results/hdd

# Compare all pixels instead of small previews; may use substantial RAM.
photo-dedup scan "/Volumes/Photos" --full-images --output results/full

# Change the review threshold, bypass the cache, or exclude an earlier export.
photo-dedup scan "/Volumes/Photos" --threshold 0.01 --no-cache \
  --exclude "/Volumes/Photos/Curated" --output results/strict

photo-dedup scan --help
photo-dedup export --help
```

Report and export destinations must be new directories. Reports may sit inside an
input directory and are automatically excluded during that scan. Exports must be
outside input directories. Exclude old report/export directories on future scans
if they are inside the next input root. Overlapping roots are inventoried once;
symbolic links are recorded but not followed. Media files with unsupported filename
extensions are listed as ignored, even if their bytes are a supported format.

`--cache PATH` selects a SQLite cache (default `.dedup-cache.sqlite3` in the current
working directory). Cache entries are validated with absolute path, device, inode,
size, modification/change times, and inspection-engine versions. Every scan
rebuilds matching decisions with its current options. Missing files disappear from
the new manifest; obsolete cache entries may remain on disk. Delete the cache to
reclaim it, or use `--no-cache`. Legacy pickle caches are never loaded.

Progress goes to stderr; summaries go to stdout. Exit codes: `0` completed without
per-file warnings/errors; `1` scan completed with per-file warnings/errors and a
report to inspect; `2` invalid input or operation failed; `130` interrupted. A missing
optional ExifTool produces a capability notice rather than a failed scan.

## Output layout

```text
scan-report/
  summary.txt
  files.csv
  manifest.json

export/
  curated/<camera>/<YYYY-MM or Unknown-date>/...
  duplicates/exact/<camera>/<date>/...
  review/<camera>/<date>/...
  summary.txt
  files.csv
  manifest.json
```

`manifest.json` schema version 1 contains roots, scan options, cache hits, and one
record per inventoried path. Records include status, confidence, explanation,
SHA-256 and file identity where available, group ID, direct `matched_to` reference,
metadata, warnings, and companion relationships. Exports add `export_path` to each
exported record. CSV cells are escaped to prevent spreadsheet formulas from file
metadata; JSON preserves the original text.

## Development

```sh
python -m pip install -e '.[test,build]'
python -m pytest -q
npm --prefix desktop run build
# Install Playwright's test browsers once, then run both browser engines.
npx --prefix desktop playwright install chromium webkit
npm --prefix desktop test
```

The package separates inspection, caching, matching, review decisions, persistent
sessions, reports, export, and transports. The desktop shell communicates with a
bundled worker through versioned JSON-lines over private stdio. The optional
browser launcher binds only to loopback, requires a random session token, and
serves the same interface. No hosted application service is required.

CLI exports without an overlay retain the original all-readable-media policy.
App exports add export_selected flags and saved review decisions to the complete
manifest; the CLI exporter honors those flags when reusing an app export manifest.
Review choices do not overwrite the original scan evidence. Clearing inspection
caches does not erase decisions or undo history.

The old `python iphone-photo-dedup.py INPUT...` invocation now runs the safe scan
workflow. It does not immediately create hard links as the original script did.
The original scripts, README, and supplied April 2020 local notes are archived
locally under `reference/`, which is excluded from source publication. Git history
also retains the original GitHub version.
Sample media and old pickle caches were not imported into project source.

## Layout templates and source cleanup

The desktop export screen offers layout presets and a custom relative template.
The default is `{camera}/{capture_month}/{stem}{ext}` beneath fixed `curated/` and
`review/` categories. CLI exports accept `--template '{year}/{month}/{stem}{ext}'`.
Supported tokens: `{camera}`, `{capture_month}`, `{year}`, `{month}`, `{day}`,
`{date}`, `{stem}`, `{ext}`, `{kind}`. Extensions include the dot. Templates must
contain `{stem}` in the filename and end with `{ext}`. No expressions, traversal,
absolute paths, or arbitrary metadata evaluation. Missing date components use
explicit Unknown values. Sidecars follow media; collision suffixes remain deterministic.

The completed export says **We don't delete anything** and provides links to the
original top-level input folders plus a paginated list of files without a retained
byte-identical counterpart. An omitted exact copy is distinguished from a visual
alternative with distinct bytes. The manifest and `source-cleanup.csv` preserve this
snapshot. It does not audit new or changed files after scanning, or count files
inside ignored directories. Verify the result and backup before manually cleaning up.

APFS, HFS+ and NTFS support file hard links; exFAT and FAT32 do not. **Check local
disks** lists local mounted volume capabilities without reading media or writing
probes. **Test hard links in this folder** explicitly creates and removes a tiny
probe in the chosen folder; access errors report unverified support. Hard links
provide new paths to retained data without copying media bytes. Directory entries,
reports and previews still use some space, and this is not a separate backup.
