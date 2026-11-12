# Photo Dedup application plan

## Product

A temporary, local workspace for consolidating years of loose iPhone, smartphone,
and SD-card imports from Mac and Windows hosts. The finished result is an ordinary
organized folder tree with portable reports, usable after the app and its catalog
are removed. Users verify that result and manually remove old import paths.

The app catalog is a database-backed view of source files and the proposed output
structure. It does not mount a virtual disk or implement a filesystem driver. Scan
records, duplicate evidence, review choices and proposed folder paths support the
short-lived consolidation task. The app is not intended to become a permanent
photo-library manager.

Reviewing and planning do not require hard-link support or sources on one volume.
Hard links are an efficient build method when retained sources and the output share
a compatible volume; independent copies support other layouts and filesystems.
Gathering imports onto one volume remains an optional efficiency strategy, with
its own copying/space costs. Neither the catalog nor its previews is a backup.
Source files must remain available until a real output has been built and verified.

First-run onboarding should explain the temporary workspace, preparation choices,
review decisions, the real output, and manual source cleanup. Unsupported/error
entries and omitted alternatives must be accounted for before source trees are
removed.

The catalog workflow below is the next planned milestone. The current app already
has SQLite scan records and decisions, cached previews, and output-path planning;
it still presents those mainly as scan sessions and duplicate groups. The proposed
folder browser, workspace completion/removal flow, and revised preparation screens
are not implemented yet.

## Next milestone: temporary catalog workflow

1. **Add imports.** Choose ordinary source folders. Show which drives need to remain
   connected. Explain the same-volume condition when choosing hard-link output,
   rather than requiring it to begin reviewing. Keep Apple Photos export guidance.
2. **Review the proposed library.** Provide a folder tree and thumbnail/list view
   derived from the chosen layout. Every virtual file points to an original source
   record. Expose duplicate alternatives, matching evidence, source locations and
   inclusion decisions in that context. Show excluded/unreadable items in a distinct
   source-items view; do not make them disappear from the accounting.
3. **Adjust the plan.** Saved inclusion choices and layout changes update the proposed
   tree and counts without renaming, moving or copying source media. Keep full-detail
   inspection available through the user's existing viewer. Preserve the focused
   comparison UI; general file management remains in Finder/Explorer.
4. **Build the real library.** Preview the exact output paths and choose hard links
   or independent copies, with capability/device/space checks. Freeze the plan used
   for the build and revalidate source contents. Preview and export must use the
   same path, collision and companion-preservation rules.
5. **Verify and finish.** Open the real output, review source-only items and saved
   cleanup evidence, then let the user close the completed workspace. Later edits
   to the catalog never silently update an existing output. Newly added imports
   require a new scan or an explicit rescan.
6. **Remove temporary workspace data.** Offer explicit removal of saved workspace
   records and previews. It must leave original media and completed outputs intact.
   Keep manifests, decisions and cleanup reports in the output so users can leave
   the app behind. Shared inspection-cache management is a separate storage action.

### Catalog model and reliability

- Keep physical source identity separate from logical output location. A source
  record stores its root, original path, file identity/stat data, content hash and
  inspection evidence; a planned entry references that record and a relative
  proposed output path. Equal hashes do not erase individual source records.
- Start by deriving directory nodes from planned-path prefixes. Add indexed/materialized
  directory records only if measured browsing/pagination performance needs them.
  Use the existing SQLite/service stack; no additional storage engine is required.
- Reuse `plan_export` for the proposed tree and actual build. The plan includes
  decisions, fixed review categories, templates, deterministic collision suffixes
  and preserved companions. Associate the build with the exact plan revision.
- Distinguish available, disconnected, missing and changed sources. Cached thumbnails
  may remain visible with an explicit availability label; they cannot establish that
  a source is still present or unchanged. Offer deliberate relinking/rescanning;
  changed evidence must not inherit stale approval without verification.
- Design for interrupted sessions and builds, not continuous background sync.
  Keep the source-cleanup checklist tied to a dated, completed output snapshot.

## Design references

Borrow established interaction patterns, not branding, assets, or screen layouts:

- [Gemini 2](https://macpaw.com/support/gemini/knowledgebase/gemini-2-smart-selection):
  separate identical files from similar files that require human review.
- [dupeGuru](https://dupeguru.voltaicideas.net/help/en/results.html):
  grouped results, a reference file, and comparison details.

Use an original, calm visual language: warm backgrounds, forest green controls,
photo-led group cards, explicit confidence labels, and clear action language.
Keep matching thresholds and filesystem details out of the primary workflow.

## Existing implementation foundation

1. Preserve scan evidence and add persistent, undoable review decisions.
2. Make export planning honor decisions and preserve companions and audit records.
3. Build React/TypeScript screens around the shared Python service.
4. Bundle Python, HEIC decoding, and ExifTool in a Tauri desktop app.
5. Verify the full workflow, recovery, keyboard operation, and compact windows.
6. Add GitHub Actions checks and native build/release automation.
7. Prepare local application artifacts for review before remote publication.

## Architecture

- React + TypeScript + Vite, Radix dialogs, original CSS, Lucide icons.
- Small Tauri 2 shell: native folder dialogs, file-manager reveal, single instance.
- Python worker with a versioned JSON-lines protocol over private stdio.
- SQLite sessions, review decisions, and undo history, separate from the rebuildable
  inspection cache. Scan manifests remain immutable.
- Paginated group responses and on-demand image previews. Large scans still use
  the current engine's in-memory classification; measure before claiming support
  for million-file libraries.
- Optional authenticated, loopback-only browser launcher for development and local
  use. No cloud service, login, telemetry, or photo upload.

## Release strategy

GitHub Actions handles tests, native builds on each target, engine smoke tests,
and draft GitHub Releases. Release workflows use the repository GITHUB_TOKEN;
Apple/Windows signing credentials belong in protected repository environments.
Public releases require platform signing setup and clean-machine testing.

Build Apple Silicon and Intel macOS packages separately and Windows x64 installers.
Start with direct release downloads. Automatic in-app updates are a later layer
that requires signed update metadata and an explicit update endpoint; GitHub
Releases alone is not an updater protocol.

All development work and samples stay inside this checkout. ChatGPT Sites is not
used for this application. The user’s current restriction requires explicit approval
before sending content to ChatGPT Sites or its associated remote repositories.

## Verification passes

1. Welcome screen: hierarchy, spacing, typography, folder selection, empty history.
2. Review: demo scan, confidence filters, group navigation, zoom, choices, undo.
3. Persistence and export: reload a session, preview decisions, verify outputs,
   handle changed sources, existing output directories, cancellation, and warnings.
4. Keyboard/accessibility: dialog focus, Escape, form labels, contrast, reduced motion.
5. Compact and large windows: responsive layout, long paths, overflow, pagination.
6. Packaging: launch the bundled engine without Python on PATH, run the native app,
   inspect its logs, and build a local installer.

## Deliberate limits

No direct Apple Photos library editing, permanent deletion, cloud sync, video frame
matching, or automatic removal of visually similar media. Supported RAW sample
comparison and generated renders are now included; the optional bounded deeper
photo pass uses existing NumPy/Pillow plus a small LibRaw wrapper. Similarity remains review evidence rather
than proof. The catalog has no hard-link requirement. Real output uses hard links
when appropriate or copies with their additional storage cost. Neither mode deletes
source files. No mounted filesystem, background folder watching, or general-purpose
Finder replacement is planned.


## Freshman CS student walkthrough — 2026-10-30

Three independent agents reviewed wording, workflow behavior, and actual desktop/
compact screenshots, using a first-year CS student as the target audience. This
was a heuristic walkthrough, not an observed study with a student. The default
preset path was judged approachable; advanced media evidence and template syntax
should remain optional.

The review identified and corrected these obstacles:

- Contradictory RAW onboarding and jargon-heavy headings: describe supported RAW
  capabilities accurately and use “organized library” in the primary screens.
- Ambiguous checkboxes and silent loss of changes: state that checked means include;
  guard close/navigation with Save, Discard or Keep reviewing; show save errors
  inside the dialog and restore keyboard focus when returning to the comparison.
- Misleading zoom percentages: show Fit/preview magnification, state the 1,600-pixel
  preview limit, and expose original-file access beside source-folder context.
- Invalid output-folder expectations: explain that the app creates a folder that
  does not exist yet, with a visible containing-folder action in the native app.
- Broken Custom selection and exposed syntax: explicit custom mode, focused editor,
  token explanations, presets first, and concrete paths in the export preview.
- Success/cleanup hierarchy: put the new-library action first, show the saved
  checklist's destination/time and limits before folder actions, and use an amber
  source-item summary. Changing choices/layout affects the next export only.
- Conflicting feedback: clear stale selection-saved messages and keep comparison
  feedback inline so toasts cannot obscure modal actions.

A useful real-user follow-up is to have one student prepare a small mixed import,
choose one of two similar photos, create a library without pre-creating its folder,
and explain which source items would be lost by deleting the imports. Success means
they can explain the hard-link/shared-content behavior and identify source-only
items without coaching. That observation has not been performed.
