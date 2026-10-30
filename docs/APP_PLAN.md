# Photo Dedup application plan

## Product

A local consolidator for years of loose iPhone, smartphone, and SD-card imports
from Mac and Windows hosts. Users gather imports on one hard-link-capable volume,
review repeated downloads, and build an opinionated camera/month directory tree.
The app leaves source contents and paths in place; users verify the result and
manually remove the old import paths to finish. Unsupported/error entries and
omitted alternatives must be accounted for before source trees are removed.

First-run onboarding explains why the tool exists, how to prepare inputs, what the
canonical output contains, and how to finish. The guide stays available from help.

## Design references

Borrow established interaction patterns, not branding, assets, or screen layouts:

- [Gemini 2](https://macpaw.com/support/gemini/knowledgebase/gemini-2-smart-selection):
  separate identical files from similar files that require human review.
- [dupeGuru](https://dupeguru.voltaicideas.net/help/en/results.html):
  grouped results, a reference file, and comparison details.

Use an original, calm visual language: warm backgrounds, forest green controls,
photo-led group cards, explicit confidence labels, and clear action language.
Keep matching thresholds and filesystem details out of the primary workflow.

## Implementation order

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
than proof. Hard links are the same-volume default; copies are a less efficient
fallback that requires additional storage. Neither mode deletes source files.


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
