# Photo Dedup application plan

## Product

A private desktop application for repeated photo imports: select folders, scan,
compare matches, save review choices, and create an organized library. Source
files remain in place. Export is not advertised as reclaiming source disk space.

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

All work and samples stay inside this checkout. Pushing source, uploading artifacts,
Apple notarization, and publishing releases require explicit user approval before
those remote actions. Local build scripts and workflow definitions do not grant it.

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
matching, or rendered RAW comparison. Similarity remains review evidence rather
than proof. Hard links are an advanced option; GUI exports default to copies.
