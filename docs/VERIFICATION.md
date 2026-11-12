# Current verification — 0.3.1 / 2026-10-30

Three independent agents reviewed copy, workflow behavior, and actual desktop and
compact screenshots using a freshman CS student as the intended audience. This
was a heuristic review, not an observed human usability study. Findings and the
real-user follow-up scenario are recorded in APP_PLAN.md.

- 61 Python tests pass, including saved export destination/time and partial disk
  discovery when a volume disconnects. Existing RAW, matching, export, cache,
  companion and cleanup checks remain green.
- 26 Chromium/WebKit scenarios pass. New checks cover unsaved review selections
  through Next, Escape, outside click and close; saving versus discarding; visible
  save errors; restored keyboard focus; custom-layout editor focus; and narrow
  screens with all modal actions visible and no stale toast covering them.
- WCAG-tagged automated checks pass on the tested screens. This is not full
  accessibility conformance certification.
- A second visual review confirmed the output action, saved-checklist scope,
  source-only warning, source context, custom-layout guidance, and unsaved actions
  are readable and discoverable at desktop and narrow sizes.
- TypeScript/Vite builds, workflow validation and whitespace checks pass.
- The local Apple Silicon app/installer is rebuilt for 0.3.1. It uses development
  signing; public signing/notarization and native Windows/Intel checks remain
  release work. No new runtime dependency was added for this UX pass.

Primary fixes: plain-language onboarding with correct RAW capabilities, explicit
include/leave-at-source meaning, saved-selection guard, scaled-preview labels,
nonexistent output-folder instructions, working custom mode, export-only settings
scope, early library access and source warnings, dated snapshot limits, and
readable recovery when a disk disappears during discovery.

Current artifact: dist/Photo-Dedup-0.3.1-macos-arm64.dmg.

---

# Previous verification — 0.3.0 / 2026-10-30

This version adds decoded RAW sample evidence and generated renders, an optional
bounded deeper photo pass, output layout templates, local volume capability checks
and an explicit folder probe, and a saved source cleanup review.

- 60 Python tests pass, including real LibRaw decoding of synthetic Bayer DNGs,
  sample equality with different file metadata, unsupported RAW retention, cache
  reuse, deeper image matching without names/dates, video exclusion, direct visual
  representatives, template validation/sidecars, hard-link probe cleanup, denied
  access classified as unknown, read-only volume reporting, and saved cleanup evidence.
- 24 Chromium/WebKit scenarios pass, covering disk discovery and an explicit local folder test,
  custom templates and invalid traversal, concrete output paths, saved source cleanup,
  generated RAW previews and sample evidence, first-run guidance, compact layouts,
  keyboard operation and WCAG-tagged audits. This is not full accessibility conformance certification.
- Inspected actual screenshots of disk checks, custom paths, desktop and compact
  cleanup, and the RAW comparison dialog. Fixed read-only volume reporting, long
  source-path hierarchy, RAW detail spacing, and the stretched export explainer.
- TypeScript/Vite, GitHub workflow validation and whitespace checks pass.
- The bundled engine smoke test runs without Python on PATH and exercises ExifTool,
  normal image scans/previews, real RAW decoding/previews/sample evidence, a custom
  same-volume hard-link export and its persisted cleanup review/CSV.
- Bundled native dependencies support the advertised macOS 14 minimum. Native
  Apple Silicon application build and local DMG verification pass.

Dependency impact: rawpy 0.27.1 / LibRaw 0.22.1 adds one runtime package and reuses
NumPy. The installed decoder is about 5.5 MiB; the compressed installer grew from
47.5 MiB to about 50.7 MiB. No machine-learning stack or video decoder was added.

Current artifact: dist/Photo-Dedup-0.3.0-macos-arm64.dmg. Development signing only;
public signing/notarization and native Windows/Intel runtime checks remain release
work. RAW camera/compression variants depend on LibRaw support. Deeper matching is
bounded and can miss crops, rotations, or weak candidates; suggestions remain review
items. Cleanup is saved export evidence, not a fresh audit of later source changes
or uncounted files inside ignored directories. The app never deletes source files.

---

# Previous verification — 0.2.1 / 2026-10-30

The first-run experience now explains the actual niche: old iPhone, smartphone,
and SD-card imports accumulated across Mac and Windows hosts, gathered onto one
hard-link-capable volume and consolidated into an opinionated camera/month tree.
The user verifies the result and manually removes old import paths afterward.

- 39 Python tests pass, including read-only source-location preparation checks,
  default same-volume hard-link exports, explicit unexported-entry counts, retained
  output after a synthetic source link is removed, and unsupported-hard-link errors
  without an automatic copy fallback.
- 18 Chromium/WebKit checks pass, including the existing scan/review/export flows
  and first-run guidance at desktop, compact browser, and 760×650 native-minimum sizes.
- First-run guide steps pass automated WCAG-tagged checks. Completion persists;
  returning users can reopen the guide from help. This is not full accessibility
  conformance certification.
- Reviewed actual screenshots of preparation, cleanup, and the revised welcome.
- TypeScript/Vite build and workflow validation pass.
- Rebuilt the portable 0.2.1 worker and macOS application. Packaged-worker smoke
  checks verify the current version, source preparation, default hard-link plan,
  unhandled-entry counts, scanning, matching, and previews.
- The local 0.2.1 macOS arm64 installer passes disk-image checksum verification.

Current artifact: dist/Photo-Dedup-0.2.1-macos-arm64.dmg. Development signing only;
public signing/notarization and other-platform builds remain release work.
Screenshots are kept locally under local-validation/onboarding-review/.

The original baseline below records the earlier 0.2.0 behavior. Its GUI copy default
has been superseded by the explicit same-volume hard-link consolidation workflow.

---

# Original 0.2.0 baseline — 2026-10-29

## Verified

- 36 Python tests pass: classification, arithmetic correctness, HEIC/orientation,
  cache invalidation, complete inventory, safe export, changed sources, filename
  collisions, companion preservation, persistent decisions, undo, cancellation,
  single workspace ownership, and marked partial-export recovery.
- 14 browser integration checks pass: seven scenarios each in Chromium and WebKit.
- Automated WCAG-tagged checks report no violations on the welcome screen, results,
  zoomed comparison dialog, and export confirmation. This is not a claim of complete
  WCAG conformance or a substitute for testing with assistive technology users.
- TypeScript/Vite production build passes. The frontend dependency audit reported
  zero known vulnerabilities after the final dependency installation.
- GitHub workflows pass actionlint validation. No remote workflow has been run.
- A self-contained portable Python engine runs with Python absent from PATH, from
  an unrelated working directory: ExifTool, SQLite sessions, demo scan, matching,
  and image previews pass.
- Every bundled macOS Mach-O library checked supports the declared macOS 14 minimum.
- Native macOS application launches and starts its bundled worker; data stays in
  local-validation/app-data during development launch.
- Ad-hoc native application signature passes codesign --verify --deep --strict.
- Local macOS arm64 DMG passes hdiutil verification. Mounted read-only inside this
  checkout, its contained application also passes signature verification.
- Installer SHA-256 checksum is saved next to the DMG.

## UI/UX passes and resulting corrections

1. Welcome and onboarding: clear folder selection, disabled empty scan, validation,
   synthetic demo, understandable privacy and source-preservation language.
2. Review workflow: real scan results, confidence filters, comparison, choice saving,
   undo, reopening sessions, and choices applied to a verified export.
3. Readability/accessibility: increased secondary text contrast and size, restored
   dialog focus on close, visible keyboard focus, and reduced-motion support.
4. Comparison: sharpened card previews, fixed clipped file details, synchronized
   zoom/scroll, keyboard-scrollable preview regions, clear capture dates, plain
   language explanations, and expandable technical evidence.
5. Window sizes and empty states: checked 1280px, 820px, and 390px widths; added compact
   navigation; verified zero-file scans, search without matches, and warning lists.
6. Export: preview validation, existing-destination errors, copy default, final
   confirmation, complete reports, source verification, and instant-job completion.
7. Local transport: missing tokens, foreign origins, and invalid hosts are rejected.

## Installer and release status

Local artifact: dist/Photo-Dedup-0.2.0-macos-arm64.dmg (approximately 48 MB).

This is a development build with ad-hoc signing. Developer ID signing, Apple
notarization, Windows signing, approved source upload, and actual GitHub-hosted
builds remain publication steps. Windows and Intel macOS installers have not been
built or tested locally; their workflows are prepared for the appropriate runners.

Finder-driven DMG layout initially timed out. The local packaging path now writes
Finder layout metadata directly and creates/verifies the disk image with hdiutil.
The release workflow uses the same packaging method; it requires no Finder
AppleEvent permission. The initial Homebrew-based Python bundle was replaced with
a portable runtime so the application does not inherit this host's macOS 26 minimum.

UI verification used separate local browsers after the in-app browser connection
failed. Native window screenshots were unavailable without system capture access;
no new system permission was requested. Native launch and worker operation were
smoke-tested, while the interaction checks ran in Chromium and WebKit.

## Deliberate limits

- No permanent deletion, direct Photos library editing, or space reclamation.
- Similarity and metadata agreement remain review evidence, not confirmed equality.
- No video-frame/audio comparison or rendered RAW preview.
- Large-library memory/performance benchmarking and clean-machine testing on each
  release platform are still needed before broad production claims.
- Updates are distributed through GitHub Releases; in-app automatic updating is not
  configured in this version.

Application source, synthetic fixtures, copied samples, reports, and installers
were kept in this checkout during application development. No NAS access
was used for this application work. No source push or release publication occurred.
