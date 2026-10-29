# Local verification — 2026-10-29

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
