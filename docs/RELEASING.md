# Building and releasing Photo Dedup

GitHub Actions is the build service and GitHub Releases is the download host. No
separate application server, website hosting, or release-upload service is needed.
The application runs locally and does not upload photos.

## Local development

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,build]'
npm ci --prefix desktop
npm --prefix desktop run build
photo-dedup ui --data-dir .photo-dedup
```

The browser launcher uses a random per-process token and binds only to 127.0.0.1.
The desktop application uses private standard-input/output messaging instead of
an HTTP server. Browser folder selection accepts typed local paths; the desktop
app provides native folder pickers and folder drag-and-drop.

For a desktop build, install Rust and the native platform development tools, then:

```sh
python scripts/build_engine.py
python scripts/check_macos_runtime.py
python scripts/smoke_engine.py
npm --prefix desktop run tauri -- build --bundles app
python scripts/package_macos.py  # macOS: create the DMG without Finder automation
```

Build the engine with the same OS and architecture as the desktop shell. On macOS,
use a portable Python distribution (or setup-python on GitHub Actions), not a
Homebrew interpreter tied to a newer OS than the advertised macOS 14 minimum.
The runtime checker rejects bundled libraries requiring a newer OS.

Use `python scripts/run_desktop.py` in this checkout to keep all native app data
under local-validation/app-data.

The engine includes Python, Pillow, NumPy, pillow-heif/native codecs, and ExifTool.
macOS supplies Perl for ExifTool; the Windows distribution includes its Perl
runtime. ExifTool 13.60 is pinned and downloaded from its official GitHub source
and official SourceForge Windows distribution. Its source/README and package
license notices travel with the engine. Downloads/build caches stay under .tools.

For development in this checkout, set PHOTO_DEDUP_DATA_DIR to an absolute path
inside the checkout when launching the native app. The installed application uses
the normal per-user app-data directory. Never package .photo-dedup, local-validation,
personal photos, caches, or credentials into the application.

## GitHub checks

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch:

- Python tests on Linux, Apple Silicon macOS, and Windows.
- TypeScript/Vite production build and dependency audit.
- Chromium and WebKit integration tests using synthetic demo media only.
- Failure screenshots and traces retained as short-lived workflow artifacts.

These workflow definitions have been prepared locally. They do not represent a
successful remote CI run until an approved push has triggered GitHub Actions.

## Draft release automation

`.github/workflows/release.yml` builds separate installers for:

- Apple Silicon macOS (`macos-15`, `aarch64-apple-darwin`).
- Intel macOS (`macos-15-intel`, `x86_64-apple-darwin`).
- Windows x64 (`windows-2025`, `x86_64-pc-windows-msvc`).

The workflow checks version consistency, tests the engine, builds the frontend,
freezes the Python worker, verifies the packaged worker independently of system
Python, checks macOS compatibility, and attaches installers plus SHA-256 checksums to a **draft prerelease**.
The Tauri action builds the native bundles; our small upload script uses GitHub CLI
to manage only draft releases. macOS DMGs are created without Finder automation.
The upload script refuses local execution and refuses to modify published releases.

It uses the repository GITHUB_TOKEN with contents:write. There is no personal
access token to create or maintain. Public download publication remains a separate
review step. The workflow does not configure automatic in-app updates; users get
new versions from GitHub Releases.

Create a protected GitHub environment named `release`, with required reviewers.
Manual dispatch with `signed=false` produces development-only draft artifacts.
Tag pushes request signed builds and fail if signing credentials are missing.
A manually supplied tag must match every package version; no arbitrary commit/ref
is interpolated into shell commands.

## One-time signing setup

macOS needs a Developer ID Application identity and an App Store Connect API key.
Put these secrets in the protected `release` environment:

- APPLE_CERTIFICATE: base64-encoded exported .p12 certificate.
- APPLE_CERTIFICATE_PASSWORD: password for that export.
- APPLE_SIGNING_IDENTITY: full Developer ID Application identity.
- APPLE_API_ISSUER, APPLE_API_KEY: App Store Connect issuer and key ID.
- APPLE_API_KEY_CONTENT: contents of the private .p8 API key.

The runner imports the certificate into a temporary keychain before signing the
Python engine. Tauri signs the outer app and submits it for Apple notarization.
Credentials are not stored in the repository or printed by our setup script.

Windows signing supports an exportable code-signing certificate through
WINDOWS_CERTIFICATE_BASE64 and WINDOWS_CERTIFICATE_PASSWORD. The runner imports
it into the current-user certificate store, configures SHA-256 signing and a
timestamp service, and deletes the temporary PFX. If the certificate provider
requires hardware or cloud signing, replace that provisioning step with the
provider's official integration; a certificate thumbprint alone is insufficient.

GitHub can automate signing, but it cannot issue these identities. Without valid
credentials, artifacts remain development builds and should not be presented as
normal, publicly installable signed software.

## Approval and publication

The user's global AGENTS.md requires explicit approval before uploading source,
builds, or project files. Do not push commits/tags, dispatch upload workflows,
submit for notarization, or publish a release without destination/content approval.
Prepare and test everything locally first. That restriction also applies to private
repositories, Actions artifacts, and draft releases.

After approval, push the reviewed source to dcliftreaves/iphone-photo-dedup, allow
Checks to pass, and manually dispatch the draft-release workflow. Install and test
the signed artifacts on clean machines before changing the draft into a public
release. Windows and Intel macOS builds require those target runners/machines;
local Apple Silicon verification is not evidence that they already pass.
