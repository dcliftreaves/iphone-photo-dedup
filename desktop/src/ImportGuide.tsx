import { useEffect, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FolderTree,
  HardDrive,
  Layers2,
  ShieldCheck,
  X,
} from "lucide-react";

import { DiskCheck } from "./DiskCheck";

export const GUIDE_KEY = "photo-dedup-import-guide-v2";
const steps = [
  {
    label: "Why it exists",
    title: "Years of imports. One canonical library.",
    icon: Layers2,
  },
  {
    label: "Prepare your media",
    title: "Gather the imports on one volume.",
    icon: HardDrive,
  },
  {
    label: "Review & build",
    title: "An opinionated structure, not a folder mirror.",
    icon: FolderTree,
  },
  {
    label: "Finish the consolidation",
    title: "We don't delete anything",
    icon: ShieldCheck,
  },
];

export function ImportGuide({
  open,
  onClose,
  returnFocus,
}: {
  open: boolean;
  onClose: () => void;
  returnFocus: () => void;
}) {
  const [step, setStep] = useState(0);
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (open) setStep(0);
  }, [open]);
  useEffect(() => {
    if (open) heading.current?.focus();
  }, [open, step]);
  const current = steps[step];
  const Icon = current.icon;
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content
          className="import-guide"
          aria-describedby="guide-description"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            heading.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            returnFocus();
          }}
        >
          <div className="guide-topline">
            <span>PHOTO DEDUP · IMPORT CONSOLIDATION</span>
            <Dialog.Close
              className="icon-button"
              aria-label="Close import guide"
            >
              <X size={20} />
            </Dialog.Close>
          </div>
          <div className="guide-layout">
            <nav className="guide-steps" aria-label="Import guide steps">
              {steps.map((item, index) => (
                <button
                  key={item.label}
                  aria-current={index === step ? "step" : undefined}
                  onClick={() => setStep(index)}
                  className={index === step ? "current" : ""}
                >
                  <span>{index < step ? <Check size={14} /> : index + 1}</span>
                  {item.label}
                </button>
              ))}
              <p>
                For loose iPhone, smartphone, and SD-card imports accumulated
                across Macs and Windows PCs.
              </p>
            </nav>
            <div className="guide-main">
              <Icon size={29} className="guide-icon" />
              <Dialog.Title ref={heading} tabIndex={-1}>
                {current.title}
              </Dialog.Title>
              <Dialog.Description id="guide-description">
                {step === 0
                  ? "Consolidate old media imports into one camera-and-date archive while leaving source file contents unchanged."
                  : step === 1
                    ? "The efficient workflow needs every source folder and the new library on the same hard-link-capable filesystem."
                    : step === 2
                      ? "Scan the gathered folders, inspect uncertain matches, then preview and build a new library outside those source folders."
                      : "Building the library does not remove the old imports. You finish that step yourself after checking the result."}
              </Dialog.Description>
              {step === 0 && (
                <div className="guide-body">
                  <p>
                    The same iPhone media may have been downloaded many times: a
                    Windows import, a later Mac import, JPEG conversions of HEIC
                    originals, and separate copies of videos or edits. SD-card
                    transfers can accumulate the same way.
                  </p>
                  <p>
                    Photo Dedup brings those loose files into one opinionated
                    structure. It compares whole-file fingerprints, image
                    appearance where supported, and capture metadata.{" "}
                    <strong>
                      Similar appearance or matching metadata does not prove two
                      files are interchangeable.
                    </strong>
                  </p>
                  <div className="guide-callout">
                    <strong>This is an archive-consolidation workflow.</strong>
                    <span>
                      It does not edit your media, clean an Apple Photos
                      library, or automatically delete old imports. Videos and
                      RAW files currently use byte and metadata matching; their
                      content is not visually verified.
                    </span>
                  </div>
                </div>
              )}
              {step === 1 && (
                <div className="guide-body">
                  <ol>
                    <li>
                      Copy the old Mac, Windows, phone, and SD-card imports onto
                      one working volume. Keep the source folders separate.
                    </li>
                    <li>
                      Bring the companions too: Live Photo MOV files, AAE/XMP
                      edit files, and RAW/JPEG pairs. Keep their names and local
                      folder relationships intact.
                    </li>
                    <li>
                      Choose a new output folder beside the imports, on that
                      same volume. Do not select their common parent as an
                      input.
                    </li>
                  </ol>
                  <pre className="guide-tree">{`Media volume/\n  Imports/                 ← scan this folder\n    Old Mac downloads/\n    Windows downloads/\n    SD-card transfers/\n  Canonical library/       ← created by the app`}</pre>
                  <p className="guide-small">
                    One physical disk can contain multiple volumes; hard links
                    cannot cross between them. The volume must support hard
                    links. APFS, Mac OS Extended (HFS+) and NTFS support them;
                    exFAT and FAT32 do not. Cross-disk or unsupported-volume
                    workflows require independent copies and enough space for
                    another copy of the selected media.
                  </p>
                  <DiskCheck />
                  <p className="guide-small">
                    If the media is in Apple Photos, export unmodified originals
                    into ordinary folders first. Work from a verified backup; a
                    hard-linked library on the same disk is not a separate
                    backup.
                  </p>
                </div>
              )}
              {step === 2 && (
                <div className="guide-body">
                  <ol>
                    <li>
                      <strong>Scan imports.</strong> Identical bytes can use a
                      suggested representative. Review similar photos and
                      metadata-only matches; keep both when uncertain.
                    </li>
                    <li>
                      <strong>Choose what to retain.</strong> Suggestions help
                      review repeated imports, but do not certify which download
                      was the original.
                    </li>
                    <li>
                      <strong>Build with hard links.</strong> Retained files get
                      new names and locations without duplicating their media
                      data. File contents and embedded metadata are not
                      re-encoded.
                    </li>
                  </ol>
                  <p>
                    You can choose a layout preset or build a template from
                    camera, capture date, filename and other tokens. Curated and
                    review categories remain separate.
                  </p>
                  <p>
                    Hard links build a new way to view your existing files: a
                    new directory on the same volume, without copying the media
                    data. Think of each retained file as a copy that takes no
                    additional media space. Folders and reports use a little
                    extra space. Only retained files appear there; editing
                    shared content through either path changes both.
                  </p>
                  <pre className="guide-tree">{`Canonical library/\n  curated/<camera>/<YYYY-MM>/…\n  review/<camera>/<YYYY-MM>/…\n  manifest.json   files.csv   summary.txt`}</pre>
                  <p className="guide-small">
                    Unresolved similar and possible matches remain in review/.
                    Missing dates use Unknown-date. Sidecars and associated
                    companions are preserved where identified; inspect the
                    manifest. Unsupported files and read errors are listed, not
                    exported.
                  </p>
                  <div className="guide-callout">
                    <strong>Hard links share file contents.</strong>
                    <span>
                      Editing a retained file through either path can affect
                      both. Removing an old source link leaves a retained
                      library link intact.
                    </span>
                  </div>
                </div>
              )}
              {step === 3 && (
                <div className="guide-body">
                  <p>
                    After export, the source cleanup review links to your
                    original folders and lists files without a retained
                    byte-identical counterpart. Open them in Finder or Explorer
                    before deciding what to remove.
                  </p>
                  <ol>
                    <li>
                      <strong>Inspect the actual output.</strong> Open curated/
                      and review/, spot-check photos and videos, and check
                      camera/date placement.
                    </li>
                    <li>
                      <strong>Account for everything left behind.</strong> Read
                      files.csv, summary.txt, and Skipped & warnings. Omitted
                      alternatives, unsupported files, and errors may exist only
                      in the old import folders. Preserve anything you still
                      need before deleting those folders.
                    </li>
                    <li>
                      <strong>
                        Then manually remove the old source files.
                      </strong>{" "}
                      Use Finder or Explorer after verifying the retained
                      library and your backup. The app never performs this
                      cleanup for you.
                    </li>
                  </ol>
                  <div className="guide-callout">
                    <strong>
                      Consolidation is complete after your cleanup.
                    </strong>
                    <span>
                      With hard links, selected media remains accessible through
                      the canonical library when the old source paths are
                      removed. Space from unwanted copies is reclaimed only when
                      their last links are removed; items in Trash may still
                      occupy space.
                    </span>
                  </div>
                  <p className="guide-small">
                    You can reopen this guide anytime from Guide & help. Closing
                    the guide does not confirm that your media is prepared or
                    that it is safe to delete source folders.
                  </p>
                </div>
              )}
            </div>
          </div>
          <div className="guide-footer">
            <span>
              Step {step + 1} of {steps.length}
            </span>
            <div>
              <button
                className="button secondary"
                disabled={step === 0}
                onClick={() => setStep(step - 1)}
              >
                <ArrowLeft size={15} /> Back
              </button>
              {step < steps.length - 1 ? (
                <button
                  className="button primary"
                  onClick={() => setStep(step + 1)}
                >
                  Next <ArrowRight size={15} />
                </button>
              ) : (
                <button className="button primary" onClick={onClose}>
                  Ready to prepare my media <Check size={15} />
                </button>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
