import { test as base, expect, Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { spawn, execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { resolve, join } from "node:path";

const root = resolve(import.meta.dirname, "../..");
const test = base.extend<{}, { appServer: { url: string; data: string } }>({
  appServer: [
    async ({}, use) => {
      await mkdir(join(root, "local-validation"), { recursive: true });
      const directory = await mkdtemp(
        join(root, "local-validation", "ui-pass-"),
      );
      const python =
        process.platform === "win32"
          ? join(root, ".venv/Scripts/python.exe")
          : join(root, ".venv/bin/python");
      const child = spawn(
        python,
        [
          "-m",
          "photo_dedup.web",
          "--data-dir",
          join(directory, "data"),
          "--port",
          "0",
          "--no-open",
        ],
        { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
      );
      let output = "";
      const url = await new Promise<string>((accept, reject) => {
        const timeout = setTimeout(
          () => reject(new Error(`Local app did not start: ${output}`)),
          20000,
        );
        child.stdout.on("data", (data) => {
          output += data.toString();
          const match = output.match(/Local review app: (\S+)/);
          if (match) {
            clearTimeout(timeout);
            accept(match[1]);
          }
        });
        child.stderr.on("data", (data) => {
          output += data.toString();
        });
        child.on("exit", (code) => {
          clearTimeout(timeout);
          reject(new Error(`Server exited (${code}): ${output}`));
        });
      });
      await use({ url, data: directory });
      child.kill("SIGINT");
      await new Promise<void>((accept) => {
        child.once("exit", () => accept());
        setTimeout(() => {
          child.kill("SIGTERM");
          accept();
        }, 5000).unref();
      });
    },
    { scope: "worker" },
  ],
});

async function startDemo(page: Page, url: string) {
  await page.goto(url);
  await page.getByRole("button", { name: "Close import guide" }).click();
  await page
    .getByRole("button", { name: "Take a look with demo photos" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Review your imports." }),
  ).toBeVisible();
  await expect(page.locator(".group-card")).toHaveCount(3);
  await expect(page.locator(".group-card img")).toHaveCount(3);
}

test("pass 1: welcome screen, folder validation, and keyboard help", async ({
  page,
  appServer,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(appServer.url);
  await page.getByRole("button", { name: "Close import guide" }).click();
  await expect(
    page.getByRole("heading", { name: "One home for years of imports." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Scan imports" }),
  ).toBeDisabled();
  const welcomeAudit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(welcomeAudit.violations.map((v) => v.id)).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "01-welcome.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Guide & help" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Guide & help" }),
  ).toBeFocused();
  await page
    .getByLabel("Folder path")
    .fill(join(appServer.data, "missing-folder"));
  await page.getByRole("button", { name: "Choose a folder" }).click();
  await page.getByRole("button", { name: "Scan imports" }).click();
  await expect(page.getByRole("alert")).toContainText("Folder not found");
});

test("pass 2: compare, save, undo, reopen, and verified export", async ({
  page,
  appServer,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await startDemo(page, appServer.url);
  await page.screenshot({
    path: join(appServer.data, "02-results.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Similar", exact: true }).click();
  await expect(page.locator(".group-card")).toHaveCount(2);
  await page.locator(".group-card").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("checkbox")).toHaveCount(2);
  await expect(page.getByRole("dialog").locator("img")).toHaveCount(2);
  await page.getByRole("checkbox").nth(1).uncheck();
  await page.screenshot({
    path: join(appServer.data, "03-comparison.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page
    .getByRole("button", { name: "Save selection", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".reviewed-badge")).toHaveCount(1);
  await page.reload();
  await expect(page.locator(".reviewed-badge")).toHaveCount(1);
  await page.getByRole("button", { name: "Undo last review choice" }).click();
  await expect(page.locator(".reviewed-badge")).toHaveCount(0);
  await page.getByRole("button", { name: "Similar", exact: true }).click();
  await page.locator(".group-card").first().click();
  await page.getByRole("checkbox").nth(1).uncheck();
  await page
    .getByRole("button", { name: "Save selection", exact: true })
    .click();
  await page
    .getByRole("main")
    .getByRole("button", { name: "Create library", exact: true })
    .click();
  const destination = join(appServer.data, "My organized photos");
  await page.getByLabel("New library folder").fill(destination);
  await page.getByRole("button", { name: "Preview my library" }).click();
  await expect(
    page.getByRole("heading", { name: "Here’s what will happen" }),
  ).toBeVisible();
  await page.screenshot({
    path: join(appServer.data, "04-export-preview.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Create my library" }).click();
  await expect(
    page.getByRole("heading", { name: "Your organized library is built." }),
  ).toBeVisible();
  const manifest = JSON.parse(
    await readFile(join(destination, "manifest.json"), "utf8"),
  );
  expect(manifest.records.filter((r: any) => r.export_path)).toHaveLength(4);
  expect(manifest.records).toHaveLength(7);
  expect(manifest.export.mode).toBe("hardlink");
  await expect(page.locator(".cleanup-next")).toContainText(
    "manually remove the old import files",
  );
  await expect(page.locator(".cleanup-review")).toContainText(
    "need attention before deleting this folder",
  );
  await expect(page.locator(".cleanup-next")).toContainText(
    "Removing an old source path leaves its retained library link intact",
  );
  expect(errors).toEqual([]);
});

test("pass 3: accessibility, responsive layouts, empty results, and warnings", async ({
  page,
  appServer,
}) => {
  await startDemo(page, appServer.url);
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    result.violations.map((v) => ({
      id: v.id,
      description: v.description,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.setViewportSize({ width: 820, height: 760 });
  await page.screenshot({
    path: join(appServer.data, "05-compact.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: join(appServer.data, "06-mobile.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.setViewportSize({ width: 1280, height: 900 });
  await page
    .getByRole("textbox", { name: "Search filenames and paths" })
    .fill("no-such-photo");
  await expect(
    page.getByRole("heading", { name: "No matches for that search." }),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "Search filenames and paths" })
    .fill("");
  await page.getByRole("button", { name: "Skipped & warnings" }).click();
  await expect(
    page.getByRole("heading", { name: "1 entries to know about" }),
  ).toBeVisible();
  await expect(page.locator(".issue-row")).toContainText(
    "Unsupported file extension",
  );
});

test("pass 4: comparison keyboard focus, zero selection, zoom, and image details", async ({
  page,
  appServer,
}) => {
  await startDemo(page, appServer.url);
  const first = page.locator(".group-card").first();
  await first.click();
  await expect(page.getByRole("dialog").locator("img")).toHaveCount(2);
  await page.getByRole("checkbox").nth(0).uncheck();
  await page.getByRole("checkbox").nth(1).uncheck();
  await expect(
    page.getByRole("button", { name: "Save selection", exact: true }),
  ).toBeDisabled();
  await page.getByRole("slider", { name: "Preview zoom" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(
    page.getByRole("slider", { name: "Preview zoom" }),
  ).not.toHaveValue("1");
  await expect(
    page.getByRole("dialog").locator(".file-details .file-path"),
  ).toHaveCount(2);
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(audit.violations.map((v) => v.id)).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "07-detail-and-zoom.png"),
    fullPage: true,
    animations: "disabled",
  });
  await page.keyboard.press("Escape");
  await expect(
    page.getByRole("heading", { name: "Your selection has not been saved" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Discard changes", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(first).toBeFocused();
});

test("pass 5: empty folders, completed zero-file scans, and compact navigation", async ({
  page,
  appServer,
}) => {
  const empty = join(appServer.data, "Empty folder");
  await mkdir(empty, { recursive: true });
  await page.goto(appServer.url);
  await page.getByRole("button", { name: "Close import guide" }).click();
  await page.getByLabel("Folder path").fill(empty);
  await page.getByRole("button", { name: "Choose a folder" }).click();
  await expect(page.locator(".source-preparation")).toContainText(
    "on the same volume",
  );
  await page.getByRole("button", { name: "Scan imports" }).click();
  await expect(
    page.getByRole("heading", {
      name: "No matches found with these scan settings.",
    }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Toggle navigation" }).click();
  await expect(
    page.getByRole("button", { name: "Skipped & warnings" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Skipped & warnings" }).click();
  await expect(page.getByText("No skipped files or warnings.")).toBeVisible();
  await page.screenshot({
    path: join(appServer.data, "08-empty-library.png"),
    fullPage: true,
    animations: "disabled",
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("pass 6: export destination errors and accessible export confirmation", async ({
  page,
  appServer,
}) => {
  await startDemo(page, appServer.url);
  await page
    .getByRole("main")
    .getByRole("button", { name: "Create library", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Preview my library" }),
  ).toBeDisabled();
  await page.getByLabel("New library folder").fill(appServer.data);
  await page.getByRole("button", { name: "Preview my library" }).click();
  await expect(page.getByRole("alert")).toContainText("already exists");
  await page
    .getByLabel("New library folder")
    .fill(join(appServer.data, "Another library"));
  await page.getByRole("button", { name: "Preview my library" }).click();
  await expect(
    page.getByRole("button", { name: "Create my library" }),
  ).toBeVisible();
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "09-export-confirmation.png"),
    fullPage: true,
    animations: "disabled",
  });
});

test("pass 7: local API rejects missing authorization, foreign origins, and invalid hosts", async ({
  request,
  appServer,
}) => {
  const [base, token] = appServer.url.split("#");
  const endpoint = new URL("api/rpc", base).href;
  const data = { version: 1, method: "sessions", params: {} };
  const missing = await request.post(endpoint, { data });
  expect(missing.status()).toBe(403);
  const foreign = await request.post(endpoint, {
    data,
    headers: {
      Authorization: `Bearer ${token}`,
      Origin: "https://example.invalid",
    },
  });
  expect(foreign.status()).toBe(403);
  const host = await request.post(endpoint, {
    data,
    headers: { Authorization: `Bearer ${token}`, Host: "example.invalid" },
  });
  expect(host.status()).toBe(403);
  const accepted = await request.post(endpoint, {
    data,
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(accepted.status()).toBe(200);
});

test("first-run guide explains the niche, one-volume prep, output, and manual cleanup", async ({
  page,
  appServer,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(appServer.url);
  const guide = page.getByRole("dialog");
  await expect(
    guide.getByRole("heading", {
      name: "Years of imports. One organized library.",
    }),
  ).toBeVisible();
  await expect(guide).toContainText("Windows import");
  await expect(guide).toContainText("JPEG conversions of HEIC originals");
  await expect(guide).toContainText("SD-card transfers");
  for (let step = 0; step < 4; step++) {
    const audit = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(
      audit.violations.map((v) => ({
        id: v.id,
        nodes: v.nodes.map((n) => n.target),
      })),
    ).toEqual([]);
    await page.screenshot({
      path: join(appServer.data, `10-guide-${step + 1}.png`),
      fullPage: true,
      animations: "disabled",
    });
    if (step < 3)
      await guide.getByRole("button", { name: "Next", exact: true }).click();
  }
  await expect(guide).toContainText(
    "Omitted alternatives, unsupported files, and errors may exist only in the old import folders",
  );
  await expect(guide).toContainText("manually remove the old source files");
  await guide
    .getByRole("button", { name: "Ready to prepare my media" })
    .click();
  await expect(guide).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "One home for years of imports." }),
  ).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "Guide & help" }).click();
  await expect(
    page.getByRole("heading", {
      name: "Years of imports. One organized library.",
    }),
  ).toBeVisible();
});

test("first-run guide remains navigable on a small window", async ({
  page,
  appServer,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(appServer.url);
  const guide = page.getByRole("dialog");
  for (let step = 0; step < 4; step++) {
    await expect(guide).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    if (step < 3) {
      await expect(
        guide.getByRole("button", { name: "Next", exact: true }),
      ).toBeInViewport();
      await guide.getByRole("button", { name: "Next", exact: true }).click();
    }
  }
  await page.screenshot({
    path: join(appServer.data, "11-guide-compact.png"),
    fullPage: true,
    animations: "disabled",
  });
  await expect(
    guide.getByRole("button", { name: "Ready to prepare my media" }),
  ).toBeInViewport();
  await guide
    .getByRole("button", { name: "Ready to prepare my media" })
    .click();
  await expect(guide).toHaveCount(0);
});

test("disk checks explain capability, test a chosen folder, and clean up", async ({
  page,
  appServer,
}) => {
  await page.goto(appServer.url);
  await page.getByRole("button", { name: "Close import guide" }).click();
  await page
    .getByText("Which disks support hard links?", { exact: true })
    .click();
  await expect(page.locator(".disk-check")).toContainText(
    "exFAT and FAT32 do not",
  );
  await page
    .getByRole("button", { name: "Check local disks", exact: true })
    .click();
  await expect(page.locator(".disk-check [role=status]")).toContainText(
    "No media was read",
    { timeout: 20000 },
  );
  await page.getByLabel("Folder to test").fill(appServer.data);
  await page
    .getByRole("button", { name: "Test hard links in this folder" })
    .click();
  await expect(page.locator(".disk-check [role=status]")).toContainText(
    "Temporary test files were removed",
  );
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "12-disk-check.png"),
    fullPage: true,
  });
});

test("custom output layouts and saved source cleanup are understandable at desktop and phone sizes", async ({
  page,
  appServer,
}) => {
  await startDemo(page, appServer.url);
  await page
    .getByRole("main")
    .getByRole("button", { name: "Create library", exact: true })
    .click();
  await page
    .getByLabel("New library folder")
    .fill(join(appServer.data, "Templated library"));
  await page
    .getByLabel("Layout preset", { exact: true })
    .selectOption("custom");
  await expect(
    page.getByLabel("Output template", { exact: true }),
  ).toBeFocused();
  await page
    .getByLabel("Output template", { exact: true })
    .fill("../{stem}{ext}");
  await page.getByRole("button", { name: "Preview my library" }).click();
  await expect(page.getByRole("alert")).toContainText("cannot contain");
  await page
    .getByLabel("Output template", { exact: true })
    .fill("{kind}/{year}/{stem}{ext}");
  await page.getByRole("button", { name: "Preview my library" }).click();
  await expect(
    page.getByRole("heading", { name: "Here’s what will happen" }),
  ).toBeVisible();
  await page.getByText("Preview actual output paths", { exact: false }).click();
  await expect(page.locator(".planned-paths")).toContainText("Unknown-year");
  await page.screenshot({
    path: join(appServer.data, "13-custom-layout.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Create my library" }).click();
  await expect(
    page.getByRole("heading", {
      name: "We don't delete anything",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator(".cleanup-review")).toContainText(
    "1 omitted copy has identical bytes",
  );
  await expect(page.locator(".cleanup-files")).toContainText("readme.txt");
  await expect(page.locator(".cleanup-review")).toContainText(
    "Files added or changed after the scan are not covered",
  );
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "14-cleanup-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: join(appServer.data, "15-cleanup-compact.png"),
    fullPage: true,
  });
  await page.reload();
  await page
    .getByRole("main")
    .getByRole("button", { name: "Create library", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Review the last export and source cleanup" })
    .click();
  await expect(page.locator(".cleanup-files")).toContainText("readme.txt");
});

test("RAW review shows decoded renders and distinguishes sensor evidence from file identity", async ({
  page,
  appServer,
}) => {
  const inputs = join(appServer.data, "RAW imports");
  await mkdir(inputs);
  const python =
    process.platform === "win32"
      ? join(root, ".venv/Scripts/python.exe")
      : join(root, ".venv/bin/python");
  execFileSync(
    python,
    [
      "-c",
      'from pathlib import Path; import sys; from scripts.synthetic_dng import make_dng; root=Path(sys.argv[1]); make_dng(root/"original.dng"); make_dng(root/"rewritten.dng", "changed file metadata")',
      inputs,
    ],
    { cwd: root },
  );
  await page.goto(appServer.url);
  await page.getByRole("button", { name: "Close import guide" }).click();
  await page.getByLabel("Folder path").fill(inputs);
  await page
    .getByRole("button", { name: "Choose a folder", exact: true })
    .click();
  await page.getByRole("button", { name: "Scan imports", exact: true }).click();
  await expect(page.locator(".group-card")).toHaveCount(1);
  await expect(page.locator(".group-card img")).toHaveCount(1);
  await page.locator(".group-card").click();
  await expect(page.getByRole("dialog")).toContainText("RAW sample evidence");
  await expect(page.locator(".compare-file img")).toHaveCount(2);
  await expect(page.locator(".raw-render-note")).toHaveCount(2);
  await expect(page.getByRole("dialog")).toContainText(
    "Metadata and rendering instructions may differ",
  );
  await page.getByRole("dialog").evaluate(async (el) => {
    await Promise.all(
      el.getAnimations().map((animation) => animation.finished.catch(() => {})),
    );
  });
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await page.screenshot({
    path: join(appServer.data, "16-raw-sample-review.png"),
    fullPage: true,
  });
});

test("unsaved selections survive navigation attempts until explicitly saved or discarded", async ({
  page,
  appServer,
}) => {
  await startDemo(page, appServer.url);
  await page.getByRole("button", { name: "Similar", exact: true }).click();
  const first = page.locator(".group-card").first();
  await first.click();
  const title = await page
    .getByRole("dialog")
    .getByRole("heading", { level: 2 })
    .innerText();
  await expect(page.getByRole("dialog")).toContainText(
    "Unchecked files stay in the source folders",
  );
  await page.getByRole("checkbox").nth(1).uncheck();
  await page.getByRole("button", { name: "Next group", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Your selection has not been saved" }),
  ).toBeFocused();
  let failSave = true;
  await page.route("**/api/rpc", async (route) => {
    if (route.request().postDataJSON().method === "decide" && failSave) {
      failSave = false;
      await route.fulfill({
        json: { error: "Could not save this selection. Please try again." },
      });
    } else await route.continue();
  });
  await page
    .getByRole("button", { name: "Save and continue", exact: true })
    .click();
  await expect(page.locator(".comparison-error")).toContainText(
    "Could not save this selection",
  );
  await expect(
    page.getByRole("heading", { name: "Your selection has not been saved" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Keep reviewing", exact: true })
    .click();
  await expect(page.getByRole("checkbox").first()).toBeFocused();
  await expect(page.getByRole("checkbox").nth(1)).not.toBeChecked();
  await page.mouse.click(3, 3);
  await expect(
    page.getByRole("heading", { name: "Your selection has not been saved" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Keep reviewing", exact: true })
    .click();
  await page.keyboard.press("Escape");
  await page
    .getByRole("button", { name: "Discard changes", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await first.click();
  await expect(page.getByRole("checkbox").nth(1)).toBeChecked();
  await page.getByRole("checkbox").nth(1).uncheck();
  await page.getByRole("button", { name: "Next group", exact: true }).click();
  await page
    .getByRole("button", { name: "Save and continue", exact: true })
    .click();
  await expect(
    page.getByRole("dialog").getByRole("heading", { level: 2 }),
  ).not.toHaveText(title);
  await page
    .getByRole("button", { name: "Previous group", exact: true })
    .click();
  await expect(page.getByRole("checkbox").nth(1)).not.toBeChecked();
  await page.getByRole("checkbox").nth(1).check();
  await page.setViewportSize({ width: 390, height: 844 });
  await page
    .getByRole("button", { name: "Close comparison", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your selection has not been saved" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("dialog").evaluate(async (el) => {
    await Promise.all(
      el.getAnimations().map((a) => a.finished.catch(() => {})),
    );
  });
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map((v) => ({
      id: v.id,
      nodes: v.nodes.map((n) => n.target),
    })),
  ).toEqual([]);
  await expect(page.locator(".toast")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Save and continue", exact: true }),
  ).toBeInViewport();
  await page.screenshot({
    path: join(appServer.data, "17-unsaved-selection.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Discard changes", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
