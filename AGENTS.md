# Cloud Service Space

Python is the primary development language. Preserve support for other
languages and do not choose an application framework without task context.

- Prepare the environment with `bash scripts/cloud-environment.sh setup`.
- Run commands through `bash scripts/cloud-environment.sh run COMMAND` to use
  the Python virtual environment and writable caches.
- For environment changes, run `bash scripts/cloud-environment.sh verify`.
  This requires three consecutive passing runs; restart the three-run check
  after fixing a failure.
- Use failed assertions, tracebacks, test durations, and browser screenshots to
  identify the failed operation before making a fix. Keep JUnit reports in
  `artifacts/verification/` and screenshots in `artifacts/browser/`. Do not
  suppress, skip, or weaken a failing check just to obtain a passing result.
- Keep credentials out of files and logs. The Cloud Environment plugin's
  runtime status is distinct from saved configuration drafts.

## Websites and games

Start the actual project server and inspect its actual pages with Playwright.
Check desktop and mobile viewports, main content visibility, controls, asset
loading, and browser console/network errors. For canvas or WebGL experiences,
verify nonblank pixels, animation, and interaction. Test that source changes
appear after the appropriate reload or rebuild; a running process or open
port alone is not a passing display test. Save screenshots and inspect them.

The fixture in `tests/test_browser.py` validates browser tooling only. It does
not replace testing the real website or game produced by a future task.
