import functools
import os
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright


class FreshContentHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *_args):
        pass


@pytest.mark.parametrize(
    ("viewport_name", "viewport"),
    [
        ("desktop", {"width": 1440, "height": 900}),
        ("mobile", {"width": 390, "height": 844}),
    ],
)
def test_render_interaction_animation_and_reload(tmp_path, viewport_name, viewport):
    repository = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(repository / ".cache" / "ms-playwright"))
    source = (Path(__file__).parent / "fixtures" / "browser.html").read_text(encoding="utf-8")
    page_file = tmp_path / "index.html"
    page_file.write_text(source, encoding="utf-8")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(FreshContentHandler, directory=str(tmp_path))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    run = os.environ.get("CLOUDSPACE_CHECK_RUN", "manual")
    screenshots = repository / "artifacts" / "browser" / f"run-{run}"
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            executable = shutil.which("chromium") or shutil.which("google-chrome")
            browser = playwright.chromium.launch(executable_path=executable)
            page = browser.new_page(viewport=viewport)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "console",
                lambda message: errors.append(message.text) if message.type == "error" else None,
            )
            page.on("requestfailed", lambda request: errors.append(request.url))
            page.on(
                "response",
                lambda response: errors.append(response.url) if response.status >= 400 else None,
            )
            try:
                response = page.goto(f"http://127.0.0.1:{server.server_port}/")
                assert response.status == 200
                expect(page.get_by_role("heading")).to_have_text("Cloud Service Space")
                expect(page.locator("#version")).to_have_text("Version 1")
                expect(page.locator("canvas")).to_be_visible()
                page.wait_for_function("window.renderedFrames >= 5")
                first_frame = page.locator("canvas").evaluate("canvas => canvas.toDataURL()")
                frame_count = page.evaluate("window.renderedFrames")
                page.wait_for_function(
                    "count => window.renderedFrames >= count + 10", arg=frame_count
                )
                assert (
                    page.locator("canvas").evaluate("canvas => canvas.toDataURL()") != first_frame
                )
                page.get_by_role("button", name="Update").click()
                expect(page.get_by_role("status")).to_have_text("Updated")
                page.wait_for_function("""() => {
                    const data = document.querySelector('canvas').getContext('2d')
                        .getImageData(0, 0, 480, 240).data;
                    let visiblePixels = 0;
                    for (let index = 0; index < data.length; index += 4) {
                        if (data[index] === 8 && data[index + 1] === 127) visiblePixels++;
                    }
                    return visiblePixels >= 1000;
                }""")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page_file.write_text(source.replace("Version 1", "Version 2"), encoding="utf-8")
                page.reload()
                expect(page.locator("#version")).to_have_text("Version 2")
                assert not errors, errors
            finally:
                page.screenshot(path=str(screenshots / f"{viewport_name}.png"), full_page=True)
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
