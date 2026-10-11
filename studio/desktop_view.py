"""Display the local studio inside a native Windows WebView2 application window."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    target = urlsplit(args.url)
    if target.scheme != "http" or target.hostname != "127.0.0.1" or not target.port:
        raise ValueError("Desktop view requires the local application server")
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True
    window = webview.create_window(
        "DeeL Pyess",
        args.url,
        width=1280,
        height=900,
        min_size=(390, 640),
        background_color="#080d16",
        text_select=True,
    )
    outcome = {"passed": not bool(args.smoke_report)}

    def verify():
        try:
            for _ in range(100):
                if window.evaluate_js("document.querySelector('#app-version')?.textContent || ''"):
                    break
                time.sleep(0.2)
            assert window.evaluate_js(
                "document.querySelector('h1').textContent.includes('モデルに')"
            )
            window.evaluate_js("document.querySelector('[data-tab=generate]').click()")
            assert window.evaluate_js("!document.querySelector('#generate').hidden")
            window.evaluate_js("document.querySelector('#update-button').click()")
            assert window.evaluate_js("document.querySelector('#update-dialog').open")
            window.evaluate_js(
                "document.querySelector('#close-update').click(); document.querySelector('#gpu').click()"
            )
            assert window.evaluate_js("document.querySelector('#hardware-dialog').open")
            from PIL import ImageGrab

            ImageGrab.grab().save(args.smoke_report.with_suffix(".png"))
            outcome["passed"] = True
            args.smoke_report.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "checks": [
                            "webview2",
                            "real_app_dom",
                            "generation_navigation",
                            "update_dialog",
                            "hardware_dialog",
                        ],
                    }
                ),
                encoding="utf-8",
            )
        finally:
            window.destroy()

    webview.start(
        verify if args.smoke_report else None,
        gui="edgechromium",
        private_mode=False,
        storage_path=str(args.storage),
    )
    if not outcome["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
