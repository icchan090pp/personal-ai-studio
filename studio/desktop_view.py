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
                if window.evaluate_js(
                    "document.querySelector('#app-version')?.textContent && !document.querySelector('#quick-tiny')?.disabled"
                ):
                    break
                time.sleep(0.2)
            assert window.evaluate_js(
                "document.querySelector('h1').textContent.includes('モデルに')"
            )
            assert window.evaluate_js(
                "document.querySelector('#projects').getAttribute('aria-label') === '作業プロジェクトを選択'"
            )
            assert window.evaluate_js(
                "document.querySelector('#training-base').options.length === 5 && document.querySelector('#training-base').value === 'tiny'"
            )
            assert window.evaluate_js(
                "document.querySelector('#training-base-help').textContent.includes('CPU')"
            )
            assert window.evaluate_js(
                "Boolean(document.querySelector('#finish-button') && document.querySelector('#add-generation-image') && document.querySelector('#add-training-image'))"
            )
            window.evaluate_js("document.querySelector('[data-workspace=video]').click()")
            assert window.evaluate_js(
                "!document.querySelector('#video-training-controls').hidden && document.querySelector('#video-training-base').options.length === 2"
            )
            window.evaluate_js("document.querySelector('[data-workspace=agent]').click()")
            assert window.evaluate_js(
                "!document.querySelector('#agent').hidden && document.querySelector('#agent-model').value === 'gpt-oss:20b'"
            )
            window.evaluate_js("document.querySelector('#quick-tiny').click()")
            assert window.evaluate_js("!document.querySelector('#generate').hidden")
            assert window.evaluate_js(
                "document.querySelector('#public-model').value === 'tiny' && !document.querySelector('#public-model-label').hidden"
            )
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
                            "tiny_sd_quick_selection",
                            "five_training_configurations_and_cpu_guidance",
                            "video_training_and_agent_sections",
                            "image_finishing_and_prompt_attachments",
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
