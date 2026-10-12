import json
import os
import shutil
import socket
import sqlite3
import threading
import wave
from pathlib import Path

import pytest
import uvicorn
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import expect, sync_playwright

from studio.app import create_app
from tests.test_studio import png


@pytest.mark.parametrize(
    "viewport", [{"width": 1440, "height": 1000}, {"width": 390, "height": 844}]
)
def test_actual_studio_upload_edit_and_navigation(tmp_path, viewport):
    repository = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(repository / ".cache/ms-playwright"))
    app = create_app(tmp_path / "data", run_workers=False)
    app.mount("/website", StaticFiles(directory=repository / "website", html=True), name="website")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    image = tmp_path / "sample.png"
    image.write_bytes(png())
    sound = tmp_path / "sound.wav"
    with wave.open(str(sound), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(bytes(32000))
    screenshots = (
        repository / "artifacts/browser" / f"run-{os.getenv('CLOUDSPACE_CHECK_RUN', 'manual')}"
    )
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=shutil.which("chromium"))
            page = browser.new_page(viewport=viewport)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.on("requestfailed", lambda req: errors.append(f"{req.url}: {req.failure}"))
            page.on("response", lambda res: errors.append(res.url) if res.status >= 400 else None)
            page.goto(f"http://127.0.0.1:{port}/")
            expect(page.get_by_role("heading", name="あなたの世界を、 モデルに。")).to_be_visible()
            expect(page.locator("#gpu")).to_contain_text("GPU")
            page.locator("#gpu").click()
            expect(page.get_by_role("heading", name="PC環境・AI機能")).to_be_visible()
            expect(page.locator("#hardware-summary")).to_contain_text("CPU")
            expect(page.locator("#setup-runtime")).to_be_disabled()
            expect(page.locator("#setup-backend option")).to_have_count(4)
            page.screenshot(path=str(screenshots / f"hardware-{viewport['width']}.png"))
            page.locator("#close-hardware").click()
            page.locator("#update-button").click()
            expect(page.locator("#update-dialog")).to_be_visible()
            expect(page.locator("#update-version")).to_contain_text("0.1.6")
            page.route(
                "**/api/system/update",
                lambda route: route.fulfill(
                    json={"current": "0.1.6", "latest": "0.1.7", "available": True}
                ),
            )
            page.locator("#check-update").click()
            expect(page.locator("#update-status")).to_contain_text("0.1.7")
            expect(page.locator("#install-update")).to_be_disabled()
            page.screenshot(path=str(screenshots / f"update-{viewport['width']}.png"))
            page.locator("#close-update").click()
            expect(page.locator("#public-model")).to_have_value("tiny")
            expect(page.get_by_label("作業プロジェクトを選択")).to_be_visible()
            page.get_by_role("button", name="Tiny-SDで画像をつくる").click()
            expect(page.locator("#public-model")).to_be_visible()
            expect(page.locator("#public-model")).to_have_value("tiny")
            expect(page.locator("#projects option")).to_have_count(1)
            first_project = page.locator("#projects").input_value()
            expect(page.locator("#prompt-help")).to_contain_text("追加学習不要")
            page.reload()
            page.get_by_role("button", name="Tiny-SDで画像をつくる").click()
            expect(page.locator("#projects")).to_have_value(first_project)
            expect(page.locator("#projects option")).to_have_count(1)
            page.get_by_role("button", name="＋ 新しいプロジェクト").click()
            page.get_by_label("プロジェクト名").fill("ブラウザテスト")
            page.get_by_label("トリガーワード", exact=True).fill("browserstyle")
            page.get_by_role("button", name="プロジェクトを作成する ↗").click()
            expect(page.locator("#projects")).to_contain_text("ブラウザテスト")
            page.get_by_role("button", name="モデルを育てる").click()
            page.locator("#upload").set_input_files(str(image))
            expect(page.locator("#image-count")).to_have_text("1 / 100枚")
            expect(page.locator("#images img")).to_be_visible()
            page.get_by_label("画像の説明", exact=True).fill("a drawing of browserstyle")
            page.get_by_label("画像の説明", exact=True).press("Tab")
            expect(page.get_by_role("status")).to_have_text("画像の説明を保存しました。")
            page.reload()
            expect(page.get_by_label("画像の説明", exact=True)).to_have_value(
                "a drawing of browserstyle"
            )
            page.locator('[data-workspace="agent"]').click()
            expect(page.locator("#agent")).to_be_visible()
            page.locator("#agent-name").fill("私の作品アシスタント")
            page.locator("#agent-instruction").fill("短く具体的に提案してください")
            page.locator("#agent-knowledge").fill("作品名は青い庭です")
            page.locator("#agent-add-example").click()
            page.get_by_label("お手本の依頼", exact=True).fill("作品名は？")
            page.get_by_label("お手本の回答", exact=True).fill("青い庭です。")
            page.locator("#agent-save").click()
            expect(page.locator("#agent-version option")).to_have_count(1)
            expect(page.locator("#agent-version")).to_contain_text("私の作品アシスタント")
            saved_agent = page.locator("#agent-version").input_value()
            page.reload()
            page.locator('[data-workspace="agent"]').click()
            page.locator("#agent-start").select_option(saved_agent)
            expect(page.locator("#agent-knowledge")).to_have_value("作品名は青い庭です")
            expect(page.get_by_label("お手本の回答", exact=True)).to_have_value("青い庭です。")
            page.locator('[data-workspace="video"]').click()
            expect(page.locator("#video-training-controls")).to_be_visible()
            expect(page.locator("#video-training-base option")).to_have_count(2)
            expect(page.locator("#video-training-base")).to_have_value("animatediff")
            expect(page.locator("#video-training-help")).to_contain_text("CPU")
            expect(page.locator("#steps")).to_have_value("10")
            expect(page.locator("#media-upload-controls")).to_have_attribute("open", "")
            page.locator('[data-workspace="image"]').click()
            expect(page.locator("#training-base option")).to_have_count(4)
            expect(page.locator("#training-base")).to_have_value("tiny")
            page.locator("#training-base").select_option("small")
            expect(page.locator("#training-base-help")).to_contain_text("CPU学習対応")
            page.wait_for_function("projects.find(p=>p.id===current)?.training_model==='small'")
            page.reload()
            expect(page.locator("#training-base")).to_have_value("small")
            page.locator("#training-base").select_option("dreamshaper")
            expect(page.locator("#train-button")).to_be_disabled()
            expect(page.locator("#training-base-help")).to_contain_text(
                "必要条件を満たしていません"
            )
            page.locator("#training-base").select_option("tiny")
            expect(page.locator("#train-button")).to_be_enabled()
            page.locator("#training-mode").select_option("style")
            expect(page.locator("#mode-help")).to_contain_text("画風")
            expect(page.locator("#steps")).to_have_value("30")
            expect(page.locator("#training-group")).to_contain_text("default")
            expect(page.locator("#folder-upload")).to_have_attribute("webkitdirectory", "")
            page.locator("#model-level").select_option("excellent")
            expect(page.locator("#steps")).to_have_value("100")
            page.locator("#model-level").select_option("ultra")
            expect(page.locator("#steps")).to_have_value("200")
            expect(page.locator("#assistant-model option")).to_have_count(2)
            page.locator("#training-instruction").fill("watercolor texture")
            page.locator("#send-training-chat").click()
            expect(page.locator("#training-chat-history")).to_contain_text("watercolor texture")
            page.get_by_text("サイト・検索から自動収集する", exact=True).click()
            page.locator("#collection-provider").select_option("commons")
            expect(page.locator("#collection-url")).to_be_disabled()
            page.locator("#collection-query").fill("watercolor")
            page.get_by_text("動画・音声の学習データを追加", exact=True).click()
            page.locator("#media-kind").select_option("audio")
            page.locator("#media-upload").set_input_files([str(sound)] * 3)
            expect(page.locator("#media-count")).to_have_text("3 / 30件")
            expect(page.locator("#media-list audio")).to_have_count(3)
            page.locator("#training-target").select_option("audio")
            expect(page.locator("#mode-help")).to_contain_text("環境音")
            page.locator("#training-target").select_option("video")
            expect(page.locator("#mode-help")).to_contain_text("CogVideoX")
            page.locator("#training-target").select_option("voice")
            expect(page.locator("#mode-help")).to_contain_text("特徴を抽出")
            expect(page.locator("#extract-video-audio")).to_be_visible()
            page.locator("#extract-video-audio").check()
            page.locator("#collection-media").select_option("audio")
            expect(page.locator("#collection-media")).to_have_value("audio")
            page.locator("#training-target").select_option("image")
            page.get_by_role("button", name="AIルールの設定").click()
            expect(page.locator("#rules-prompt")).to_have_value("")
            page.get_by_label("共通ルール・表現").fill("watercolor, soft colors")
            page.get_by_label("除外する要素", exact=True).last.fill("blur")
            page.get_by_role("button", name="設定を保存 ↗").click()
            page.get_by_role("button", name="AIルールの設定").click()
            expect(page.locator("#rules-prompt")).to_have_value("watercolor, soft colors")
            page.get_by_role("button", name="ルールをリセット").click()
            expect(page.locator("#rules-prompt")).to_have_value("")
            page.get_by_role("button", name="設定を閉じる").click()
            page.get_by_role("button", name="AI開発コース").click()
            expect(page.locator(".course-module")).to_have_count(8)
            page.locator(".course-module input[type=checkbox]").first.check()
            expect(page.locator("#course-count")).to_contain_text("1 / 8")
            page.locator(".course-module textarea").first.fill("My first experiment")
            page.locator(".course-module textarea").first.press("Tab")
            expect(page.get_by_role("status")).to_have_text(
                "コースの進捗・実験ノートを保存しました。"
            )
            page.get_by_role("button", name="ギャラリー").click()
            expect(page.locator("#empty-gallery")).to_be_visible()
            page.get_by_role("button", name="モデルを育てる").click()
            pid = page.locator("#projects").input_value()
            with sqlite3.connect(tmp_path / "data/studio.sqlite3") as db:
                db.execute(
                    "INSERT INTO jobs(id,project_id,kind,status,config) VALUES('chart-test',?,'train','failed',?)",
                    (
                        pid,
                        json.dumps(
                            {"steps": 500, "rank": 8, "learning_rate": 0.0001, "mode": "supervised"}
                        ),
                    ),
                )
                db.execute(
                    "INSERT INTO jobs(id,project_id,kind,status,config) VALUES('audio-preview', ?, 'audio', 'completed', ?)",
                    (pid, json.dumps({"audio_model": "kokoro"})),
                )
                audio_folder = tmp_path / "data/jobs/audio-preview"
                audio_folder.mkdir(parents=True)
                (audio_folder / "output.wav").write_bytes(sound.read_bytes())
                db.executemany(
                    "INSERT INTO training_metrics VALUES(?,?,?,?,?,?)",
                    [
                        ("chart-test", 1, 1.2, 0.0001, 0.8, 3.5),
                        ("chart-test", 10, 0.7, 0.0001, 0.5, 3.6),
                    ],
                )
            expect(page.locator(".metric-bar")).to_have_count(2, timeout=10000)
            page.locator("#metric-type").select_option("lora_norm")
            expect(page.locator("#metric-chart")).to_have_attribute(
                "aria-label", "学習重みの大きさ（L2）の棒グラフ。ステップ1: 3.50、ステップ10: 3.60"
            )
            page.get_by_role("button", name="モデルを育てる").click()
            page.locator("#reset-training").click()
            expect(page.get_by_role("status")).to_contain_text("学習をリセットしました（1件）")
            expect(page.locator("#image-count")).to_have_text("1 / 100枚")
            page.get_by_role("button", name="作品をつくる").click()
            expect(page.locator("#generation-form")).to_be_visible()
            expect(page.locator("#preview audio")).to_be_visible()
            page.wait_for_function("document.querySelector('#preview audio').readyState >= 1")
            assert page.locator("#preview audio").evaluate("audio=>audio.duration>=1")
            preview_audio = page.locator("#preview audio").element_handle()
            expect(page.locator("#public-model option")).to_have_count(6)
            page.locator("#public-model").select_option("ltx")
            expect(page.get_by_label("動画", exact=True)).to_be_checked()
            page.locator("#public-model").select_option("sdxl")
            expect(page.get_by_label("画像", exact=True)).to_be_checked()
            page.locator("#public-model").select_option("")
            page.locator("#audio-model").select_option("kokoro")
            expect(page.get_by_label("音声", exact=True)).to_be_checked()
            expect(page.locator("#use-voice")).to_be_visible()
            page.get_by_role("button", name="Tiny-SDで画像をつくる").click()
            expect(page.locator("#public-model")).to_be_visible()
            expect(page.locator("#public-model")).to_have_value("tiny")
            expect(page.locator("#projects")).to_have_value(pid)
            expect(page.locator("#projects option")).to_have_count(2)
            page.get_by_label("画像", exact=True).check()
            expect(page.locator("#audio-model")).to_have_value("")
            page.locator("#output-resolution").select_option("custom")
            expect(page.locator("#output-resolution")).to_have_value("custom")
            page.locator("#output-resolution").select_option("4k")
            expect(page.locator("#output-resolution")).to_have_value("4k")
            page.get_by_label("動画", exact=True).check()
            expect(page.locator("#generation-video-model")).to_have_value("animatediff")
            expect(page.locator("#source-label")).to_be_hidden()
            page.locator("#generation-video-model").select_option("cogvideo")
            expect(page.locator("#source-label")).to_be_visible()
            page.get_by_role("button", name="ギャラリー").click()
            expect(page.locator("#gallery-items audio")).to_have_count(1)
            expect(page.locator("#gallery-items")).to_contain_text("Kokoro")
            page.get_by_role("button", name="モデルを育てる").click()
            sample_folder = tmp_path / "dataset"
            sample_folder.mkdir()
            (sample_folder / "folder-image.png").write_bytes(png("blue"))
            page.locator("#folder-upload").set_input_files(str(sample_folder))
            expect(page.locator("#image-count")).to_have_text("2 / 100枚")
            assert preview_audio.evaluate("audio=>audio.isConnected")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert page.locator("#images img").first.evaluate(
                "img => img.complete && img.naturalWidth > 0"
            )
            assert not errors, errors
            page.screenshot(
                path=str(screenshots / f"studio-{viewport['width']}.png"), full_page=True
            )
            page.locator("#upload").set_input_files(str(image))
            expect(page.locator("#image-count")).to_have_text("3 / 100枚")
            page.locator("#steps").fill("10")
            with page.expect_response(f"**/api/projects/{pid}/train") as trained:
                page.locator("#train-button").click()
            assert trained.value.status == 202
            config = json.loads(trained.value.json()["config"])
            assert config["training_model"] == "tiny" and config["steps"] == 10
            expect(page.locator("#train-button")).to_be_disabled()
            page.locator("#reset-training").click()
            expect(page.get_by_role("status")).to_contain_text("学習をリセットしました（1件）")
            expect(page.locator("#image-count")).to_have_text("3 / 100枚")
            page.get_by_role("button", name="Tiny-SDで画像をつくる").click()
            page.locator("#prompt").fill("a blue watercolor bird")
            with page.expect_response(f"**/api/projects/{pid}/generate") as generated:
                page.locator("#generate-button").click()
            assert generated.value.status == 202
            with sqlite3.connect(tmp_path / "data/studio.sqlite3") as db:
                row = db.execute(
                    "SELECT config FROM jobs WHERE project_id=? AND kind='image' ORDER BY rowid DESC LIMIT 1",
                    (pid,),
                ).fetchone()
                assert json.loads(row[0])["public_model"] == "tiny"
            page.screenshot(path=str(screenshots / f"tiny-{viewport['width']}.png"))
            page.goto(f"http://127.0.0.1:{port}/website/")
            expect(
                page.get_by_role("heading", name="自分のAIを、 つくるところから。")
            ).to_be_visible()
            page.locator('[data-os="linux"]').click()
            expect(page.locator("#os-title")).to_contain_text("Linux")
            expect(page.locator("#os-instructions")).to_contain_text("./install.sh --gpu")
            page.locator('[data-os="windows"]').click()
            expect(page.locator("#os-instructions")).to_contain_text("セットアップEXE")
            assert (
                page.locator(
                    '#download-link[href$="DeeL-Pyess-Setup-v0.1.6-Windows-x64.exe"]'
                ).count()
                == 1
            )
            page.locator('[data-os="linux"]').click()
            expect(page.locator("#download-link")).to_have_attribute(
                "href",
                "https://github.com/icchan090pp/personal-ai-studio/releases/download/v0.1.6/DeeL-Pyess-v0.1.6.zip",
            )
            page.locator('[data-os="windows"]').click()
            page.locator(".usage-screen img").scroll_into_view_if_needed()
            expect(page.locator(".usage-screen img")).to_be_visible()
            for screenshot in page.locator("img").all():
                screenshot.scroll_into_view_if_needed()
                page.wait_for_function(
                    "image=>image.complete && image.naturalWidth>0",
                    arg=screenshot.element_handle(),
                    timeout=15000,
                )
            assert page.locator(".usage-screen img").evaluate(
                "image=>image.complete && image.naturalWidth>0"
            )
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors, errors
            page.screenshot(
                path=str(screenshots / f"website-{viewport['width']}.png"), full_page=True
            )
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
