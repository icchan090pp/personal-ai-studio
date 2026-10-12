"""Native desktop launcher; the server and model workers use the bundled Python."""

import argparse
import io
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import messagebox

from studio.updates import CURRENT_VERSION, check_update, download_update


class Application:
    def __init__(self, root, home):
        self.root = root
        self.home = home
        self.python = home / "runtime/python.exe"
        self.data = Path(os.environ.get("LOCALAPPDATA", str(home))) / "DeeL Pyess"
        self.data.mkdir(parents=True, exist_ok=True)
        self.log_path = self.data / "startup.log"
        self.log = self.log_path.open("a", encoding="utf-8")
        self.process = None
        self.viewer = None
        self.viewer_started = False
        self.ready = False
        self.installer = None
        self.installing = False
        self.smoke = False
        self.smoke_basic = False
        self.setup_count = 0
        self.command_path = self.data / f"setup-command-{os.getpid()}.json"
        self.command_path.unlink(missing_ok=True)
        self.events = queue.Queue()
        self.closing = False
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.env = os.environ.copy()
        self.env.update(
            STUDIO_PYTHON=str(self.python),
            STUDIO_DATA_DIR=str(self.data / "data"),
            STUDIO_DESKTOP_COMMAND=str(self.command_path),
            STUDIO_DML_PYTHON=str(self.data / "directml/Scripts/python.exe"),
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
        )
        # A user's Python configuration must not redirect the bundled interpreter.
        for key in ("PYTHONHOME", "PYTHONPATH"):
            self.env.pop(key, None)
        root.title("DeeL Pyess " + CURRENT_VERSION)
        root.geometry("640x610")
        root.minsize(580, 610)
        root.configure(bg="#080d16")
        tk.Label(root, text="DeeL Pyess", fg="#67b9ff", bg="#080d16", font=("Segoe UI", 26)).pack(
            pady=18
        )
        self.status = tk.StringVar(value="アプリを起動しています…")
        tk.Label(root, textvariable=self.status, fg="white", bg="#080d16", wraplength=550).pack(
            pady=8
        )
        self.hardware = tk.StringVar(value="PCのGPUとAI機能を確認しています…")
        tk.Label(root, textvariable=self.hardware, fg="#67b9ff", bg="#080d16", wraplength=600).pack(
            pady=8
        )
        self.backend_choices = {
            "自動選択（推奨）": "auto",
            "AMD・Intel / DirectML": "directml",
            "CPU互換モード": "cpu",
            "NVIDIA / CUDA": "cuda",
        }
        self.backend = tk.StringVar(value="自動選択（推奨）")
        tk.OptionMenu(root, self.backend, *self.backend_choices).pack(pady=4)
        self.buttons = []
        for text, command in (
            ("アプリを開く", self.open_app),
            ("このPCのAI機能をセットアップ / 修復", lambda: self.request_install(False)),
            ("AI機能 ＋ 日本語音声・声質機能をセットアップ", lambda: self.request_install(True)),
            ("PC環境を再診断", self.diagnose),
            (
                "アップデートを確認・取得",
                self.request_update,
            ),
            ("診断ログを開く", lambda: os.startfile(self.log_path)),
            (
                "専用画面の表示エンジンを修復",
                lambda: subprocess.Popen(
                    [str(self.home / "MicrosoftEdgeWebview2Setup.exe"), "/install"]
                ),
            ),
        ):
            button = tk.Button(root, text=text, command=command, width=46)
            button.pack(pady=4)
            self.buttons.append(button)
        for button in self.buttons[:3]:
            button.configure(state="disabled")
        tk.Label(
            root,
            text="このウィンドウを閉じるとアプリも停止します。\n追加機能・モデルの初回取得にはネット接続と空き容量が必要です。",
            fg="#a5b7cf",
            bg="#080d16",
        ).pack(pady=10)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def spawn(self, arguments):
        self.log.flush()
        return subprocess.Popen(
            [str(self.python), *arguments],
            cwd=self.home,
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def start(self, browser=True):
        self.ready = False
        if not self.python.is_file():
            raise RuntimeError(
                "同梱Pythonが見つかりません。Windowsセットアップを再インストールしてください。"
            )
        self.process = self.spawn(["launch.py", "--no-browser", "--port", str(self.port)])

        def wait():
            deadline = time.monotonic() + 60
            while not self.closing and time.monotonic() < deadline and self.process.poll() is None:
                try:
                    with urllib.request.urlopen(self.url + "/api/models", timeout=1) as response:
                        models = json.load(response)
                    if len(models) >= 2 and {model["kind"] for model in models} == {
                        "image",
                        "video",
                    }:
                        self.events.put(("ready", browser))
                        return
                except (OSError, ValueError):
                    time.sleep(0.2)
            if not self.closing:
                self.events.put(("error", "起動できませんでした。診断ログをご確認ください。"))

        threading.Thread(target=wait, daemon=True).start()

    def stop(self):
        self.ready = False
        if self.process and self.process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.process.wait(timeout=15)

    def open_app(self):
        if self.ready:
            if self.viewer and self.viewer.poll() is None:
                return
            self.viewer = self.spawn(
                [
                    "-m",
                    "studio.desktop_view",
                    "--url",
                    self.url,
                    "--storage",
                    str(self.data / "webview"),
                ]
            )
            self.viewer_started = True
            self.root.withdraw()
        elif not self.installing and (not self.process or self.process.poll() is not None):
            self.status.set("アプリを再起動しています…")
            self.start()

    def request_install(self, voice):
        if self.installing:
            return
        body = json.dumps(
            {"backend": self.backend_choices[self.backend.get()], "voice": voice}
        ).encode()
        try:
            request = urllib.request.Request(
                self.url + "/api/system/setup",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                json.load(response)
            self.status.set("セットアップを開始します…")
        except Exception as exc:
            messagebox.showerror(
                "DeeL Pyess",
                "セットアップを開始できません。実行中の学習・生成を終了してください。\n" + str(exc),
            )

    def request_update(self):
        if self.installing:
            return
        try:
            request = urllib.request.Request(
                self.url + "/api/system/update",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                json.load(response)
        except Exception as exc:
            messagebox.showerror(
                "DeeL Pyess",
                "更新を開始できません。処理が完了してから再試行してください。\n" + str(exc),
            )

    def update(self):
        if self.installing:
            return
        self.installing = True
        self.root.deiconify()
        self.status.set("最新版を確認しています…")

        def work():
            try:
                release = check_update()
                if not release["available"]:
                    self.events.put(("update_current", release))
                    return
                path = download_update(
                    release,
                    self.data / "updates",
                    lambda value: self.events.put(("update_progress", value)),
                )
                self.events.put(("update_downloaded", path))
            except Exception as exc:
                self.events.put(
                    ("error", "更新を完了できません。現在のアプリは使用できます。\n" + str(exc))
                )
            finally:
                self.installing = False

        threading.Thread(target=work, daemon=True).start()

    def diagnose(self):
        def work():
            try:
                with urllib.request.urlopen(
                    self.url + "/api/system/hardware?refresh=true", timeout=45
                ) as response:
                    self.events.put(("hardware", json.load(response)))
            except Exception as exc:
                self.events.put(
                    (
                        "hardware",
                        {
                            "adapters": [],
                            "label": "診断を完了できませんでした",
                            "message": str(exc),
                        },
                    )
                )

        threading.Thread(target=work, daemon=True).start()

    def install(self, voice, backend="auto"):
        if self.installing:
            return
        self.installing = True
        self.root.deiconify()
        if self.viewer and self.viewer.poll() is None:
            self.viewer_started = False
            subprocess.run(
                ["taskkill", "/PID", str(self.viewer.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.viewer.wait(timeout=15)
        self.stop()
        for button in self.buttons[:3]:
            button.configure(state="disabled")
        self.status.set("追加機能をダウンロード中です。数GB以上になるため時間がかかります…")

        def work():
            try:
                arguments = ["-m", "studio.setup_runtime", "--backend", backend]
                if voice:
                    arguments.append("--voice")
                self.installer = self.spawn(arguments)
                if self.installer.wait() != 0:
                    raise RuntimeError(
                        "追加機能のインストールに失敗しました。診断ログをご確認ください。"
                    )
                self.events.put(("installed", None))
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.installer = None
                self.installing = False

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.events.empty():
            event, value = self.events.get()
            if event == "ready":
                self.ready = True
                for button in self.buttons:
                    button.configure(state="normal")
                self.status.set("起動しました。専用画面で画像の投稿・学習・生成を操作できます。")
                if value:
                    self.open_app()
                self.diagnose()
            elif event == "hardware":
                devices = " / ".join(value.get("adapters", [])) or "GPU名を取得できませんでした"
                self.hardware.set(devices + "\n" + value["label"] + "\n" + value["message"])
            elif event == "installed":
                self.setup_count += 1
                for button in self.buttons:
                    button.configure(state="normal")
                self.status.set("追加機能の準備ができました。再起動しています…")
                self.start(browser=not self.smoke)
            elif event == "update_progress":
                self.status.set(f"更新版を取得しています… {value}%（現在のアプリは保持します）")
            elif event == "update_current":
                self.status.set("現在のバージョン " + CURRENT_VERSION + " は最新です。")
            elif event == "update_downloaded":
                self.root.deiconify()
                self.status.set("更新版の検証が完了しました。")
                if messagebox.askyesno(
                    "DeeL Pyess の更新",
                    "保存済みのデータを保持して更新します。アプリを終了してセットアップを開始しますか？",
                ):
                    # Check again: the user may have started work during download.
                    try:
                        request = urllib.request.Request(
                            self.url + "/api/system/update",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=10) as response:
                            json.load(response)
                        self.command_path.unlink(missing_ok=True)
                        subprocess.Popen(
                            [str(value), f"/DIR={self.home}", "/NORESTART"], cwd=self.home
                        )
                        self.close()
                        return
                    except Exception as exc:
                        self.status.set(
                            "更新を開始できません。実行中の処理を終了して再試行してください。"
                            + str(exc)
                        )
            else:
                self.root.deiconify()
                for button in self.buttons:
                    button.configure(state="normal")
                self.status.set(value)
                if not self.smoke:
                    messagebox.showerror("DeeL Pyess", value + "\n" + str(self.log_path))
        if self.ready and self.process.poll() is not None:
            self.ready = False
            self.events.put(("error", "アプリが停止しました。診断ログをご確認ください。"))
        if self.viewer_started and self.viewer and self.viewer.poll() is not None:
            self.viewer_started = False
            if self.viewer.returncode == 0:
                if not self.installing:
                    self.close()
                    return
                self.root.deiconify()
            self.root.deiconify()
            self.status.set(
                "専用画面を起動できません。Microsoft Edge WebView2 Runtimeをインストールして再試行してください。"
            )
        if self.ready and not self.installing and self.command_path.exists():
            try:
                command = json.loads(self.command_path.read_text(encoding="utf-8"))
                if command == {"action": "update"}:
                    self.command_path.unlink()
                    self.update()
                elif command.get("backend") in {"auto", "cpu", "cuda", "directml"} and isinstance(
                    command.get("voice"), bool
                ):
                    self.command_path.unlink()
                    self.root.after(750, lambda: self.install(command["voice"], command["backend"]))
            except (OSError, ValueError):
                pass
        self.root.after(200, self.poll)

    def close(self):
        if self.installing:
            messagebox.showinfo("DeeL Pyess", "追加機能のインストールが終わるまでお待ちください。")
            return
        self.closing = True
        if self.viewer and self.viewer.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(self.viewer.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.viewer.wait(timeout=15)
        self.stop()
        self.log.close()
        self.command_path.unlink(missing_ok=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-setup", action="store_true")
    parser.add_argument("--smoke-basic", action="store_true")
    args = parser.parse_args()
    home = (
        Path(sys.executable).parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1]
    )
    root = tk.Tk()
    app = Application(root, home)
    app.smoke = args.smoke_test
    app.smoke_basic = args.smoke_basic
    app.start(browser=not args.smoke_test)
    if args.smoke_test:
        # Exercise the same GUI and server launch path as a real double-click.
        deadline = time.monotonic() + 180

        outcome = {"passed": False, "requested_setup": False}

        def check():
            if args.smoke_setup and app.ready and not outcome["requested_setup"]:
                outcome["requested_setup"] = True
                app.backend.set("CPU互換モード")
                app.buttons[1].invoke()
            if (
                app.ready
                and not app.hardware.get().startswith("PCのGPUとAI機能")
                and (not args.smoke_setup or app.setup_count == 1)
            ):
                try:
                    validate_installed_app(app)
                    root.update_idletasks()
                    from PIL import ImageGrab

                    ImageGrab.grab(
                        bbox=(
                            root.winfo_rootx(),
                            root.winfo_rooty(),
                            root.winfo_rootx() + root.winfo_width(),
                            root.winfo_rooty() + root.winfo_height(),
                        )
                    ).save(app.data / "windows-desktop.png")
                    outcome["passed"] = True
                    (app.data / "windows-smoke.json").write_text(
                        json.dumps(
                            {
                                "passed": True,
                                "python": str(app.python),
                                "data": str(app.data / "data"),
                                "checks": ["gui", "server", "models", "upload", "persistence"],
                            }
                        ),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    app.log.write(str(exc) + "\n")
                finally:
                    app.close()
            elif time.monotonic() > deadline or app.status.get().startswith("起動できません"):
                app.close()
            else:
                root.after(200, check)

        root.after(200, check)
    root.mainloop()
    if args.smoke_test and not outcome["passed"]:
        raise SystemExit(1)


def validate_installed_app(app):
    from PIL import Image

    with urllib.request.urlopen(app.url + "/", timeout=5) as response:
        assert b"DeeL" in response.read()
    with urllib.request.urlopen(app.url + "/api/models", timeout=5) as response:
        assert {model["kind"] for model in json.load(response)} == {"image", "video"}
    with urllib.request.urlopen(app.url + "/api/system", timeout=45) as response:
        system = json.load(response)
        assert system["version"] == CURRENT_VERSION and system["desktop_setup"]
        if not app.smoke_basic:
            assert system["hardware"]["cpu_ready"] is True
        assert "未接続" not in system["hardware"]["message"]
    request = urllib.request.Request(
        app.url + "/api/projects",
        data=json.dumps({"name": "Windows 起動試験", "trigger": "windowscheck"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 201
        project = json.load(response)["id"]
    image = io.BytesIO()
    Image.new("RGB", (128, 128), "blue").save(image, format="PNG")
    boundary = "DeeLPyessSmokeBoundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="image.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + image.getvalue()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        app.url + f"/api/projects/{project}/images",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 201
        assert len(json.load(response)) == 1
    with urllib.request.urlopen(app.url + f"/api/projects/{project}/images", timeout=5) as response:
        assert len(json.load(response)) == 1
    assert (app.data / "data").is_dir()
    report = app.data / "windows-webview.json"
    report.unlink(missing_ok=True)
    viewer = app.spawn(
        [
            "-m",
            "studio.desktop_view",
            "--url",
            app.url,
            "--storage",
            str(app.data / "webview-test"),
            "--smoke-report",
            str(report),
        ]
    )
    assert viewer.wait(timeout=60) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["passed"]


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--smoke-test" not in sys.argv:
            messagebox.showerror("DeeL Pyess", str(error))
        raise
