"""Native desktop launcher; the server and model workers use the bundled Python."""

import argparse
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
import webbrowser
from pathlib import Path
from tkinter import messagebox


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
        self.installer = None
        self.installing = False
        self.smoke = False
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
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
        )
        # A user's Python configuration must not redirect the bundled interpreter.
        for key in ("PYTHONHOME", "PYTHONPATH"):
            self.env.pop(key, None)
        root.title("DeeL Pyess")
        root.geometry("600x360")
        root.configure(bg="#080d16")
        tk.Label(root, text="DeeL Pyess", fg="#67b9ff", bg="#080d16", font=("Segoe UI", 26)).pack(
            pady=18
        )
        self.status = tk.StringVar(value="アプリを起動しています…")
        tk.Label(root, textvariable=self.status, fg="white", bg="#080d16", wraplength=550).pack(
            pady=8
        )
        self.buttons = []
        for text, command in (
            ("アプリを開く", lambda: webbrowser.open(self.url)),
            ("画像・動画の学習機能を追加", lambda: self.install(False)),
            ("画像・動画・日本語音声・声質機能を追加", lambda: self.install(True)),
            ("診断ログを開く", lambda: os.startfile(self.log_path)),
        ):
            button = tk.Button(root, text=text, command=command, width=46)
            button.pack(pady=4)
            self.buttons.append(button)
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
                    if len(models) == 2:
                        self.events.put(("ready", browser))
                        return
                except (OSError, ValueError):
                    time.sleep(0.2)
            if not self.closing:
                self.events.put(("error", "起動できませんでした。診断ログをご確認ください。"))

        threading.Thread(target=wait, daemon=True).start()

    def stop(self):
        if self.process and self.process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.process.wait(timeout=15)

    def install(self, voice):
        if self.installing:
            return
        self.installing = True
        self.stop()
        for button in self.buttons[:3]:
            button.configure(state="disabled")
        self.status.set("追加機能をダウンロード中です。数GB以上になるため時間がかかります…")

        def work():
            try:
                requirements = "requirements-voice.txt" if voice else "requirements-gpu.txt"
                self.installer = self.spawn(["-m", "pip", "install", "-r", requirements])
                if self.installer.wait() != 0:
                    raise RuntimeError(
                        "追加機能のインストールに失敗しました。診断ログをご確認ください。"
                    )
                if voice:
                    self.installer = self.spawn(["scripts/install-voice-backend.py"])
                    if self.installer.wait() != 0:
                        raise RuntimeError(
                            "声質機能のインストールに失敗しました。診断ログをご確認ください。"
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
                self.status.set("起動しました。ブラウザで画像の投稿・学習コースを使えます。")
                if value:
                    webbrowser.open(self.url)
            elif event == "installed":
                for button in self.buttons:
                    button.configure(state="normal")
                self.status.set("追加機能の準備ができました。再起動しています…")
                self.start()
            else:
                for button in self.buttons:
                    button.configure(state="normal")
                self.status.set(value)
                if not self.smoke:
                    messagebox.showerror("DeeL Pyess", value + "\n" + str(self.log_path))
        self.root.after(200, self.poll)

    def close(self):
        if self.installing:
            messagebox.showinfo("DeeL Pyess", "追加機能のインストールが終わるまでお待ちください。")
            return
        self.closing = True
        self.stop()
        self.log.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    home = (
        Path(sys.executable).parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1]
    )
    root = tk.Tk()
    app = Application(root, home)
    app.smoke = args.smoke_test
    app.start(browser=not args.smoke_test)
    if args.smoke_test:
        # Exercise the same GUI and server launch path as a real double-click.
        root.withdraw()
        deadline = time.monotonic() + 75

        outcome = {"passed": False}

        def check():
            if app.status.get().startswith("起動しました"):
                with urllib.request.urlopen(app.url + "/", timeout=5) as response:
                    assert b"DeeL" in response.read()
                with urllib.request.urlopen(app.url + "/api/models", timeout=5) as response:
                    assert {model["kind"] for model in json.load(response)} == {"image", "video"}
                outcome["passed"] = True
                (app.data / "windows-smoke.json").write_text(
                    json.dumps(
                        {"passed": True, "python": str(app.python), "data": str(app.data / "data")}
                    ),
                    encoding="utf-8",
                )
                app.close()
            elif time.monotonic() > deadline or app.status.get().startswith("起動できません"):
                app.close()
            else:
                root.after(200, check)

        root.after(200, check)
    root.mainloop()
    if args.smoke_test and not outcome["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--smoke-test" not in sys.argv:
            messagebox.showerror("DeeL Pyess", str(error))
        raise
