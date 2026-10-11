"""Install fixed CPU/CUDA packages and an isolated optional DirectML runtime."""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

from studio.hardware import adapter_names, inspect_system


def installation_plan(root, backend, voice=False):
    index = "https://download.pytorch.org/whl/" + ("cu128" if backend == "cuda" else "cpu")
    suffix = "cu128" if backend == "cuda" else "cpu"
    torch_packages = [
        f"torch==2.7.0+{suffix}",
        f"torchvision==0.22.0+{suffix}",
        f"torchaudio==2.7.0+{suffix}",
    ]
    return [
        ["-m", "pip", "install", "--upgrade", *torch_packages, "--index-url", index],
        [
            "-m",
            "pip",
            "install",
            "-r",
            str(root / ("requirements-voice.txt" if voice else "requirements-gpu.txt")),
        ],
        *([[str(root / "scripts/install-voice-backend.py")]] if voice else []),
        ["-m", "pip", "check"],
    ]


def main():
    # Redirected output on English Windows defaults to cp1252. Keep Japanese
    # diagnostics and child installer logs usable regardless of system locale.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    parser.add_argument("--voice", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    backend = args.backend
    if backend == "auto":
        backend = (
            "cuda"
            if any("nvidia" in name.lower() for name in adapter_names())
            else "directml"
            if os.name == "nt"
            else "cpu"
        )
    print(f"AI機能の準備: {backend} / 数GBの空き容量とネット接続が必要です。", flush=True)
    for command in installation_plan(root, backend, args.voice):
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    if backend == "directml":
        if os.name != "nt" or sys.version_info[:2] != (3, 12):
            print(
                "DirectMLはWindows/Python 3.12で利用できます。CPU機能を準備しました。", flush=True
            )
        else:
            directory = Path(os.environ.get("LOCALAPPDATA", str(root))) / "DeeL Pyess/directml"
            python = directory / "Scripts/python.exe"
            try:
                usable = (
                    python.exists()
                    and subprocess.run(
                        [str(python), "-c", "import sys"], capture_output=True
                    ).returncode
                    == 0
                )
                if not usable:
                    venv.create(directory, with_pip=True, clear=directory.exists())
                subprocess.run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(root / "requirements-directml.txt"),
                    ],
                    cwd=root,
                    check=True,
                )
                subprocess.run([str(python), "-m", "pip", "check"], check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                print(
                    f"DirectMLの準備に失敗したためCPU機能を使用します。再試行できます: {exc}",
                    flush=True,
                )
    inspect_system.cache_clear()
    print(inspect_system()["message"], flush=True)
    print("セットアップ完了。", flush=True)


if __name__ == "__main__":
    main()
