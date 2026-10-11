"""Install the application into its own environment; no administrator access needed."""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DeeL Pyess installer (Python 3.12+)")
    parser.add_argument(
        "--gpu", action="store_true", help="Install the CUDA learning/generation stack"
    )
    args = parser.parse_args()
    if not (3, 12) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit(
            "Python 3.12 or 3.13 is required by the pinned model libraries: https://www.python.org/downloads/"
        )
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.create(environment, with_pip=True)
    dependencies = root / ("requirements-gpu.txt" if args.gpu else "requirements-app.txt")
    print(
        "Installing GPU learning libraries…" if args.gpu else "Installing the application…",
        flush=True,
    )
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(dependencies)], check=True)
    print("Installed. Run Start-Windows.bat or ./start.sh. Data is saved in data/.")


if __name__ == "__main__":
    main()
