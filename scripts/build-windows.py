"""Build a complete relocatable Windows runtime, GUI launcher and installer."""

import base64
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bundle = root / "build/windows"
if os.name != "nt":
    raise SystemExit("Build this on Windows, not a cross-compiled Linux runtime")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)
runtime = bundle / "runtime"
bootstrap = bundle / "MicrosoftEdgeWebview2Setup.exe"
with urllib.request.urlopen(
    "https://go.microsoft.com/fwlink/p/?LinkId=2124703", timeout=60
) as response:
    content = response.read(10 * 1024**2 + 1)
if not content.startswith(b"MZ") or len(content) > 10 * 1024**2:
    raise RuntimeError("Invalid WebView2 runtime bootstrapper")
bootstrap.write_bytes(content)
script = (
    "$s=Get-AuthenticodeSignature -LiteralPath '"
    + str(bootstrap).replace("'", "''")
    + "'; if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notlike '*Microsoft Corporation*') { throw 'Invalid Microsoft signature' }"
)
subprocess.run(
    [
        shutil.which("pwsh") or "powershell",
        "-NoProfile",
        "-EncodedCommand",
        base64.b64encode(script.encode("utf-16le")).decode(),
    ],
    env={key: value for key, value in os.environ.items() if key.lower() != "psmodulepath"},
    check=True,
)
shutil.copytree(sys.base_prefix, runtime, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for directory in ("studio", "docs/licenses"):
    shutil.copytree(
        root / directory, bundle / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
(bundle / "scripts").mkdir()
shutil.copy2(root / "scripts/install-voice-backend.py", bundle / "scripts")
for filename in (
    "launch.py",
    "LICENSE",
    "README.md",
    "requirements-app.txt",
    "requirements-gpu.txt",
    "requirements-directml.txt",
    "requirements-desktop.txt",
    "requirements-audio.txt",
    "requirements-voice.txt",
):
    shutil.copy2(root / filename, bundle)
subprocess.run(
    [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--paths",
        str(root),
        "--name",
        "DeeL Pyess",
        "--distpath",
        str(bundle),
        "--workpath",
        str(root / "build/pyinstaller"),
        str(root / "windows/desktop.py"),
    ],
    cwd=root,
    check=True,
)
compiler = Path(os.environ.get("ISCC", r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"))
subprocess.run([str(compiler), str(root / "windows/setup.iss")], check=True)
