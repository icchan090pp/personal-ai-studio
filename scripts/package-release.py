"""Build a source installer ZIP without data, caches, tokens or virtual environments."""

import hashlib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
destination = root / "artifacts/release"
destination.mkdir(parents=True, exist_ok=True)
archive_path = destination / "DeeL-Pyess-v0.1.6.zip"
files = [
    root / file
    for file in (
        "README.md",
        "LICENSE",
        "install.py",
        "launch.py",
        "install.sh",
        "start.sh",
        "start-phone.sh",
        "Install-Windows.bat",
        "Start-Windows.bat",
        "Start-Phone.bat",
        "requirements-app.txt",
        "requirements-gpu.txt",
        "requirements-directml.txt",
        "requirements-desktop.txt",
        "requirements-audio.txt",
        "requirements-voice.txt",
        "requirements-dev.txt",
        "pyproject.toml",
    )
]
for directory in ("studio", "docs", "tests", "scripts", "website", "windows"):
    files.extend(
        path
        for path in (root / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(files):
        archive.write(file, "DeeL-Pyess/" + str(file.relative_to(root)))
checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
(destination / "SHA256SUMS.txt").write_text(
    checksum + "  " + archive_path.name + "\n", encoding="utf-8"
)
print(f"Packaged {len(files)} files; {archive_path.stat().st_size} bytes; SHA256 {checksum}")
