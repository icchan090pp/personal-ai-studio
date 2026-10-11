"""Install pinned MIT OpenVoice code with this app's compatible dependency set."""

import hashlib
import io
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
DIGEST = "d08cbc84f4ec7abc76f9dddb5bbb221e906e49cfe5febd37133152bdeacb8be4"


def main():
    url = f"https://github.com/myshell-ai/OpenVoice/archive/{COMMIT}.zip"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != DIGEST:
        raise RuntimeError("OpenVoiceソースのチェックサムが一致しません")
    with tempfile.TemporaryDirectory(prefix="deel-openvoice-") as tmp:
        folder = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                parts = Path(entry.filename).parts[1:]
                if not parts or (parts[0] != "openvoice" and parts != ("LICENSE",)):
                    continue
                if ".." in parts or entry.is_dir():
                    continue
                target = folder.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
        # Upstream setup.py forces numpy 1.22, which cannot support Python 3.12.
        # Runtime code is unchanged. Dependencies are pinned in requirements-voice.txt.
        (folder / "pyproject.toml").write_text("""[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "deel-openvoice-backend"
version = "0.0.1"
description = "Pinned OpenVoice MIT runtime for DeeL Pyess"
requires-python = ">=3.12"
license = {file = "LICENSE"}
[tool.setuptools.packages.find]
include = ["openvoice*"]
[tool.setuptools.package-data]
openvoice = ["**/*.txt", "**/*.json"]
""")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", str(folder)], check=True
        )


if __name__ == "__main__":
    main()
