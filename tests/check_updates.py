"""Updater regression assertions included in the existing readiness case."""

import hashlib
import io
import json

import pytest

from studio import updates


def release(tag, content=b"MZverified-installer"):
    filename = f"DeeL-Pyess-Setup-{tag}-Windows-x64.exe"
    return {
        "tag_name": tag,
        "draft": False,
        "prerelease": True,
        "assets": [
            {
                "name": name,
                "state": "uploaded",
                "size": len(content),
                "browser_download_url": updates.DOWNLOADS + tag + "/" + name,
            }
            for name in (filename, "SHA256SUMS.txt")
        ],
    }


def check_updates(root, monkeypatch):
    newer, old = release("v0.1.8"), release("v0.1.3")
    assert updates.select_release([old, newer])["available"]
    assert not updates.select_release([old])["available"]
    assert not updates.select_release([release("v0.1.7")])["available"]
    assert updates.select_release([release("v0.1.10"), newer])["latest"] == "0.1.10"
    bad = release("v0.1.9")
    bad["assets"][0]["browser_download_url"] = "https://example.com/malicious.exe"
    assert updates.select_release([newer, bad])["latest"] == "0.1.8"
    assert updates.version("../../evil") is None
    with pytest.raises(ValueError):
        updates.select_release([{"tag_name": "v0.1.9", "assets": []}])
    content = b"MZverified-installer"
    item = updates.select_release([newer])
    expected = hashlib.sha256(content).hexdigest()
    checksum = f"{expected}  {item['filename']}\n".encode()
    with monkeypatch.context() as patch:
        patch.setattr(
            updates,
            "open_url",
            lambda url: (
                io.BytesIO(json.dumps([newer]).encode())
                if url == updates.RELEASES
                else io.BytesIO(checksum if url.endswith(".txt") else content)
            ),
        )
        assert updates.check_update()["latest"] == "0.1.8"
        progress = []
        path = updates.download_update(item, root / "updates", progress.append)
        assert path.read_bytes() == content and progress[-1] == 100
        patch.setattr(
            updates,
            "open_url",
            lambda url: io.BytesIO(checksum if url.endswith(".txt") else b"MZcorrupt-installer!"),
        )
        with pytest.raises(ValueError):
            updates.download_update(item, root / "updates")
        assert path.read_bytes() == content
        assert not list((root / "updates").glob("*.part"))
        patch.setattr(updates, "open_url", lambda url: io.BytesIO(b"no valid checksum"))
        with pytest.raises(ValueError):
            updates.download_update(item, root / "updates")
