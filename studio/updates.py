"""Check official releases and download an installer only after SHA256 verification."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

CURRENT_VERSION = "0.1.6"
REPOSITORY = "icchan090pp/personal-ai-studio"
RELEASES = f"https://api.github.com/repos/{REPOSITORY}/releases?per_page=30"
DOWNLOADS = f"https://github.com/{REPOSITORY}/releases/download/"
MAX_INSTALLER = 1024**3


def version(value):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else None


def open_url(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "DeeL-Pyess-Updater"}), timeout=30
    )


def select_release(releases, current=CURRENT_VERSION):
    candidates = []
    for release in releases:
        tag = release.get("tag_name", "")
        parsed = version(tag)
        if release.get("draft") or not parsed or not tag.startswith("v"):
            continue
        filename = f"DeeL-Pyess-Setup-{tag}-Windows-x64.exe"
        assets = {asset.get("name"): asset for asset in release.get("assets", [])}
        installer, checksum = assets.get(filename, {}), assets.get("SHA256SUMS.txt", {})
        prefix = DOWNLOADS + tag + "/"
        if (
            installer.get("browser_download_url") != prefix + filename
            or checksum.get("browser_download_url") != prefix + "SHA256SUMS.txt"
            or installer.get("state") != "uploaded"
            or checksum.get("state") != "uploaded"
            or not 0 < installer.get("size", 0) <= MAX_INSTALLER
        ):
            continue
        candidates.append((parsed, tag, filename, installer["size"]))
    if not candidates:
        raise ValueError("公開済みの更新ファイルが見つかりません。後でもう一度お試しください。")
    parsed, tag, filename, size = max(candidates)
    return {
        "current": current,
        "latest": tag[1:],
        "available": parsed > version(current),
        "tag": tag,
        "filename": filename,
        "size": size,
        "release_url": f"https://github.com/{REPOSITORY}/releases/tag/{tag}",
    }


def check_update(current=CURRENT_VERSION):
    with open_url(RELEASES) as response:
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("更新情報が大きすぎます")
    releases = json.loads(raw)
    if not isinstance(releases, list):
        raise ValueError("更新情報の形式を確認できません")
    return select_release(releases, current)


def download_update(release, directory, progress=lambda _value: None):
    tag = release["tag"]
    if not version(tag) or not tag.startswith("v"):
        raise ValueError("不正な更新バージョンです")
    filename = f"DeeL-Pyess-Setup-{tag}-Windows-x64.exe"
    size = release["size"]
    if not 0 < size <= MAX_INSTALLER:
        raise ValueError("更新ファイルのサイズが不正です")
    prefix = DOWNLOADS + tag + "/"
    with open_url(prefix + "SHA256SUMS.txt") as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("更新ファイルの検証情報が大きすぎます")
    matches = re.findall(r"^([a-fA-F0-9]{64})\s+" + re.escape(filename) + r"$", raw.decode(), re.M)
    if len(matches) != 1:
        raise ValueError("更新ファイルのチェックサムを確認できません")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (filename + ".part")
    destination = directory / filename
    digest, total = hashlib.sha256(), 0
    try:
        with open_url(prefix + filename) as response, temporary.open("wb") as output:
            while chunk := response.read(1024**2):
                total += len(chunk)
                if total > size:
                    raise ValueError("更新ファイルのサイズが一致しません")
                output.write(chunk)
                digest.update(chunk)
                progress(total * 100 // size)
        if total != size or digest.hexdigest() != matches[0].lower():
            raise ValueError("更新ファイルの検証に失敗しました。現在のアプリは変更していません。")
        with temporary.open("rb") as stream:
            if stream.read(2) != b"MZ":
                raise ValueError("Windowsの更新プログラムを確認できません")
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
