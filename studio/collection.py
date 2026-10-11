"""Public-site media collection with pinned DNS, bounded downloads and provenance."""

import base64
import http.client
import io
import ipaddress
import json
import os
import socket
import ssl
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from PIL import Image, ImageOps

LIMIT = 20 * 1024 * 1024


def public_address(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("公開HTTP/HTTPSサイトのURLを指定してください")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("標準のHTTP/HTTPSポートを指定してください")
    answers = socket.getaddrinfo(
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    ips = {answer[4][0] for answer in answers}
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError("ローカル・内部ネットワークのURLは収集できません")
    return parsed, sorted(ips)[0]


def fetch_public(url, max_bytes=LIMIT):
    for _ in range(4):
        parsed, ip = public_address(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        proxy = os.getenv("HTTPS_PROXY" if parsed.scheme == "https" else "HTTP_PROXY", "")
        proxy_headers = {}
        if proxy:
            gateway = urlsplit(proxy)
            if gateway.scheme != "http" or not gateway.hostname:
                raise ValueError("HTTP形式の管理用プロキシを設定してください")
            if gateway.username:
                value = ((gateway.username or "") + ":" + (gateway.password or "")).encode()
                proxy_headers["Proxy-Authorization"] = "Basic " + base64.b64encode(value).decode()
            conn = http.client.HTTPConnection(gateway.hostname, gateway.port or 80, timeout=8)
            if parsed.scheme == "https":
                conn.set_tunnel(ip, port, headers=proxy_headers)
            conn.connect()
        else:
            conn = http.client.HTTPConnection(parsed.hostname, port, timeout=8)
            # Pin the exact validated address, retaining TLS hostname verification.
            conn.sock = socket.create_connection((ip, port), timeout=8)
        if parsed.scheme == "https":
            conn.sock = ssl.create_default_context().wrap_socket(
                conn.sock, server_hostname=parsed.hostname
            )
        try:
            path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
            if proxy and parsed.scheme == "http":
                path = urlunsplit(("http", ip, parsed.path or "/", parsed.query, ""))
            conn.request(
                "GET",
                path,
                headers={
                    "User-Agent": "DeeL-Pyess/0.1 media-collector",
                    "Accept": "*/*",
                    "Host": parsed.netloc,
                    **(proxy_headers if proxy and parsed.scheme == "http" else {}),
                },
            )
            response = conn.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200:
                raise ValueError(f"サイトがHTTP {response.status}を返しました")
            length = response.getheader("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("ファイルが容量上限を超えています")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("ファイルが容量上限を超えています")
            return data, response.getheader("Content-Type", "").split(";")[0], url
        finally:
            conn.close()
    raise ValueError("リダイレクト回数が上限を超えています")


class MediaPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.media = []
        self.pages = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"img", "video", "audio", "source"}:
            source = attrs.get("src") or attrs.get("data-src")
            if source:
                self.media.append((source, attrs.get("alt", "")))
        elif tag == "a" and attrs.get("href"):
            url = attrs["href"]
            if (
                urlsplit(url)
                .path.lower()
                .endswith(
                    (
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".mp4",
                        ".webm",
                        ".wav",
                        ".mp3",
                        ".ogg",
                        ".m4a",
                    )
                )
            ):
                self.media.append((url, attrs.get("title", "")))
            else:
                self.pages.append(url)


def discover(options, fetcher=fetch_public):
    """Search Commons or crawl at most three pages within the requested site."""
    query = options.get("query", "").strip()
    if options.get("provider") == "commons":
        search = query + (
            " filetype:video"
            if options.get("media") == "video"
            else " filetype:audio"
            if options.get("media") == "audio"
            else ""
            if options.get("media") == "all"
            else " filetype:bitmap"
        )
        url = (
            "https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6&gsrlimit=10&prop=imageinfo&iiprop=url%7Cextmetadata&gsrsearch="
            + quote(search)
        )
        data, _, _ = fetcher(url, 2 * 1024 * 1024)
        pages = json.loads(data).get("query", {}).get("pages", {})
        results = []
        for item in pages.values():
            info = item.get("imageinfo", [{}])[0]
            if info.get("url"):
                results.append(
                    {
                        "url": info["url"],
                        "page": info.get("descriptionurl", url),
                        "caption": item.get("title", ""),
                        "license": info.get("extmetadata", {})
                        .get("LicenseShortName", {})
                        .get("value", ""),
                    }
                )
        return results
    first = options["url"]
    queue, visited, results, seen = [first], set(), [], set()
    host = urlsplit(first).hostname
    while queue and len(visited) < 3:
        page = queue.pop(0)
        if page in visited:
            continue
        visited.add(page)
        try:
            data, content_type, final = fetcher(page, 2 * 1024 * 1024)
        except (OSError, ValueError):
            if page == first:
                raise
            continue
        if content_type.startswith(("image/", "video/", "audio/")):
            return [{"url": final, "page": final, "caption": query, "license": ""}]
        parser = MediaPage()
        parser.feed(data.decode("utf-8", errors="replace"))
        for source, description in parser.media:
            source = urljoin(final, source)
            if source not in seen and (
                not query or query.casefold() in (source + " " + description).casefold()
            ):
                seen.add(source)
                results.append(
                    {"url": source, "page": final, "caption": description, "license": ""}
                )
        for link in parser.pages:
            link = urljoin(final, link)
            if urlsplit(link).hostname == host and link not in visited and len(queue) < 10:
                queue.append(link)
    return results[:30]


def extract_media(data, content_type, kind):
    if content_type.startswith("video/"):
        if kind == "image":
            return []
        import imageio.v2 as imageio

        # ffmpeg only reads a temporary local file; it never receives a remote URL.
        with tempfile.TemporaryDirectory(prefix="deel-video-") as tmp:
            path = Path(tmp) / "input.mp4"
            path.write_bytes(data)
            with imageio.get_reader(path) as reader:
                metadata = reader.get_meta_data()
                width, height = metadata["size"]
                if width > 1920 or height > 1080:
                    raise ValueError("収集する動画は1920×1080以下にしてください")
                frames = []
                duration = min(float(reader.get_meta_data().get("duration", 3)), 30)
                fps = float(reader.get_meta_data().get("fps", 8))
                for seconds in (0, duration / 3, duration * 2 / 3):
                    try:
                        frames.append(Image.fromarray(reader.get_data(int(seconds * fps))))
                    except (IndexError, RuntimeError):
                        break
                return frames
    if kind == "video":
        return []
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if min(image.size) < 64 or image.width * image.height > 25_000_000:
            raise ValueError("画像サイズが対応範囲外です")
        return [ImageOps.exif_transpose(image).convert("RGB")]


def collect(options, fetcher=fetch_public):
    rows, errors = [], []
    for item in discover(options, fetcher):
        if len(rows) >= options.get("max_files", 10):
            break
        try:
            data, content_type, final = fetcher(item["url"], LIMIT)
            kind = options.get("media", "both")
            if kind in {"audio", "all"} and content_type.startswith(("audio/", "video/")):
                from studio.sound import read_audio

                with tempfile.TemporaryDirectory(prefix="deel-web-audio-") as tmp:
                    path = Path(tmp) / (
                        "input.mp4" if content_type.startswith("video/") else "input.wav"
                    )
                    path.write_bytes(data)
                    try:
                        waveform = read_audio(path)
                        rows.append({**item, "url": final, "image": None, "audio_data": waveform})
                    except Exception:
                        if kind == "audio":
                            raise
                        errors.append(
                            "音声トラックを取得できませんでした。動画・画像のみ取り込みます"
                        )
                if kind == "audio" or len(rows) >= options.get("max_files", 10):
                    continue
            if content_type.startswith("audio/") or kind == "audio":
                continue
            for frame_index, frame in enumerate(
                extract_media(data, content_type, options.get("media", "both"))
            ):
                frame = frame.convert("RGB")
                frame.thumbnail((2048, 2048))
                rows.append(
                    {
                        **item,
                        "url": final,
                        "image": frame,
                        "video_data": data
                        if content_type.startswith("video/") and frame_index == 0
                        else None,
                    }
                )
                if len(rows) >= options.get("max_files", 10):
                    break
        except Exception:
            errors.append("取得できない、または対応外のメディアを1件除外しました")
    return rows, errors
