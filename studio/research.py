"""Bounded, cited learning research using official documentation and public media metadata."""

from html.parser import HTMLParser
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from studio.collection import discover, public_address

REFERENCES = {
    "image": [
        (
            "Diffusers：LoRA追加学習",
            "https://huggingface.co/docs/diffusers/v0.33.1/en/training/lora",
        ),
        (
            "DreamShaper 8：配布元の説明",
            "https://huggingface.co/Lykon/dreamshaper-8/raw/main/README.md",
        ),
    ],
    "video": [
        (
            "Diffusers：AnimateDiff",
            "https://huggingface.co/docs/diffusers/v0.33.1/en/api/pipelines/animatediff",
        ),
        (
            "DreamShaper 8：配布元の説明",
            "https://huggingface.co/Lykon/dreamshaper-8/raw/main/README.md",
        ),
    ],
}


def fetch_reference(url, limit):
    """Fixed trusted documentation hosts use the proxy's normal hostname CONNECT."""

    def validate(value):
        parsed = urlsplit(value)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError("学習資料の配布元が不正です")
        public_address(value)

    class Redirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            validate(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    validate(url)
    request = Request(url, headers={"User-Agent": "DeeL-Pyess/0.1.7 learning-research"})
    with build_opener(Redirect()).open(request, timeout=12) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("学習資料が容量上限を超えています")
        return data, response.headers.get_content_type(), response.url


class ReadableText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            self.parts.clear()
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def research(target, query, fetcher=fetch_reference, discoverer=discover):
    sources, warnings = [], []
    for title, url in REFERENCES[target]:
        try:
            raw, kind, final = fetcher(url, 2 * 1024 * 1024)
            content = raw.decode("utf-8", errors="replace")
            if "html" in kind:
                parser = ReadableText()
                parser.feed(content)
                content = " ".join(parser.parts)
            sources.append({"title": title, "url": final, "kind": "guide", "text": content[:2800]})
        except (OSError, ValueError):
            warnings.append(title + "を取得できませんでした。")
    if query.strip():
        try:
            for row in discoverer({"provider": "commons", "query": query.strip(), "media": target})[
                :5
            ]:
                sources.append(
                    {
                        "title": row["caption"],
                        "url": row["page"],
                        "kind": "material",
                        "license": row["license"],
                        "text": row["caption"] + " / " + row["license"],
                    }
                )
        except (OSError, ValueError):
            warnings.append("学習素材の検索に失敗しました。時間をおいて再実行してください。")
    return {"sources": sources, "warnings": warnings, "query": query, "target": target}
