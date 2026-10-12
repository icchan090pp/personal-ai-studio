"""Google Interactions image editing; keys stay local and never enter job records."""

import base64
import ctypes
import io
import json
import os
import tempfile
from pathlib import Path

import httpx
from PIL import Image

MODELS = {
    "gemini-nano-banana-2.1": "Google Nano Banana 2.1",
    "gemini-3-pro-image": "Google Nano Banana Pro",
}
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
MAX_RESPONSE = 40 * 1024 * 1024


def protect(data, decrypt=False):
    """Windows DPAPI binds the key to the signed-in Windows account."""
    if os.name != "nt":
        return data
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if decrypt:
        success = crypt.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)
        )
    else:
        success = crypt.CryptProtectData(
            ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)
        )
    if not success:
        raise OSError("WindowsのAPIキー保護を利用できません")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(ctypes.cast(target.data, ctypes.c_void_p))


def save_key(root, key):
    path = Path(root) / "google-api-key.bin"
    if not key:
        path.unlink(missing_ok=True)
        return
    data = protect(key.encode("utf-8"))
    with tempfile.NamedTemporaryFile(dir=root, prefix="google-key-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if os.name != "nt":
        path.chmod(0o600)


def load_key(root):
    path = Path(root) / "google-api-key.bin"
    return protect(path.read_bytes(), decrypt=True).decode("utf-8") if path.is_file() else ""


def edit(image, prompt, model, resolution, key, client_factory=httpx.Client):
    if model not in MODELS or resolution not in {"1K", "2K", "4K"}:
        raise ValueError("Googleの画像モデル・出力サイズを確認してください")
    if not key:
        raise ValueError("Google APIキーを『仕上げ』の設定から登録してください")
    data = io.BytesIO()
    source = image.convert("RGB")
    source.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    source.save(data, format="PNG")
    payload = {
        "model": model,
        "store": False,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(data.getvalue()).decode("ascii"),
            },
        ],
        "response_format": {"type": "image", "mime_type": "image/png", "image_size": resolution},
    }
    try:
        # Never retry a billable POST automatically; redirects must not receive the API key.
        with client_factory(
            timeout=httpx.Timeout(180, connect=15), follow_redirects=False
        ) as client:
            with client.stream(
                "POST", ENDPOINT, headers={"x-goog-api-key": key}, json=payload
            ) as response:
                if response.status_code != 200:
                    errors = {
                        400: "画像・指示・モデル設定を確認してください。",
                        401: "APIキーを確認してください。",
                        403: "APIキー・利用権限・課金設定を確認してください。",
                        404: "このGoogleモデルを利用できません。別のモデルを選んでください。",
                        429: "Googleの利用上限に達しました。課金・上限を確認してから再実行してください。",
                    }
                    raise ValueError(
                        "Google API："
                        + errors.get(
                            response.status_code,
                            "接続先が処理を完了できませんでした。時間をおいて確認してください。",
                        )
                    )
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_RESPONSE:
                        raise ValueError("Googleの画像応答が容量上限を超えています")
        result = json.loads(chunks)
    except httpx.TimeoutException:
        raise ValueError(
            "Googleの応答待ちが時間切れです。処理済みでも課金される場合があります。自動再送はしていません。"
        ) from None
    except (httpx.HTTPError, json.JSONDecodeError):
        raise ValueError(
            "Googleへの接続・応答を確認できませんでした。APIキーはログへ保存していません。"
        ) from None
    if result.get("status") != "completed":
        raise ValueError("Googleの画像編集が完了しませんでした。指示を見直して再実行してください。")
    for step in result.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []):
            if part.get("type") == "image" and part.get("data"):
                try:
                    raw = base64.b64decode(part["data"], validate=True)
                    with Image.open(io.BytesIO(raw)) as output:
                        if output.width * output.height > 25_000_000:
                            raise ValueError("画像の容量上限を超えています")
                        output.load()
                        return output.convert("RGB"), {
                            "model": model,
                            "requested_resolution": resolution,
                            "input_size": list(source.size),
                            "output_size": list(output.size),
                        }
                except (OSError, ValueError):
                    raise ValueError("Googleから有効な画像を取得できませんでした。") from None
    raise ValueError("Googleから画像が返されませんでした。指示を変更してください。")
