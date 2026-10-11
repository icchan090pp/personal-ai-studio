"""OpenAI gpt-oss via an explicitly local Ollama runtime; no cloud API fees."""

import http.client
import json

MODELS = {"gpt-oss:20b": "gpt-oss-20b", "gpt-oss:120b": "gpt-oss-120b"}


def ask(model, prompt, level):
    if model not in MODELS:
        raise ValueError("対応するgpt-ossモデルを選択してください")
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=180)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": "high"
            if level in {"excellent", "ultra"}
            else "medium"
            if level == "power"
            else "low",
            "options": {"num_predict": 2048},
        }
    )
    try:
        connection.request("POST", "/api/chat", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(raw) > 1024 * 1024:
            raise RuntimeError(
                "Ollamaで選択モデルを取得・起動してください（ollama pull " + model + "）"
            )
        try:
            result = json.loads(raw).get("message", {}).get("content", "").strip()
        except (ValueError, AttributeError) as exc:
            raise RuntimeError("Ollamaの回答形式を確認してください") from exc
        if not result:
            raise RuntimeError("モデルから回答を取得できませんでした")
        return result
    finally:
        connection.close()
