"""OpenAI gpt-oss via an explicitly local Ollama runtime; no cloud API fees."""

import http.client
import json

MODELS = {"gpt-oss:20b": "gpt-oss-20b", "gpt-oss:120b": "gpt-oss-120b"}


def ask(model, prompt, level, *, memory="", examples=()):
    if model not in MODELS:
        raise ValueError("対応するgpt-ossモデルを選択してください")
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=600)
    messages = [{"role": "system", "content": memory}] if memory else []
    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        )
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": "high"
            if level in {"excellent", "ultra"}
            else "medium"
            if level == "power"
            else "low",
            "options": {"num_predict": 1024, "num_ctx": 8192, "num_batch": 64, "use_mmap": True},
            "keep_alive": "0s",
        }
    )
    try:
        connection.request("POST", "/api/chat", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise RuntimeError("Ollamaの応答が大きすぎます。短い依頼で再実行してください")
        if response.status != 200:
            # Map runtime failures to actionable messages without returning raw server internals.
            detail = raw.decode("utf-8", errors="replace").lower()
            if "memory" in detail or "allocate" in detail:
                raise RuntimeError(
                    "メモリが不足しました。他のアプリを閉じて再実行してください。20bはRAM16GB以上、120bは約80GB以上が目安です。育成内容は保持されています。"
                )
            if "llama-server" in detail or "runner" in detail:
                raise RuntimeError(
                    "Ollamaの実行環境を起動できませんでした。公式インストーラーでOllamaを更新・再インストールしてください。育成内容は保持されています。"
                )
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
