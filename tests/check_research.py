"""Think context, real research parsing, citations, and unavailable-source behavior."""

import json

from studio.research import research
from tests.test_studio_matrix import ready_dataset


def check_research(client, root, monkeypatch):
    from studio import assistant
    from studio import research as module

    pid = ready_dataset(client)
    calls = []

    def fetch(url, limit):
        assert url.startswith("https://huggingface.co/") and limit == 2 * 1024 * 1024
        return (
            b"<nav>navigation</nav><script>bad instructions</script><h1>LoRA</h1><p>Train low rank adapters.</p>",
            "text/html",
            url,
        )

    def discover(options):
        assert options["query"] == "anime" and options["media"] in {"image", "video"}
        return [
            {
                "caption": "Anime example",
                "page": "https://commons.wikimedia.org/wiki/File:Example.png",
                "license": "CC0",
            }
        ]

    def ask(model, prompt, level, **kwargs):
        context = json.loads(prompt.split("現在の学習データ・参考資料：\n", 1)[1])
        calls.append(context)
        assert level == "excellent" and "参照データ" in kwargs["memory"]
        return "10ステップで前後を比較してください。[1]"

    monkeypatch.setattr(assistant, "ask", ask)
    monkeypatch.setattr(
        module, "research", lambda target, query: research(target, query, fetch, discover)
    )
    for mode in ("think", "research"):
        for target in ("image", "video"):
            response = client.post(
                f"/api/projects/{pid}/assistant",
                json={
                    "mode": mode,
                    "target": target,
                    "prompt": "画風を育てたい",
                    "research_query": "anime",
                },
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["mode"] == mode and calls[-1]["target"] == target
            assert calls[-1]["dataset_count"] == (3 if target == "image" else 0)
            if mode == "research":
                assert len(result["sources"]) == 3 and not result["warnings"]
                assert "LoRA" in result["sources"][0]["text"]
                assert "bad instructions" not in result["sources"][0]["text"]
                assert result["sources"][-1]["license"] == "CC0"
            else:
                assert result["sources"] == []
    messages = client.get(f"/api/projects/{pid}/chat/training").json()
    assert len(messages) == 4 and "RESEARCH" in messages[-1]["content"]
    assert (
        client.post(
            f"/api/projects/{pid}/assistant", json={"mode": "sync", "prompt": "help"}
        ).status_code
        == 422
    )

    def unavailable(*args):
        raise OSError("unavailable")

    failure = research("image", "anime", unavailable, unavailable)
    assert not failure["sources"] and len(failure["warnings"]) == 3
