"""Exercise installed CPU image learning with real public weights and worker processes."""

import argparse
import io
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.app.resolve()))
    import torch
    from fastapi.testclient import TestClient
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file

    from studio import app as installed

    assert Path(installed.__file__).resolve().is_relative_to(args.app.resolve())
    assert not torch.cuda.is_available(), "This smoke proves the CUDA-free path"
    root = args.data.resolve()
    records = []
    with TestClient(installed.create_app(root)) as client:
        pid = client.post(
            "/api/projects", json={"name": "CPU育成試験", "trigger": "circleart"}
        ).json()["id"]
        files = []
        for i, color in enumerate(["#4dc9ed", "#54b1ff", "#6587ef"]):
            image = Image.new("RGB", (256, 256), "#111827")
            ImageDraw.Draw(image).ellipse((30 + i * 10, 30, 200, 210), fill=color)
            data = io.BytesIO()
            image.save(data, format="PNG")
            files.append(("files", (f"sample{i}.png", data.getvalue(), "image/png")))
        assert client.post(f"/api/projects/{pid}/images", files=files).status_code == 201

        def run(path, body):
            response = client.post(f"/api/projects/{pid}/{path}", json=body)
            assert response.status_code == 202, response.text
            job_id = response.json()["id"]
            started = time.monotonic()
            while time.monotonic() - started < 900:
                row = next(
                    j for j in client.get(f"/api/projects/{pid}/jobs").json() if j["id"] == job_id
                )
                if row["status"] not in ("queued", "running"):
                    break
                time.sleep(1)
            folder = root / "jobs" / job_id
            assert row["status"] == "completed", (
                row,
                (folder / "worker.log").read_text(encoding="utf-8", errors="replace")[-4000:],
            )
            evidence = json.loads((folder / "execution.json").read_text(encoding="utf-8"))
            assert evidence["backend"] == "cpu"
            records.append({"id": job_id, **evidence})
            print(json.dumps(records[-1]), flush=True)
            return job_id

        first = run(
            "train", {"training_model": "tiny", "steps": 10, "rank": 4, "related_images": True}
        )
        related = json.loads(
            (root / "jobs" / first / "related-report.json").read_text(encoding="utf-8")
        )
        assert related["stage"] in {"ready", "no_matches"}, related
        state = load_file(root / "jobs" / first / "adapter/pytorch_lora_weights.safetensors")
        assert all(torch.isfinite(t).all() for t in state.values())
        assert any(t.abs().sum() > 0 for k, t in state.items() if "lora.up" in k or "lora_B" in k)
        continued = run(
            "train", {"training_model": "tiny", "start_job_id": first, "steps": 10, "rank": 8}
        )
        after = load_file(root / "jobs" / continued / "adapter/pytorch_lora_weights.safetensors")
        assert state.keys() == after.keys()
        assert any(not torch.equal(state[k], after[k]) for k in state)
        generated = run(
            "generate",
            {
                "trained_job_id": continued,
                "compare_before": True,
                "prompt": "a blue circle in the style of circleart",
                "resolution": "native",
            },
        )
        result = client.get(f"/api/jobs/{generated}/result?download=true")
        assert result.status_code == 200
        with Image.open(io.BytesIO(result.content)) as image:
            assert image.size == (256, 256) and len(set(image.getdata())) > 100
        before = client.get(f"/api/jobs/{generated}/before?download=true")
        assert before.status_code == 200 and before.content != result.content
        assert client.get(f"/api/projects/{pid}/jobs").json()[0]["has_comparison"]
        assert client.get(f"/api/jobs/{continued}/result?download=true").content.startswith(b"PK")
    # Reopen the database as an upgrade/restart would, and retain both learned versions.
    with TestClient(installed.create_app(root, run_workers=False)) as client:
        jobs = client.get(f"/api/projects/{pid}/jobs").json()
        assert len([j for j in jobs if j["kind"] == "train" and j["status"] == "completed"]) == 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "passed": True,
                "cuda": False,
                "jobs": records,
                "adapter_changed": True,
                "restart_preserved": True,
                "related_discovery": related,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
