"""Shared, versioned image finishing for every base model and video stills."""

import argparse
import io
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, SecretStr

from studio import google_images


def register_routes(
    app, root, query, project_or_404, enqueue, process_guard, active_processes, stop_process
):
    @app.post("/api/projects/{project_id}/finish/attachments", status_code=201)
    async def attachment(project_id: str, file: UploadFile = File(...)):
        project_or_404(project_id)
        data = await file.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "追加画像は20MB以下にしてください")
        try:
            with Image.open(io.BytesIO(data)) as original:
                if original.width * original.height > 25_000_000:
                    raise ValueError("too large")
                original.load()
                clean = ImageOps.exif_transpose(original).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "読み込めるPNG・JPEG・WebP画像を選んでください") from None
        folder = root / "projects" / project_id / "references"
        folder.mkdir(parents=True, exist_ok=True)
        if len(list(folder.glob("*.png"))) >= 100:
            raise HTTPException(422, "編集用の参照画像は100件までです")
        iid = uuid.uuid4().hex
        clean.save(folder / (iid + ".png"))
        return {
            "id": iid,
            "width": clean.width,
            "height": clean.height,
            "url": f"/api/projects/{project_id}/finish/attachments/{iid}",
        }

    def attachment_path(project_id, iid):
        project_or_404(project_id)
        if not iid or len(iid) != 32 or any(c not in "0123456789abcdef" for c in iid):
            raise HTTPException(404, "追加画像が見つかりません")
        path = root / "projects" / project_id / "references" / (iid + ".png")
        if not path.is_file():
            raise HTTPException(404, "追加画像が見つかりません")
        return path

    @app.get("/api/projects/{project_id}/finish/attachments/{iid}")
    def attachment_file(project_id: str, iid: str):
        return FileResponse(attachment_path(project_id, iid))

    class KeyInput(BaseModel):
        api_key: SecretStr

    @app.get("/api/system/google-images")
    def settings():
        return {
            "configured": (root / "google-api-key.bin").is_file(),
            "models": google_images.MODELS,
            "key_protection": "Windows DPAPI"
            if __import__("os").name == "nt"
            else "所有者のみ読めるローカルファイル",
        }

    @app.put("/api/system/google-images")
    def key(body: KeyInput, request: Request):
        if not request.client or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "APIキーはこのPCの画面から設定してください")
        value = body.api_key.get_secret_value().strip()
        if len(value) > 256 or any(c.isspace() for c in value):
            raise HTTPException(422, "APIキーの形式を確認してください")
        try:
            google_images.save_key(root, value)
        except OSError:
            raise HTTPException(
                503, "APIキーを保存できませんでした。PCの保存先を確認してください"
            ) from None
        return settings()

    class FinishInput(BaseModel):
        provider: Literal["google", "realesrgan"]
        source_job_id: str | None = None
        source_image_id: str | None = None
        source_attachment_id: str | None = None
        prompt: str = Field(default="", max_length=2000)
        google_model: Literal["gemini-nano-banana-2.1", "gemini-3-pro-image"] = (
            "gemini-nano-banana-2.1"
        )
        resolution: Literal["1K", "2K", "4K"] = "1K"
        enhancement_model: Literal["anime", "photo"] = "anime"
        scale: Literal[2, 4] = 2
        denoise: bool = False

    @app.post("/api/projects/{project_id}/finish", status_code=202)
    def finish(project_id: str, body: FinishInput):
        project_or_404(project_id)
        if (
            sum(
                bool(value)
                for value in (body.source_job_id, body.source_image_id, body.source_attachment_id)
            )
            != 1
        ):
            raise HTTPException(422, "仕上げる画像を1つ選択してください")
        kind = "image"
        if body.source_job_id:
            source = query(
                "SELECT id,kind FROM jobs WHERE id=? AND project_id=? AND status='completed'",
                (body.source_job_id, project_id),
                True,
            )
            if not source or source["kind"] not in {"image", "video", "finish"}:
                raise HTTPException(422, "このプロジェクトの完成した画像・動画を選択してください")
            kind = source["kind"]
            path = Path("jobs") / source["id"] / ("output.mp4" if kind == "video" else "output.png")
        elif body.source_image_id:
            source = query(
                "SELECT id FROM images WHERE id=? AND project_id=?",
                (body.source_image_id, project_id),
                True,
            )
            if not source:
                raise HTTPException(422, "このプロジェクトの画像を選択してください")
            path = Path("projects") / project_id / "images" / (source["id"] + ".png")
        else:
            path = attachment_path(project_id, body.source_attachment_id).relative_to(root)
        if not (root / path).is_file():
            raise HTTPException(404, "元の作品ファイルが見つかりません")
        if body.provider == "google":
            if not body.prompt.strip():
                raise HTTPException(422, "Googleに送る編集指示を入力してください")
            try:
                configured = bool(google_images.load_key(root))
            except OSError:
                configured = False
            if not configured:
                raise HTTPException(422, "Google APIキーを『仕上げ』の設定から登録してください")
        return enqueue(
            project_id,
            "finish",
            {
                **body.model_dump(),
                "finish_provider": body.provider,
                "source_path": str(path),
                "source_kind": kind,
            },
        )

    @app.post("/api/projects/{project_id}/finish/stop")
    def stop(project_id: str):
        project_or_404(project_id)
        with process_guard:
            rows = query(
                "SELECT id FROM jobs WHERE project_id=? AND kind='finish' AND status IN ('queued','running')",
                (project_id,),
            )
            for row in rows:
                query("UPDATE jobs SET status='cancelled', progress=0 WHERE id=?", (row["id"],))
                process = active_processes.get(row["id"])
                if process:
                    stop_process(process)
        return {
            "stopped": len(rows),
            "message": "PC側の仕上げ処理を停止しました。送信済みのGoogle処理・課金は取り消せません。",
        }


def source_image(root, config):
    path = root / config["source_path"]
    if config["source_kind"] == "video":
        import imageio.v2 as imageio

        with imageio.get_reader(str(path)) as reader:
            return Image.fromarray(reader.get_data(0)).convert("RGB")
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def execute(root, job_id):
    from studio.enhancement import enhance

    started = time.monotonic()
    db_path = root / "studio.sqlite3"
    with sqlite3.connect(db_path) as db:
        row = db.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
    config = json.loads(row[0])
    folder = root / "jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    source = source_image(root, config)
    source.save(folder / "before.png")

    def progress(value):
        with sqlite3.connect(db_path) as db:
            db.execute(
                "UPDATE jobs SET progress=? WHERE id=? AND status='running'", (value, job_id)
            )

    progress(5)
    if config["finish_provider"] == "google":
        output, details = google_images.edit(
            source,
            config["prompt"],
            config["google_model"],
            config["resolution"],
            google_images.load_key(root),
        )
    else:
        output, details = enhance(
            source, root, config["enhancement_model"], config["scale"], config["denoise"], progress
        )
    output.save(folder / "output.png")
    details.update(
        provider=config["finish_provider"],
        source_job_id=config.get("source_job_id"),
        source_image_id=config.get("source_image_id"),
        source_kind=config["source_kind"],
        elapsed_seconds=round(time.monotonic() - started, 2),
    )
    (folder / "execution.json").write_text(
        json.dumps(details, ensure_ascii=False), encoding="utf-8"
    )
    progress(100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        execute(args.data, args.job)
    except Exception as error:
        # Provider responses, request headers and API keys never enter the worker log.
        message = (
            str(error)
            if isinstance(error, ValueError)
            else "画像の仕上げに失敗しました。AI機能の導入、ネット接続と空きメモリを確認してください。"
        )
        folder = args.data / "jobs" / args.job
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "error.json").write_text(
            json.dumps({"message": message}, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
