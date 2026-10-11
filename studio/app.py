"""Single-user localhost application. GPU jobs execute serially in subprocesses."""

import hmac
import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from studio.collection import collect, public_address
from studio.curriculum import MODULES
from studio.presets import LEVELS

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VIDEO_MODEL = "THUDM/CogVideoX-5b-I2V"
MAX_BYTES = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 25_000_000


class ProjectInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    trigger: str = Field(default="myconcept", pattern=r"^[a-zA-Z][a-zA-Z0-9_]{2,39}$")


class TrainingInput(BaseModel):
    target: Literal["image", "video", "audio"] = "image"
    instruction: str = Field(default="", max_length=2000)
    level: Literal["normal", "power", "excellent"] | None = None
    mode: str = Field(
        default="supervised",
        pattern=r"^(supervised|semi_supervised|style|reinforcement|generation_test|imitation)$",
    )
    group: str | None = Field(default=None, max_length=40)
    steps: int = Field(default=500, ge=10, le=5000)
    learning_rate: float = Field(default=0.0001, ge=0.000001, le=0.001)
    rank: int = Field(default=8, ge=4, le=32)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class GenerationInput(BaseModel):
    resolution: Literal["native", "720p", "1080p", "1440p", "4k", "custom"] = "720p"
    level: Literal["normal", "power", "excellent"] | None = None
    kind: str = Field(default="image", pattern=r"^(image|video)$")
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field(default="", max_length=2000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    source_id: str | None = None
    with_audio: bool = False
    audio_prompt: str = Field(default="", max_length=2000)

    @field_validator("prompt")
    @classmethod
    def nonblank_prompt(cls, value):
        if not value.strip():
            raise ValueError("プロンプトを入力してください")
        return value


def create_app(data_dir: Path | None = None, run_workers: bool = True):
    root = Path(data_dir or os.getenv("STUDIO_DATA_DIR", "data")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "studio.sqlite3"
    lock = threading.Lock()
    access_token = os.getenv("STUDIO_ACCESS_TOKEN", "")

    def query(sql, args=(), one=False):
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.execute(sql, args)
            if cursor.description:
                rows = [dict(row) for row in cursor.fetchall()]
                return (rows[0] if rows else None) if one else rows

    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY, name TEXT, trigger TEXT, level TEXT DEFAULT 'normal', rules_prompt TEXT DEFAULT '', negative_rules TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY, project_id TEXT, caption TEXT,
                labeled INTEGER DEFAULT 0, rating INTEGER, origin_job_id TEXT, group_name TEXT DEFAULT 'default',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS training_metrics (
                job_id TEXT, step INTEGER, loss REAL, learning_rate REAL,
                gradient_norm REAL, lora_norm REAL, PRIMARY KEY(job_id,step));
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY, project_id TEXT, channel TEXT, role TEXT, content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, caption TEXT, group_name TEXT DEFAULT 'default', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY, project_id TEXT, status TEXT, imported INTEGER DEFAULT 0, error TEXT, config TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS image_sources (
                image_id TEXT PRIMARY KEY, url TEXT, page TEXT, license TEXT);
            CREATE TABLE IF NOT EXISTS course_progress (
                project_id TEXT, module_id TEXT, done INTEGER DEFAULT 0, notes TEXT DEFAULT '',
                PRIMARY KEY(project_id,module_id));
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, status TEXT,
                progress INTEGER DEFAULT 0, error TEXT, config TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

    with sqlite3.connect(db_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(images)")}
        for column, definition in {
            "labeled": "INTEGER DEFAULT 0",
            "rating": "INTEGER",
            "origin_job_id": "TEXT",
            "group_name": "TEXT DEFAULT 'default'",
        }.items():
            if column not in columns:
                db.execute(f"ALTER TABLE images ADD COLUMN {column} {definition}")

    with sqlite3.connect(db_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(projects)")}
        if "level" not in columns:
            db.execute("ALTER TABLE projects ADD COLUMN level TEXT DEFAULT 'normal'")
        for column in ("rules_prompt", "negative_rules", "training_instruction"):
            if column not in columns:
                db.execute(f"ALTER TABLE projects ADD COLUMN {column} TEXT DEFAULT ''")

    def project_or_404(project_id):
        project = query("SELECT * FROM projects WHERE id=?", (project_id,), one=True)
        if not project:
            raise HTTPException(404, "プロジェクトが見つかりません")
        return project

    def gpu_available():
        if not all(importlib.util.find_spec(m) for m in ("torch", "diffusers", "peft")):
            return False
        import torch

        return torch.cuda.is_available()

    def worker():
        # One process owns the GPU; multiple Uvicorn workers are unsupported.
        with lock:
            while job := query(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1", one=True
            ):
                query("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
                folder = root / "jobs" / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                try:
                    with (folder / "worker.log").open("w") as log:
                        result = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "studio.gpu",
                                "--data",
                                str(root),
                                "--job",
                                job["id"],
                            ],
                            stdout=log,
                            stderr=log,
                            check=False,
                        )
                    if result.returncode:
                        # Keep full diagnostics local rather than exposing paths or tokens in API.
                        raise RuntimeError(
                            "GPU処理に失敗しました。ジョブのログを確認してください。"
                        )
                    expected = {
                        "train": "adapter.zip",
                        "video_train": "adapter.zip",
                        "audio_train": "adapter.zip",
                        "image": "output.png",
                        "video": "output.mp4",
                    }
                    if not (folder / expected[job["kind"]]).is_file():
                        raise RuntimeError("出力ファイルを確認できません。再実行してください。")
                    query(
                        "UPDATE jobs SET status='completed', progress=100 WHERE id=?", (job["id"],)
                    )
                except Exception as exc:
                    query(
                        "UPDATE jobs SET status='failed', error=? WHERE id=?", (str(exc), job["id"])
                    )

    @asynccontextmanager
    async def lifespan(_app):
        query(
            "UPDATE jobs SET status='failed', error='アプリが再起動されました。再実行してください。' "
            "WHERE status='running'"
        )
        if run_workers:
            threading.Thread(target=worker, daemon=True).start()
        query(
            "UPDATE collections SET status='failed', error='再起動されました。再収集してください' WHERE status='running'"
        )
        yield

    app = FastAPI(title="DeeL Pyess", lifespan=lifespan)

    @app.middleware("http")
    async def local_access(request, call_next):
        if access_token and request.url.path != "/connect":
            if not hmac.compare_digest(request.cookies.get("deel_access", ""), access_token):
                return JSONResponse(
                    {"detail": "パソコンに表示された接続URLからアクセスしてください"},
                    status_code=401,
                )
        if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "別のサイトからの操作は許可されていません"}, status_code=403
                )
        return await call_next(request)

    @app.get("/connect")
    def connect(token: str = ""):
        if not access_token or not hmac.compare_digest(token, access_token):
            raise HTTPException(401, "接続コードを確認してください")
        response = RedirectResponse("/", status_code=303)
        response.set_cookie("deel_access", access_token, httponly=True, samesite="strict")
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    class InstructionInput(BaseModel):
        instruction: str = Field(min_length=1, max_length=2000)

    @app.get("/api/projects/{project_id}/chat/{channel}")
    def chat(project_id: str, channel: Literal["training", "generation"]):
        project_or_404(project_id)
        return query(
            "SELECT * FROM (SELECT rowid AS sequence,* FROM chat_messages WHERE project_id=? AND channel=? ORDER BY rowid DESC LIMIT 30) ORDER BY sequence",
            (project_id, channel),
        )

    @app.post("/api/projects/{project_id}/chat/training", status_code=201)
    def training_chat(project_id: str, body: InstructionInput):
        project_or_404(project_id)
        text = body.instruction.strip()
        if not text:
            raise HTTPException(422, "学習指示を入力してください")
        with sqlite3.connect(db_path) as db:
            db.execute("UPDATE projects SET training_instruction=? WHERE id=?", (text, project_id))
            db.execute(
                "INSERT INTO chat_messages(id,project_id,channel,role,content) VALUES(?,?,'training','user',?)",
                (uuid.uuid4().hex, project_id, text),
            )
            db.execute(
                "INSERT INTO chat_messages(id,project_id,channel,role,content) VALUES(?,?,'training','system',?)",
                (
                    uuid.uuid4().hex,
                    project_id,
                    "次の学習データの説明にこの指示を追加します。設定を確認し、学習をはじめるボタンで実行してください。",
                ),
            )
        return {"instruction": text}

    @app.get("/api/system")
    def system():
        return {
            "gpu_ready": gpu_available(),
            "base_model": BASE_MODEL,
            "video_model": VIDEO_MODEL,
            "checkpoint": os.getenv("STUDIO_CHECKPOINT", BASE_MODEL),
            "levels": LEVELS,
        }

    @app.get("/api/course/modules")
    def course_modules():
        return MODULES

    class CourseProgress(BaseModel):
        done: bool = False
        notes: str = Field(default="", max_length=3000)

    @app.get("/api/projects/{project_id}/course")
    def course_progress(project_id: str):
        project_or_404(project_id)
        return query(
            "SELECT module_id,done,notes FROM course_progress WHERE project_id=?", (project_id,)
        )

    @app.patch("/api/projects/{project_id}/course/{module_id}")
    def save_course_progress(project_id: str, module_id: str, body: CourseProgress):
        project_or_404(project_id)
        if module_id not in {module["id"] for module in MODULES}:
            raise HTTPException(404, "コースが見つかりません")
        query(
            "INSERT INTO course_progress(project_id,module_id,done,notes) VALUES(?,?,?,?) "
            "ON CONFLICT(project_id,module_id) DO UPDATE SET done=excluded.done,notes=excluded.notes",
            (project_id, module_id, int(body.done), body.notes),
        )
        return {"module_id": module_id, "done": body.done, "notes": body.notes}

    @app.get("/api/models")
    def model_catalog():
        return [
            {
                "id": BASE_MODEL,
                "name": "Stable Diffusion 1.5",
                "kind": "image",
                "trainable": True,
                "weights_url": f"https://huggingface.co/{BASE_MODEL}",
            },
            {
                "id": VIDEO_MODEL,
                "name": "CogVideoX-5b-I2V",
                "kind": "video",
                "trainable": True,
                "weights_url": f"https://huggingface.co/{VIDEO_MODEL}",
            },
        ]

    @app.get("/api/projects")
    def projects():
        return query(
            "SELECT p.*, (SELECT COUNT(*) FROM images i WHERE i.project_id=p.id) "
            "AS image_count FROM projects p ORDER BY p.rowid DESC"
        )

    @app.post("/api/projects", status_code=201)
    def create_project(body: ProjectInput):
        project_id = uuid.uuid4().hex
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "モデル名を入力してください")
        query(
            "INSERT INTO projects(id,name,trigger) VALUES(?,?,?)", (project_id, name, body.trigger)
        )
        (root / "projects" / project_id / "images").mkdir(parents=True)
        return project_or_404(project_id)

    class LevelInput(BaseModel):
        level: Literal["normal", "power", "excellent"]

    @app.patch("/api/projects/{project_id}/level")
    def set_level(project_id: str, body: LevelInput):
        project_or_404(project_id)
        query("UPDATE projects SET level=? WHERE id=?", (body.level, project_id))
        return {"level": body.level, "preset": LEVELS[body.level]}

    class RuleSettings(BaseModel):
        rules_prompt: str = Field(default="", max_length=1000)
        negative_rules: str = Field(default="", max_length=1000)

    @app.get("/api/projects/{project_id}/settings")
    def get_settings(project_id: str):
        project = project_or_404(project_id)
        return {
            "rules_prompt": project["rules_prompt"],
            "negative_rules": project["negative_rules"],
        }

    @app.patch("/api/projects/{project_id}/settings")
    def set_settings(project_id: str, body: RuleSettings):
        project_or_404(project_id)
        query(
            "UPDATE projects SET rules_prompt=?, negative_rules=? WHERE id=?",
            (body.rules_prompt.strip(), body.negative_rules.strip(), project_id),
        )
        return get_settings(project_id)

    @app.get("/api/projects/{project_id}/images")
    def images(project_id: str):
        project_or_404(project_id)
        return query(
            "SELECT images.*,image_sources.page AS source_page FROM images LEFT JOIN image_sources ON images.id=image_sources.image_id WHERE project_id=? ORDER BY images.rowid",
            (project_id,),
        )

    @app.post("/api/projects/{project_id}/images", status_code=201)
    async def upload(
        project_id: str,
        files: list[UploadFile] = File(...),
        group_name: str = Form(default="default", max_length=40),
    ):
        project = project_or_404(project_id)
        if not 1 <= len(files) <= 50:
            raise HTTPException(422, "一度に1〜50枚アップロードできます")
        count = query("SELECT COUNT(*) AS n FROM images WHERE project_id=?", (project_id,), True)[
            "n"
        ]
        if count + len(files) > 100:
            raise HTTPException(422, "1モデルにつき100枚までです")
        prepared = []
        try:
            for file in files:
                contents = await file.read(MAX_BYTES + 1)
                if len(contents) > MAX_BYTES:
                    raise HTTPException(413, "画像は1枚20MB以下にしてください")
                with Image.open(io.BytesIO(contents)) as original:
                    if original.format not in {"PNG", "JPEG", "WEBP"}:
                        raise HTTPException(422, "PNG・JPEG・WebPを使用してください")
                    if original.width * original.height > Image.MAX_IMAGE_PIXELS:
                        raise HTTPException(422, "画像は2500万画素以下にしてください")
                    if min(original.size) < 64:
                        raise HTTPException(422, "画像は縦横64px以上にしてください")
                    from PIL import ImageOps

                    clean = ImageOps.exif_transpose(original).convert("RGB")
                    clean.thumbnail((2048, 2048))
                    prepared.append(clean)
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise HTTPException(422, "読み込めない画像、または大きすぎる画像です") from exc
        finally:
            for file in files:
                await file.close()
        for clean in prepared:
            image_id = uuid.uuid4().hex
            folder = root / "projects" / project_id / "images"
            folder.mkdir(parents=True, exist_ok=True)
            clean.save(folder / f"{image_id}.png")
            query(
                "INSERT INTO images(id,project_id,caption,group_name) VALUES(?,?,?,?)",
                (
                    image_id,
                    project_id,
                    f"a photo of {project['trigger']}",
                    group_name.strip() or "default",
                ),
            )
        return images(project_id)

    @app.post("/api/projects/{project_id}/media", status_code=201)
    async def upload_media(
        project_id: str,
        files: list[UploadFile] = File(...),
        kind: Literal["video", "audio"] = Form(...),
        group_name: str = Form("default", max_length=40),
    ):
        project = project_or_404(project_id)
        if not 1 <= len(files) <= 10:
            raise HTTPException(422, "一度に1〜10ファイルを選択してください")
        if len(query("SELECT id FROM media WHERE project_id=?", (project_id,))) + len(files) > 30:
            raise HTTPException(422, "音声・動画は合計30ファイルまでです")
        import tempfile

        from studio.sound import read_audio, write_audio
        from studio.video_learning import normalise_video

        validated = []
        for file in files:
            data = await file.read(50 * 1024 * 1024 + 1)
            if len(data) > 50 * 1024 * 1024:
                raise HTTPException(413, "音声・動画は1ファイル50MBまでです")
            try:
                with tempfile.TemporaryDirectory(prefix="deel-upload-") as tmp:
                    path = Path(tmp) / ("input.mp4" if kind == "video" else "input.wav")
                    path.write_bytes(data)
                    if kind == "video":
                        output = Path(tmp) / "normalised.mp4"
                        normalise_video(data, output)
                        validated.append(output.read_bytes())
                    else:
                        validated.append(read_audio(path))
            except Exception as exc:
                raise HTTPException(
                    422, "有効な音声・動画を選択してください（1秒以上・動画1920×1080以下）"
                ) from exc
        folder = root / "projects" / project_id / "media"
        folder.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM media WHERE project_id=?", (project_id,)
            ).fetchone()[0]
            if count + len(validated) > 30:
                raise HTTPException(422, "音声・動画は合計30ファイルまでです")
            for data in validated:
                iid = uuid.uuid4().hex
                if kind == "video":
                    (folder / f"{iid}.mp4").write_bytes(data)
                else:
                    write_audio(folder / f"{iid}.wav", data)
                caption = (
                    f"a video of {project['trigger']}"
                    if kind == "video"
                    else f"sound of {project['trigger']}"
                )
                db.execute(
                    "INSERT INTO media(id,project_id,kind,caption,group_name) VALUES(?,?,?,?,?)",
                    (iid, project_id, kind, caption, group_name.strip() or "default"),
                )
        return query("SELECT * FROM media WHERE project_id=? ORDER BY rowid", (project_id,))

    @app.get("/api/projects/{project_id}/media")
    def media_list(project_id: str):
        project_or_404(project_id)
        return query("SELECT * FROM media WHERE project_id=? ORDER BY rowid", (project_id,))

    class MediaCaption(BaseModel):
        caption: str = Field(min_length=1, max_length=1000)
        group_name: str = Field(default="default", min_length=1, max_length=40)

    @app.patch("/api/media/{media_id}")
    def media_caption(media_id: str, body: MediaCaption):
        if not query("SELECT id FROM media WHERE id=?", (media_id,), True):
            raise HTTPException(404)
        if not body.caption.strip():
            raise HTTPException(422, "説明を入力してください")
        query(
            "UPDATE media SET caption=?,group_name=? WHERE id=?",
            (body.caption.strip(), body.group_name.strip() or "default", media_id),
        )
        return {"ok": True}

    @app.get("/api/media/{media_id}/file")
    def media_file(media_id: str):
        row = query("SELECT * FROM media WHERE id=?", (media_id,), True)
        if not row:
            raise HTTPException(404)
        suffix = ".mp4" if row["kind"] == "video" else ".wav"
        return FileResponse(
            root / "projects" / row["project_id"] / "media" / (media_id + suffix),
            media_type="video/mp4" if row["kind"] == "video" else "audio/wav",
        )

    class CollectionInput(BaseModel):
        provider: Literal["site", "commons"] = "site"
        url: str = Field(default="", max_length=2000)
        query: str = Field(default="", max_length=200)
        media: Literal["image", "video", "both"] = "both"
        group: str = Field(default="web", min_length=1, max_length=40)
        max_files: int = Field(default=10, ge=1, le=10)

    @app.post("/api/projects/{project_id}/collect", status_code=202)
    def start_collection(project_id: str, body: CollectionInput):
        project = project_or_404(project_id)
        if body.provider == "site":
            try:
                public_address(body.url)
            except (OSError, ValueError) as exc:
                raise HTTPException(422, "公開サイトのURLを確認してください") from exc
        elif not body.query.strip():
            raise HTTPException(422, "検索キーワードを入力してください")
        cid = uuid.uuid4().hex
        with sqlite3.connect(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM collections WHERE status='running'").fetchone():
                raise HTTPException(409, "収集が実行中です")
            db.execute(
                "INSERT INTO collections(id,project_id,status,config) VALUES(?,?,'running',?)",
                (cid, project_id, body.model_dump_json()),
            )

        def collection_worker():
            try:
                gathered, errors = collect(body.model_dump())
                folder = root / "projects" / project_id / "images"
                folder.mkdir(parents=True, exist_ok=True)
                count = 0
                video_count = 0
                with sqlite3.connect(db_path) as db:
                    db.execute("BEGIN IMMEDIATE")
                    available = (
                        100
                        - db.execute(
                            "SELECT COUNT(*) FROM images WHERE project_id=?", (project_id,)
                        ).fetchone()[0]
                    )
                    for item in gathered[: max(0, available)]:
                        if item.get("video_data"):
                            try:
                                from studio.video_learning import normalise_video

                                if (
                                    db.execute(
                                        "SELECT COUNT(*) FROM media WHERE project_id=?",
                                        (project_id,),
                                    ).fetchone()[0]
                                    < 30
                                ):
                                    mid = uuid.uuid4().hex
                                    media_folder = root / "projects" / project_id / "media"
                                    media_folder.mkdir(parents=True, exist_ok=True)
                                    normalise_video(item["video_data"], media_folder / f"{mid}.mp4")
                                    db.execute(
                                        "INSERT INTO media(id,project_id,kind,caption,group_name) VALUES(?,?,'video',?,?)",
                                        (
                                            mid,
                                            project_id,
                                            f"a video of {project['trigger']}",
                                            body.group.strip() or "web",
                                        ),
                                    )
                                    video_count += 1
                            except Exception:
                                errors.append(
                                    "動画クリップは保存できませんでした。抽出画像のみ使用します。"
                                )
                        iid = uuid.uuid4().hex
                        item["image"].save(folder / f"{iid}.png", format="PNG")
                        db.execute(
                            "INSERT INTO images(id,project_id,caption,group_name) VALUES(?,?,?,?)",
                            (
                                iid,
                                project_id,
                                f"a photo of {project['trigger']}",
                                body.group.strip() or "web",
                            ),
                        )
                        db.execute(
                            "INSERT INTO image_sources VALUES(?,?,?,?)",
                            (iid, item["url"], item["page"], item["license"]),
                        )
                        count += 1
                    message = (
                        f"動画クリップ{video_count}件も動画学習データへ追加。 "
                        if video_count
                        else ""
                    ) + " / ".join(errors[:3])
                    if not count:
                        message = "対応する画像・動画が見つかりませんでした。URL・キーワードを変更してください。"
                    db.execute(
                        "UPDATE collections SET status=?, imported=?,error=? WHERE id=?",
                        ("completed" if count else "failed", count, message, cid),
                    )
            except Exception:
                query(
                    "UPDATE collections SET status='failed',error='サイトの取得に失敗しました。公開URL・接続・形式を確認してください' WHERE id=?",
                    (cid,),
                )

        if run_workers:
            threading.Thread(target=collection_worker, daemon=True).start()
        return {"id": cid, "status": "running"}

    @app.get("/api/projects/{project_id}/collections")
    def collections(project_id: str):
        project_or_404(project_id)
        return query(
            "SELECT * FROM collections WHERE project_id=? ORDER BY rowid DESC", (project_id,)
        )

    @app.get("/api/images/{image_id}/source")
    def image_source(image_id: str):
        return query("SELECT * FROM image_sources WHERE image_id=?", (image_id,), True) or {}

    class CaptionInput(BaseModel):
        caption: str = Field(min_length=1, max_length=1000)
        group_name: str = Field(default="default", min_length=1, max_length=40)

    @app.patch("/api/images/{image_id}")
    def caption(image_id: str, body: CaptionInput):
        row = query("SELECT * FROM images WHERE id=?", (image_id,), True)
        if not row:
            raise HTTPException(404, "画像が見つかりません")
        if not body.caption.strip():
            raise HTTPException(422, "説明を入力してください")
        query(
            "UPDATE images SET caption=?, group_name=?, labeled=? WHERE id=?",
            (
                body.caption.strip(),
                body.group_name.strip() or "default",
                1 if row["caption"] != body.caption.strip() else row["labeled"],
                image_id,
            ),
        )
        return {"ok": True}

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str):
        row = query("SELECT * FROM images WHERE id=?", (image_id,), True)
        if not row:
            raise HTTPException(404)
        return FileResponse(root / "projects" / row["project_id"] / "images" / f"{image_id}.png")

    def enqueue(project_id, kind, config):
        if not gpu_available():
            raise HTTPException(503, "GPU未接続です。CUDA GPUとrequirements-gpu.txtが必要です。")
        job_id = uuid.uuid4().hex
        with sqlite3.connect(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM jobs WHERE status IN ('queued','running')").fetchone():
                raise HTTPException(409, "別の処理が実行中です。完了してから実行してください。")
            db.execute(
                "INSERT INTO jobs(id,project_id,kind,status,config) VALUES(?,?,?,'queued',?)",
                (job_id, project_id, kind, json.dumps(config)),
            )
        if run_workers:
            threading.Thread(target=worker, daemon=True).start()
        return query("SELECT * FROM jobs WHERE id=?", (job_id,), True)

    @app.post("/api/projects/{project_id}/train", status_code=202)
    def train(project_id: str, body: TrainingInput):
        project_or_404(project_id)
        if body.target != "image":
            if body.mode not in {"supervised", "imitation", "style"}:
                raise HTTPException(
                    422, "音声・動画学習は、かんたん・写し・スタイル学習を選択してください"
                )
            rows = query(
                "SELECT * FROM media WHERE project_id=? AND kind=?", (project_id, body.target)
            )
            if body.group:
                rows = [row for row in rows if row["group_name"] == body.group]
            if len(rows) < 3:
                raise HTTPException(422, "音声・動画の学習ファイルを3件以上追加してください")
            return enqueue(
                project_id,
                body.target + "_train",
                {
                    **body.model_dump(),
                    "media": rows,
                    "instruction": body.instruction
                    or project_or_404(project_id)["training_instruction"],
                    "level": body.level or project_or_404(project_id)["level"],
                },
            )
        rows = images(project_id)
        if body.group:
            rows = [row for row in rows if row["group_name"] == body.group]
        if len(rows) < 3:
            raise HTTPException(422, "学習用の画像を3枚以上アップロードしてください")
        if body.mode == "reinforcement" and not any(row["rating"] for row in rows):
            raise HTTPException(
                422, "生成した作品を1〜5で評価してから、評価を使った学習を実行してください"
            )
        start = None
        base_model = os.getenv("STUDIO_CHECKPOINT", BASE_MODEL)
        if body.mode == "reinforcement":
            start = query(
                "SELECT id,config FROM jobs WHERE project_id=? AND kind='train' AND status='completed' ORDER BY rowid DESC LIMIT 1",
                (project_id,),
                True,
            )
            if not start:
                raise HTTPException(422, "先にモデルを学習して作品を生成・評価してください")
            base_model = json.loads(start["config"])["base_model"]
        return enqueue(
            project_id,
            "train",
            {
                **body.model_dump(),
                "instruction": body.instruction
                or project_or_404(project_id)["training_instruction"],
                "level": body.level or project_or_404(project_id)["level"],
                "rank": json.loads(start["config"]).get("rank", body.rank) if start else body.rank,
                "images": rows,
                "base_model": base_model,
                "start_job_id": start["id"] if start else None,
            },
        )

    @app.post("/api/projects/{project_id}/generate", status_code=202)
    def generate(project_id: str, body: GenerationInput):
        project_or_404(project_id)
        if body.with_audio and body.kind != "video":
            raise HTTPException(422, "音声は動画出力で利用できます")
        if body.with_audio and not query(
            "SELECT id FROM jobs WHERE project_id=? AND kind='audio_train' AND status='completed'",
            (project_id,),
            True,
        ):
            raise HTTPException(422, "先に音声モデルを学習してください")
        if not query(
            "SELECT id FROM jobs WHERE project_id=? AND kind='train' AND status='completed'",
            (project_id,),
            True,
        ):
            raise HTTPException(422, "先にモデルを学習してください")
        if body.kind == "video":
            source = query(
                "SELECT * FROM jobs WHERE id=? AND project_id=? AND kind='image' "
                "AND status='completed'",
                (body.source_id, project_id),
                True,
            )
            if not source:
                raise HTTPException(422, "動画の元になる生成済み画像を選んでください")
        project = project_or_404(project_id)
        config = body.model_dump()
        config["level"] = body.level or project["level"]
        preset = LEVELS[config["level"]]
        config["inference_steps"] = preset[f"{body.kind}_steps"]
        config["guidance_scale"] = preset[f"{body.kind}_guidance"]
        config["prompt"] = ", ".join(
            value for value in (body.prompt.strip(), project["rules_prompt"]) if value
        )
        config["negative_prompt"] = ", ".join(
            value for value in (body.negative_prompt.strip(), project["negative_rules"]) if value
        )
        config["user_prompt"] = body.prompt
        queued = enqueue(project_id, body.kind, config)
        query(
            "INSERT INTO chat_messages(id,project_id,channel,role,content) VALUES(?,?,'generation','user',?)",
            (uuid.uuid4().hex, project_id, body.prompt),
        )
        query(
            "INSERT INTO chat_messages(id,project_id,channel,role,content) VALUES(?,?,'generation','system',?)",
            (uuid.uuid4().hex, project_id, "生成を開始しました。ジョブID: " + queued["id"]),
        )
        return queued

    class FeedbackInput(BaseModel):
        rating: int = Field(ge=1, le=5)

    @app.post("/api/jobs/{job_id}/feedback")
    def feedback(job_id: str, body: FeedbackInput):
        job = query("SELECT * FROM jobs WHERE id=?", (job_id,), True)
        if not job or job["status"] != "completed" or job["kind"] == "video":
            raise HTTPException(422, "完了した生成画像か学習テスト画像を選んでください")
        filename = "test.png" if job["kind"] == "train" else "output.png"
        source = root / "jobs" / job_id / filename
        if not source.is_file():
            raise HTTPException(422, "評価する画像がありません")
        row = query("SELECT * FROM images WHERE origin_job_id=?", (job_id,), True)
        if row:
            query("UPDATE images SET rating=? WHERE id=?", (body.rating, row["id"]))
            return {"image_id": row["id"], "rating": body.rating}
        count = query(
            "SELECT COUNT(*) AS n FROM images WHERE project_id=?", (job["project_id"],), True
        )["n"]
        if count >= 100:
            raise HTTPException(422, "学習画像の上限100枚に達しています")
        image_id = uuid.uuid4().hex
        project = project_or_404(job["project_id"])
        config = json.loads(job["config"])
        caption = config.get("prompt", f"a photo of {project['trigger']}")
        destination = root / "projects" / project["id"] / "images" / f"{image_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.convert("RGB").save(destination)
        query(
            "INSERT INTO images(id,project_id,caption,labeled,rating,origin_job_id) VALUES(?,?,?,1,?,?)",
            (image_id, project["id"], caption, body.rating, job_id),
        )
        return {"image_id": image_id, "rating": body.rating}

    @app.get("/api/jobs/{job_id}/test-image")
    def test_image(job_id: str):
        job = query(
            "SELECT * FROM jobs WHERE id=? AND kind='train' AND status='completed'", (job_id,), True
        )
        if not job or not (root / "jobs" / job_id / "test.png").is_file():
            raise HTTPException(404, "学習テスト画像がありません")
        return FileResponse(root / "jobs" / job_id / "test.png")

    @app.get("/api/projects/{project_id}/metrics")
    def metrics(project_id: str):
        project_or_404(project_id)
        job = query(
            "SELECT id,config FROM jobs WHERE project_id=? AND kind IN ('train','video_train','audio_train') ORDER BY rowid DESC LIMIT 1",
            (project_id,),
            True,
        )
        if not job:
            return {"job_id": None, "parameters": {}, "points": []}
        config = json.loads(job["config"])
        return {
            "job_id": job["id"],
            "parameters": {
                key: config.get(key) for key in ("steps", "learning_rate", "rank", "mode", "level")
            },
            "points": query(
                "SELECT step,loss,learning_rate,gradient_norm,lora_norm FROM training_metrics WHERE job_id=? ORDER BY step",
                (job["id"],),
            ),
        }

    @app.get("/api/projects/{project_id}/jobs")
    def jobs(project_id: str):
        project_or_404(project_id)
        rows = query(
            "SELECT id,project_id,kind,status,progress,error,created_at FROM jobs "
            "WHERE project_id=? ORDER BY rowid DESC",
            (project_id,),
        )
        for row in rows:
            feedback_row = query(
                "SELECT rating FROM images WHERE origin_job_id=?", (row["id"],), True
            )
            row["feedback_rating"] = feedback_row["rating"] if feedback_row else None
            row["has_test_image"] = (
                row["kind"] == "train"
                and row["status"] == "completed"
                and (root / "jobs" / row["id"] / "test.png").is_file()
            )
        return rows

    @app.get("/api/jobs/{job_id}/result")
    def result(job_id: str, download: bool = False):
        job = query("SELECT * FROM jobs WHERE id=?", (job_id,), True)
        if not job or job["status"] != "completed":
            raise HTTPException(404, "出力はまだありません")
        names = {
            "train": "adapter.zip",
            "video_train": "adapter.zip",
            "audio_train": "adapter.zip",
            "image": "output.png",
            "video": "output.mp4",
        }
        name = names[job["kind"]]
        return FileResponse(root / "jobs" / job_id / name, filename=name if download else None)

    return app


app = create_app()
