"""Versioned user-authored memory for a local gpt-oss agent; no weight fine-tuning."""

import json
import sqlite3
import uuid
from typing import Literal

from fastapi import HTTPException, Response
from pydantic import BaseModel, Field, model_validator


class Example(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    answer: str = Field(min_length=1, max_length=2000)


class AgentVersion(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    model: Literal["gpt-oss:20b", "gpt-oss:120b"] = "gpt-oss:20b"
    instruction: str = Field(default="", max_length=2000)
    knowledge: str = Field(default="", max_length=4000)
    examples: list[Example] = Field(default_factory=list, max_length=8)
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def memory_budget(self):
        if not self.name.strip():
            raise ValueError("育成版の名前を入力してください")
        size = len(self.instruction) + len(self.knowledge)
        size += sum(len(e.prompt) + len(e.answer) for e in self.examples)
        if size > 5000:
            raise ValueError("ノートPC向けの育成内容は、指示・知識・お手本の合計5000文字以内です")
        return self


class AgentRun(BaseModel):
    version_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    prompt: str = Field(min_length=1, max_length=1500)
    level: Literal["normal", "power", "excellent", "ultra"] = "normal"
    compare_before: bool = False


class AgentEvaluation(BaseModel):
    rating: int = Field(ge=1, le=5)
    correction: str = Field(default="", max_length=2000)


def register_agent_routes(app, db_path, query, project_or_404, model_guard):
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_versions (
                id TEXT PRIMARY KEY, project_id TEXT, config TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY, project_id TEXT, version_id TEXT, prompt TEXT,
                answer TEXT, before_answer TEXT, rating INTEGER, correction TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """)

    def version_or_404(project_id, version_id):
        row = query(
            "SELECT * FROM agent_versions WHERE id=? AND project_id=?",
            (version_id, project_id),
            True,
        )
        if not row:
            raise HTTPException(404, "このプロジェクトの育成版を選択してください")
        return {**row, "config": json.loads(row["config"])}

    @app.get("/api/projects/{project_id}/agent")
    def agent_state(project_id: str):
        project_or_404(project_id)
        versions = query(
            "SELECT * FROM agent_versions WHERE project_id=? ORDER BY rowid DESC", (project_id,)
        )
        return {
            "versions": [{**row, "config": json.loads(row["config"])} for row in versions],
            "runs": query(
                "SELECT * FROM agent_runs WHERE project_id=? ORDER BY rowid DESC LIMIT 50",
                (project_id,),
            ),
        }

    @app.post("/api/projects/{project_id}/agent/versions", status_code=201)
    def save_version(project_id: str, body: AgentVersion):
        project_or_404(project_id)
        if body.parent_id:
            parent = version_or_404(project_id, body.parent_id)
            if parent["config"]["model"] != body.model:
                raise HTTPException(422, "続きから育てるときは同じ基盤モデルを選択してください")
        version_id = uuid.uuid4().hex
        query(
            "INSERT INTO agent_versions(id,project_id,config) VALUES(?,?,?)",
            (version_id, project_id, json.dumps(body.model_dump(), ensure_ascii=False)),
        )
        return version_or_404(project_id, version_id)

    @app.get("/api/projects/{project_id}/agent/versions/{version_id}/export")
    def export_version(project_id: str, version_id: str):
        project_or_404(project_id)
        version = version_or_404(project_id, version_id)
        return Response(
            json.dumps({"format": "deel-agent-memory-v1", **version}, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="deel-agent-profile.json"'},
        )

    @app.post("/api/projects/{project_id}/agent/run")
    def run_agent(project_id: str, body: AgentRun):
        project_or_404(project_id)
        if not body.prompt.strip():
            raise HTTPException(422, "エージェントへの依頼を入力してください")
        version = version_or_404(project_id, body.version_id)["config"]
        from studio.assistant import ask

        memory = version["instruction"]
        if version["knowledge"]:
            memory += "\n\nユーザーが保存した知識：\n" + version["knowledge"]
        # Never execute generated commands. The agent produces advice/plans for the user.
        if not model_guard.acquire(blocking=False):
            raise HTTPException(409, "エージェントが応答中です。完了後に実行してください。")
        try:
            if query("SELECT id FROM jobs WHERE status IN ('queued','running')", one=True):
                raise HTTPException(
                    409, "学習・生成の完了後に実行してください。メモリを順番に使います。"
                )
            before = ask(version["model"], body.prompt, body.level) if body.compare_before else None
            answer = ask(
                version["model"],
                body.prompt,
                body.level,
                memory=memory.strip(),
                examples=version["examples"],
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                503,
                (
                    "Ollamaを起動し、ollama pull "
                    + version["model"]
                    + " を実行してください。育成データは保存されています。"
                )
                if isinstance(exc, OSError)
                else str(exc),
            ) from exc
        finally:
            model_guard.release()
        run_id = uuid.uuid4().hex
        query(
            "INSERT INTO agent_runs(id,project_id,version_id,prompt,answer,before_answer) VALUES(?,?,?,?,?,?)",
            (run_id, project_id, body.version_id, body.prompt, answer, before),
        )
        return query("SELECT * FROM agent_runs WHERE id=?", (run_id,), True)

    @app.patch("/api/projects/{project_id}/agent/runs/{run_id}")
    def evaluate_agent(project_id: str, run_id: str, body: AgentEvaluation):
        project_or_404(project_id)
        if not query(
            "SELECT id FROM agent_runs WHERE id=? AND project_id=?", (run_id, project_id), True
        ):
            raise HTTPException(404, "評価する応答が見つかりません")
        query(
            "UPDATE agent_runs SET rating=?,correction=? WHERE id=? AND project_id=?",
            (body.rating, body.correction, run_id, project_id),
        )
        return {"saved": True}

    @app.post("/api/projects/{project_id}/agent/runs/{run_id}/learn", status_code=201)
    def learn_evaluation(project_id: str, run_id: str):
        project_or_404(project_id)
        run = query(
            "SELECT * FROM agent_runs WHERE id=? AND project_id=?", (run_id, project_id), True
        )
        if not run or run["rating"] is None:
            raise HTTPException(422, "先に応答の評価を保存してください")
        answer = run["correction"].strip() or (run["answer"] if run["rating"] >= 4 else "")
        if not answer:
            raise HTTPException(422, "低評価の応答には、望ましい回答を入力してください")
        config = version_or_404(project_id, run["version_id"])["config"]
        config["parent_id"] = run["version_id"]
        config["name"] = config["name"][:60] + " / 評価を反映"
        config["examples"] = [*config["examples"], {"prompt": run["prompt"], "answer": answer}]
        try:
            body = AgentVersion(**config)
        except ValueError as exc:
            raise HTTPException(
                422, "育成内容の上限です。お手本を整理して保存してください"
            ) from exc
        return save_version(project_id, body)
