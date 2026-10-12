"""Bounded local image understanding, public discovery, and real training augmentation."""

import hashlib
import json
import math
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path

from PIL import Image

from studio.collection import ACCESS_LIMIT_MESSAGE, CollectionAccessLimit, collect

CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
CAPTION_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
SIMILARITY_MODEL = "openai/clip-vit-base-patch32"
SIMILARITY_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def balance_related(rows):
    """Keep recent discoveries within the original count, even with discovery disabled."""
    originals = [row for row in rows if not row.get("related_job_id")]
    related = [row for row in rows if row.get("related_job_id")]
    return originals + (related[-len(originals) :] if originals else related)


class LocalVision:
    """Images stay local. Only the generated search words leave the PC."""

    def __init__(self):
        import torch
        from transformers import (
            BlipForConditionalGeneration,
            BlipProcessor,
            CLIPModel,
            CLIPProcessor,
        )

        self.torch = torch
        self.caption_processor = BlipProcessor.from_pretrained(
            CAPTION_MODEL, revision=CAPTION_REVISION
        )
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL, revision=CAPTION_REVISION, use_safetensors=False
        ).eval()
        self.processor = CLIPProcessor.from_pretrained(
            SIMILARITY_MODEL, revision=SIMILARITY_REVISION
        )
        self.model = CLIPModel.from_pretrained(
            SIMILARITY_MODEL, revision=SIMILARITY_REVISION, use_safetensors=False
        ).eval()
        self.styles = ["anime", "", "watercolor", "sketch", "painting", "illustration"]
        with torch.inference_mode():
            vectors = self.model.get_text_features(
                **self.processor(
                    text=[
                        "an anime illustration",
                        "a photograph of a real subject",
                        "a watercolor painting",
                        "a pencil sketch",
                        "an oil painting",
                        "a cartoon illustration",
                    ],
                    return_tensors="pt",
                    padding=True,
                )
            )
            self.style_vectors = (vectors / vectors.norm(dim=-1, keepdim=True)).tolist()

    def style(self, vector):
        scores = [sum(a * b for a, b in zip(vector, style)) for style in self.style_vectors]
        return self.styles[max(range(len(scores)), key=scores.__getitem__)]

    def caption(self, image):
        with self.torch.inference_mode():
            inputs = self.caption_processor(images=image, return_tensors="pt")
            output = self.caption_model.generate(**inputs, max_new_tokens=30, do_sample=False)
            return self.caption_processor.decode(output[0], skip_special_tokens=True)

    def embed(self, image):
        with self.torch.inference_mode():
            values = self.model.get_image_features(
                **self.processor(images=image, return_tensors="pt")
            )[0].float()
            return (values / values.norm().clamp_min(1e-8)).tolist()


def search_words(caption, style=""):
    stop = set(
        "a an the of on in at with and is are to for this that there it has its "
        "image picture photo photograph drawing painting illustration showing "
        "sitting standing looking wearing small large long short very next front background".split()
    )
    words = list(
        dict.fromkeys(w for w in re.findall(r"[a-z]{3,}", caption.lower()) if w not in stop)
    )
    # Commons searches all words; short queries avoid an over-specific generated sentence.
    return " ".join([style, *words[:1]] if style and words else words[:2])


def fingerprint(image):
    rgb = image.convert("RGB")
    exact = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    gray = list(rgb.resize((9, 8)).convert("L").getdata())
    difference = sum(
        (gray[y * 9 + x] > gray[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8)
    )
    color = rgb.resize((1, 1)).getpixel((0, 0))
    return exact, difference, color


def duplicate(mark, previous):
    return any(
        mark[0] == old[0]
        or (
            (mark[1] ^ old[1]).bit_count() <= 2
            and sum(abs(a - b) for a, b in zip(mark[2], old[2])) < 30
        )
        for old in previous
    )


def subjects(caption):
    """Coarse caption agreement complements visual similarity; it is not a quality score."""
    ignore = set(
        "a an the of on in at with and is are to for next close up view long short small large picture image photo photograph drawing painting illustration anime cartoon portrait art pixel white black blue red yellow green brown gray pink purple orange two three several group sitting standing lying looking wearing front background showing".split()
    )
    aliases = {
        "girl": "person",
        "woman": "person",
        "man": "person",
        "boy": "person",
        "people": "person",
        "kitten": "cat",
        "puppy": "dog",
    }
    words = [word for word in re.findall(r"[a-z]{3,}", caption.lower()) if word not in ignore]
    return {aliases.get(word.rstrip("s"), word.rstrip("s")) for word in words[:2]}


def frames_at(path, video):
    if video:
        from studio.video_learning import read_clip

        return [Image.fromarray(frame) for frame in read_clip(path, width=256, height=256, count=3)]
    with Image.open(path) as image:
        return [image.convert("RGB")]


def sequence_fingerprint(frames):
    marks = [fingerprint(frame) for frame in frames]
    return (
        hashlib.sha256("".join(mark[0] for mark in marks).encode()).hexdigest(),
        sum(mark[1] << (64 * index) for index, mark in enumerate(marks)),
        tuple(sum(mark[2][channel] for mark in marks) // len(marks) for channel in range(3)),
    )


def sequence_embedding(vision, frames):
    embeddings = [vision.embed(frame) for frame in frames]
    average = [sum(values) / len(embeddings) for values in zip(*embeddings)]
    norm = max(1e-8, math.sqrt(sum(value * value for value in average)))
    return [value / norm for value in average]


def candidate_frames(item, video):
    if not video:
        return [item["image"]]
    from studio.video_learning import normalise_video

    with tempfile.TemporaryDirectory(prefix="deel-related-video-") as temporary:
        path = Path(temporary) / "clip.mp4"
        normalise_video(item["video_data"], path)
        item["video_data"] = path.read_bytes()
        return frames_at(path, True)


def augment(root, project, config, folder, vision_factory=LocalVision, collector=collect):
    """Runs inside the cancellable training subprocess, before diffusion is loaded."""
    database = root / "studio.sqlite3"
    video = config.get("target") == "video"
    table, sources, relation, id_column = (
        ("media", "media_sources", "related_media", "media_id")
        if video
        else ("images", "image_sources", "related_images", "image_id")
    )
    data_key, suffix, capacity = ("media", ".mp4", 30) if video else ("images", ".png", 100)
    images_dir = root / "projects" / project["id"] / table
    report = {
        "stage": "analyzing",
        "kind": "video" if video else "image",
        "added": 0,
        "queries": [],
        "selected": [],
        "warnings": [],
    }
    report_path = folder / "related-report.json"

    def publish():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        temporary.replace(report_path)

    def active(db):
        status = db.execute("SELECT status FROM jobs WHERE id=?", (config["job_id"],)).fetchone()
        if not status or status[0] != "running":
            raise InterruptedError("学習を停止しました")

    originals = [row for row in config[data_key] if not row.get("related_job_id")]
    existing_related = [row for row in config[data_key] if row.get("related_job_id")]
    publish()
    if not originals:
        report["stage"] = "unavailable"
        report["warnings"].append(
            "元になる投稿データがありません。検索は行わず現在のデータで学習します。"
        )
        publish()
        return
    with sqlite3.connect(database) as db:
        active(db)
        all_ids = [
            r[0]
            for r in db.execute(
                f"SELECT id FROM {table} WHERE project_id=?"
                + (" AND kind='video'" if video else ""),
                (project["id"],),
            )
        ]
        known_pages = {
            r[0]
            for r in db.execute(
                f"SELECT page FROM {sources} JOIN {table} ON {table}.id={sources}.{id_column} WHERE {table}.project_id=?",
                (project["id"],),
            )
        }
        current_count = db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE project_id=?", (project["id"],)
        ).fetchone()[0]
    limit = min(config["related_limit"], len(originals), capacity - current_count)
    if limit <= 0:
        report["stage"] = "no_matches"
        report["warnings"].append(
            f"データは合計{capacity}件までです。検索は行わず現在のデータで学習します。"
        )
        publish()
        return
    marks = []
    for iid in all_ids:
        marks.append(sequence_fingerprint(frames_at(images_dir / (iid + suffix), video)))
    chosen = []
    try:
        vision = vision_factory()
        # Always anchor to the original uploads, never to recursively discovered images.
        anchors = originals[:3]
        vectors = []
        anchor_subjects = set()
        for row in anchors:
            frames = frames_at(images_dir / (row["id"] + suffix), video)
            vector = sequence_embedding(vision, frames)
            vectors.append(vector)
            caption = vision.caption(frames[len(frames) // 2])
            anchor_subjects.update(subjects(caption))
            query = search_words(caption, vision.style(vector))
            if query and query not in report["queries"]:
                report["queries"].append(query)
        report.update(
            stage="searching", anchors=[row["id"] for row in anchors], inspected=0, rejected=0
        )
        publish()
        for query in report["queries"][:3]:
            candidates, warnings = collector(
                {
                    "provider": "commons",
                    "query": query,
                    "media": "video" if video else "image",
                    "max_files": 10,
                    "thumbnail_width": None if video else 768,
                }
            )
            report["warnings"].extend(warnings[:1])
            for item in candidates:
                report["inspected"] += 1
                if (
                    not item.get("video_data" if video else "image")
                    or not item.get("license")
                    or item["page"] in known_pages
                ):
                    report["rejected"] += 1
                    continue
                try:
                    frames = candidate_frames(item, video)
                except (OSError, ValueError, RuntimeError):
                    report["rejected"] += 1
                    continue
                mark = sequence_fingerprint(frames)
                if duplicate(mark, marks):
                    report["rejected"] += 1
                    continue
                vector = sequence_embedding(vision, frames)
                score = max(sum(a * b for a, b in zip(vector, anchor)) for anchor in vectors)
                if score < config["related_threshold"]:
                    report["rejected"] += 1
                    continue
                caption = vision.caption(frames[len(frames) // 2])
                if anchor_subjects and not (subjects(caption) & anchor_subjects):
                    report["rejected"] += 1
                    continue
                chosen.append({**item, "similarity": round(score, 4), "description": caption})
                marks.append(mark)
                known_pages.add(item["page"])
            publish()
            if ACCESS_LIMIT_MESSAGE in warnings:
                break
        del vision
    except (OSError, RuntimeError, ValueError) as error:
        # Keep originals usable if a search server or model download is unavailable.
        report["warnings"].append(
            "関連データの解析・検索を完了できませんでした。取得済みの候補と元のデータで学習します。ネット接続・空き容量を確認してください。"
        )
        report["failure_type"] = type(error).__name__
        if isinstance(error, CollectionAccessLimit):
            report["warnings"].append(ACCESS_LIMIT_MESSAGE)
    chosen.sort(key=lambda item: item["similarity"], reverse=True)
    saved = []
    try:
        with sqlite3.connect(database) as db:
            db.execute("BEGIN IMMEDIATE")
            active(db)
            available = (
                capacity
                - db.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE project_id=?", (project["id"],)
                ).fetchone()[0]
            )
            for item in chosen[: max(0, min(limit, available))]:
                iid = uuid.uuid4().hex
                path = images_dir / (iid + suffix)
                saved.append(path)
                if video:
                    path.write_bytes(item["video_data"])
                else:
                    item["image"].save(path, format="PNG")
                row = {
                    "id": iid,
                    "caption": (item["description"] + ", " + project["trigger"])[:1000],
                    "group_name": config.get("group") or "related",
                    "labeled": 1,
                    "related_job_id": config["job_id"],
                    "similarity": item["similarity"],
                }
                db.execute(
                    "INSERT INTO media(id,project_id,caption,group_name,kind) VALUES(?,?,?,?,'video')"
                    if video
                    else "INSERT INTO images(id,project_id,caption,group_name,labeled) VALUES(?,?,?,?,1)",
                    (iid, project["id"], row["caption"], row["group_name"]),
                )
                db.execute(
                    f"INSERT INTO {sources} VALUES(?,?,?,?)",
                    (iid, item["url"], item["page"], item["license"]),
                )
                db.execute(
                    f"INSERT INTO {relation}({id_column},job_id,similarity) VALUES(?,?,?)",
                    (iid, config["job_id"], item["similarity"]),
                )
                existing_related.append(row)
                report["selected"].append(
                    {
                        "id": iid,
                        "page": item["page"],
                        "license": item["license"],
                        "similarity": item["similarity"],
                    }
                )
            # Related data never outweighs the original examples, including subsequent runs.
            existing_related.sort(
                key=lambda row: (
                    row.get("related_job_id") == config["job_id"],
                    row.get("similarity", 0),
                ),
                reverse=True,
            )
            config[data_key] = originals + existing_related[: len(originals)]
            report["added"] = len(saved)
            report["stage"] = "ready" if saved else "no_matches"
            if not saved and not report["warnings"]:
                report["warnings"].append(
                    "基準に合う新しい関連データが見つかりませんでした。元のデータで学習します。"
                )
            config["related_report"] = report
            db.execute(
                "UPDATE jobs SET config=? WHERE id=?", (json.dumps(config), config["job_id"])
            )
    except BaseException:
        for path in saved:
            path.unlink(missing_ok=True)
        raise
    publish()
