import csv
import html
import json
import math
import os
import re
import threading
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

import comfy.sd
import comfy.utils
import folder_paths

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

from .analyzer_installer import (
    analyzer_ready_details,
    ensure_analyzer_async,
    read_status as read_analyzer_status,
)

# First-run installation is automatic and non-blocking.
ensure_analyzer_async(force=False)
try:
    (Path(folder_paths.get_input_directory()) / "lora_reference").mkdir(parents=True, exist_ok=True)
except Exception:
    pass


PLUGIN_VERSION = "7.1.0"
_GRID_LOCK = threading.Lock()
_FACE_APP = None
_FACE_APP_ERROR = None


def release_analyzer_resources():
    """Drop persistent GPU-backed face sessions and clear allocator caches."""
    global _FACE_APP
    _FACE_APP = None
    try:
        from .cvlface_analyzer import release_cuda
        release_cuda()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _available_loras():
    try:
        return sorted(folder_paths.get_filename_list("loras"), key=lambda x: x.lower())
    except Exception:
        return []


def _safe_slug(value, fallback="item"):
    value = str(value or "").strip()
    value = value.replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value[:180] or fallback


def _safe_child(base, relative):
    base = Path(base).resolve()
    target = (base / str(relative or "")).resolve()
    if target != base and base not in target.parents:
        raise ValueError("Unsafe path.")
    return target


def _short_lora_label(filename):
    name = Path(str(filename)).stem
    step = re.search(r"(?:save-|step[_-]?)(\d{3,6})", name, flags=re.IGNORECASE)
    if step:
        return f"step {step.group(1)}"
    if len(name) > 50:
        return name[:47] + "..."
    return name


def _tensor_to_pil(images):
    if not isinstance(images, torch.Tensor):
        raise TypeError("Expected a ComfyUI IMAGE tensor.")
    image = images[0].detach().cpu().float().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return Image.fromarray(image, mode="L").convert("RGB")
    if image.shape[-1] == 4:
        return Image.fromarray(image, mode="RGBA").convert("RGB")
    return Image.fromarray(image, mode="RGB")


def _font(size, bold=False):
    candidates = []
    if os.name == "nt":
        win = os.environ.get("WINDIR", r"C:\Windows")
        candidates.extend([
            os.path.join(win, "Fonts", "arialbd.ttf" if bold else "arial.ttf"),
            os.path.join(win, "Fonts", "segoeuib.ttf" if bold else "segoeui.ttf"),
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_pixels(draw, text, font, max_width, max_lines=3):
    words = str(text or "").split()
    if not words:
        return [""]
    lines = []
    current = ""
    index = 0
    while index < len(words):
        word = words[index]
        proposed = word if not current else f"{current} {word}"
        box = draw.textbbox((0, 0), proposed, font=font)
        if box[2] - box[0] <= max_width:
            current = proposed
            index += 1
        else:
            if current:
                lines.append(current)
                current = ""
            else:
                lines.append(word)
                index += 1
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if index < len(words) and lines:
        while lines[-1] and draw.textbbox((0, 0), lines[-1] + "...", font=font)[2] > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines[:max_lines]


def _make_labeled_cell(image_path, label, cell_width, image_height, label_height, font_size):
    source = Image.open(image_path).convert("RGB")
    contained = ImageOps.contain(
        source,
        (cell_width, image_height),
        method=Image.Resampling.LANCZOS,
    )
    cell = Image.new("RGB", (cell_width, image_height + label_height), (18, 18, 18))
    x = (cell_width - contained.width) // 2
    y = (image_height - contained.height) // 2
    cell.paste(contained, (x, y))

    draw = ImageDraw.Draw(cell)
    draw.rectangle(
        (0, image_height, cell_width, image_height + label_height),
        fill=(26, 26, 29),
    )
    font = _font(font_size, bold=True)
    lines = _wrap_pixels(draw, _short_lora_label(label), font, cell_width - 20, max_lines=2)
    line_height = max(12, int(font_size * 1.18))
    total = line_height * len(lines)
    text_y = image_height + max(4, (label_height - total) // 2)
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        text_x = (cell_width - (box[2] - box[0])) // 2
        draw.text((text_x, text_y), line, font=font, fill=(245, 245, 245))
        text_y += line_height
    return cell


def _scaled_image_height(image_paths, cell_width):
    ratios = []
    for path in image_paths:
        try:
            with Image.open(path) as im:
                ratios.append(im.height / max(1, im.width))
        except Exception:
            pass
    ratio = max(ratios) if ratios else 1.0
    return max(160, int(round(cell_width * ratio)))


def _ui_image_entry(path, output_root):
    relative = os.path.relpath(path, output_root)
    return {
        "filename": os.path.basename(relative),
        "subfolder": os.path.dirname(relative).replace("\\", "/"),
        "type": "output",
    }


def _quality_metrics(image_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    lum = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )

    dx = np.diff(lum, axis=1)
    dy = np.diff(lum, axis=0)
    sharpness = float((np.mean(dx * dx) + np.mean(dy * dy)) * 10000.0)
    contrast = float(np.std(lum) * 100.0)
    clipping = float(np.mean((lum < 0.015) | (lum > 0.985)) * 100.0)

    hist, _ = np.histogram(lum, bins=128, range=(0.0, 1.0), density=False)
    probabilities = hist.astype(np.float64)
    probabilities /= max(1.0, probabilities.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "clipping": clipping,
        "entropy": entropy,
    }


def _minmax(values, value, inverse=False):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return 0.5
    low, high = min(finite), max(finite)
    if abs(high - low) < 1e-9:
        normalized = 0.5
    else:
        normalized = (float(value) - low) / (high - low)
    return 1.0 - normalized if inverse else normalized


def _largest_face(faces):
    if not faces:
        return None
    def area(face):
        bbox = np.asarray(face.bbox, dtype=np.float32)
        return max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    return max(faces, key=area)



def _get_face_app():
    global _FACE_APP, _FACE_APP_ERROR
    if _FACE_APP is not None:
        return _FACE_APP
    if _FACE_APP_ERROR is not None:
        raise RuntimeError(_FACE_APP_ERROR)

    _ready, details = analyzer_ready_details()
    if not details.get("antelope_ready", _ready):
        ensure_analyzer_async(force=False)
        raise RuntimeError(
            "The AntelopeV2 analyzer is still installing or is unavailable. "
            f"Current state: {read_analyzer_status()}"
        )

    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")

        root = Path(folder_paths.models_dir) / "insightface"
        root.mkdir(parents=True, exist_ok=True)
        app = FaceAnalysis(name="antelopev2", root=str(root), providers=providers)
        app.prepare(
            ctx_id=0 if "CUDAExecutionProvider" in available else -1,
            det_thresh=0.30,
            det_size=(640, 640),
        )
        _FACE_APP = app
        return _FACE_APP
    except Exception as exc:
        _FACE_APP_ERROR = (
            "AntelopeV2 identity analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )
        raise RuntimeError(_FACE_APP_ERROR)


def _face_embedding_record(image_path, face_app):
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    face = _largest_face(face_app.get(image))
    if face is None:
        # Tight portraits can place the forehead or chin on the frame edge.  SCRFD
        # may reject those at its first pass even when the face is unambiguous.
        # Add neutral context and retry instead of incorrectly scoring identity 0.
        height, width = image.shape[:2]
        pad_y = max(32, int(round(height * 0.18)))
        pad_x = max(32, int(round(width * 0.18)))
        border_pixels = np.concatenate((
            image[: max(1, height // 20), :, :].reshape(-1, 3),
            image[-max(1, height // 20):, :, :].reshape(-1, 3),
            image[:, : max(1, width // 20), :].reshape(-1, 3),
            image[:, -max(1, width // 20):, :].reshape(-1, 3),
        ), axis=0)
        border_color = tuple(int(value) for value in np.median(border_pixels, axis=0))
        padded = cv2.copyMakeBorder(
            image,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )
        face = _largest_face(face_app.get(padded))
    if face is None:
        return None
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    bbox = np.asarray(face.bbox, dtype=np.float32)
    face_side = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    detector_score = float(getattr(face, "det_score", 0.0) or 0.0)
    size_score = float(np.clip(face_side / 180.0, 0.0, 1.0))
    quality = float(np.clip(0.75 * detector_score + 0.25 * size_score, 0.0, 1.0))
    return {
        "embedding": embedding / norm,
        "quality": quality,
        "detector_score": detector_score,
        "face_size": face_side,
    }


def _face_embedding(image_path, face_app):
    record = _face_embedding_record(image_path, face_app)
    return record["embedding"] if record is not None else None


def _robust_reference_template(reference_folder):
    input_root = Path(folder_paths.get_input_directory())
    ref_root = _safe_child(input_root, reference_folder)
    if not ref_root.exists():
        raise FileNotFoundError(f"Reference folder does not exist: {ref_root}")

    image_paths = [
        path for path in ref_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    if not image_paths:
        raise FileNotFoundError(f"No reference images were found in: {ref_root}")

    face_app = _get_face_app()
    records = []
    for path in image_paths:
        try:
            record = _face_embedding_record(path, face_app)
            if record is not None:
                records.append((path, record))
        except Exception:
            continue
    if len(records) < 2:
        raise RuntimeError("At least two detectable reference faces are required; 5–20 varied images are recommended.")

    embeddings = np.stack([record[1]["embedding"] for record in records], axis=0)
    similarity = embeddings @ embeddings.T
    # Robustly reject references that disagree with the main identity cluster.
    cohesion = np.median(similarity, axis=1)
    median = float(np.median(cohesion))
    mad = float(np.median(np.abs(cohesion - median)))
    threshold = max(0.15, median - 2.5 * max(mad, 0.01))
    keep = cohesion >= threshold
    if int(np.sum(keep)) < 2:
        keep = np.ones(len(records), dtype=bool)

    inliers = embeddings[keep]
    quality = np.asarray([record[1]["quality"] for record in records], dtype=np.float64)[keep]
    q_low, q_high = float(np.quantile(quality, 0.10)), float(np.quantile(quality, 0.90))
    normalized_quality = np.clip((quality - q_low) / max(1e-6, q_high - q_low), 0.0, 1.0)
    weights = 0.25 + 0.75 * normalized_quality
    centroid = np.average(inliers, axis=0, weights=weights)
    centroid /= max(1e-12, float(np.linalg.norm(centroid)))
    return {
        "centroid": centroid,
        "reference_embeddings": inliers,
        "reference_weights": weights,
        "image_paths": [str(path) for path in image_paths],
        "face_app": face_app,
        "detected": len(records),
        "total": len(image_paths),
        "inliers": int(np.sum(keep)),
        "rejected": int(len(records) - np.sum(keep)),
        "cohesion": float(np.average(inliers @ centroid, weights=weights)),
        "mean_quality": float(np.mean(quality)),
    }


def _identity_similarity(embedding, template):
    if embedding is None:
        return None
    centroid_score = float(np.dot(embedding, template["centroid"]))
    reference_scores = template["reference_embeddings"] @ embedding
    top_k = min(5, len(reference_scores))
    top_scores = np.sort(reference_scores)[-top_k:]
    if len(top_scores) >= 4:
        top_scores = top_scores[:-1]  # remove one pose-copy maximum
    top_mean = float(np.mean(top_scores))
    # Quality-weighted centroid is stable; trimmed references retain pose variants.
    return 0.70 * centroid_score + 0.30 * top_mean


def _analysis_report_html(run_id, analysis):
    esc = html.escape
    rows = []
    for item in analysis["ranking"]:
        identity = (
            f"{item['identity_score']:.2f}"
            if item.get("identity_score") is not None
            else "n/a"
        )
        rows.append(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{esc(item['lora_label'])}</td>"
            f"<td>{item['combined_score']:.2f}</td>"
            f"<td>{identity}</td>"
            f"<td>{item.get('detection_rate', 0.0):.1f}%</td>"
            f"<td>{item['consistency_score']:.2f}</td>"
            "</tr>"
        )

    prompt_sections = []
    prompt_count = int(analysis["prompt_count"])
    lora_count = int(analysis["lora_count"])
    for p in range(prompt_count):
        cells = []
        for l in range(lora_count):
            entry = next(
                x for x in analysis["entries"]
                if int(x["prompt_index"]) == p and int(x["lora_index"]) == l
            )
            asset = (
                f"/lorapromptqueue/v4/asset?"
                f"run_id={esc(run_id)}&path=_cells/p{p:03d}_l{l:03d}.png"
            )
            score = entry.get("identity_similarity")
            score_text = f"identity {score:.3f}" if score is not None else "identity n/a"
            cells.append(
                "<figure>"
                f"<img src=\"{asset}\" loading=\"lazy\">"
                f"<figcaption>{esc(_short_lora_label(entry['lora_label']))}<br>"
                f"{score_text} · quality {entry['quality_score']:.1f}</figcaption>"
                "</figure>"
            )
        prompt_text = next(
            x["prompt_text"] for x in analysis["entries"]
            if int(x["prompt_index"]) == p
        )
        prompt_sections.append(
            f"<section><h2>Prompt {p + 1}</h2>"
            f"<p>{esc(prompt_text)}</p>"
            f"<div class=\"grid\">{''.join(cells)}</div></section>"
        )

    warning = ""
    if analysis.get("analysis_warning"):
        warning = f"<div class=\"warning\">{esc(analysis['analysis_warning'])}</div>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>LoRA Test Analysis — {esc(run_id)}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#111318;color:#eef1f6;margin:0;padding:28px}}
h1,h2{{margin:0 0 12px}}
p{{line-height:1.45;color:#cbd1db}}
table{{border-collapse:collapse;width:100%;max-width:1100px;background:#1a1d24;margin:20px 0 32px}}
th,td{{border:1px solid #343a46;padding:9px;text-align:left}}
th{{background:#252a34}}
tr:first-child td{{font-weight:700}}
.warning{{background:#4b3714;border:1px solid #8a682d;padding:12px;border-radius:8px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}
figure{{margin:0;background:#1b1e25;padding:8px;border-radius:8px}}
img{{width:100%;height:auto;display:block;border-radius:5px}}
figcaption{{padding-top:7px;color:#dfe4ec;font-size:13px}}
section{{margin-top:38px}}
a{{color:#8dc5ff}}
</style>
</head>
<body>
<h1>Controlled LoRA comparison</h1>
<p><strong>Run:</strong> {esc(run_id)}<br>
<strong>Winner:</strong> {esc(analysis['winner'])}<br>
<strong>Confidence:</strong> {esc(analysis['confidence'])}</p>
{warning}
<h2>Aggregate ranking</h2>
<table>
<thead><tr><th>Rank</th><th>LoRA</th><th>Combined</th><th>Identity</th><th>Detected</th><th>Consistency</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
{''.join(prompt_sections)}
</body>
</html>"""


def _render_ranking_png(run_root, analysis):
    width = 1180
    row_height = 54
    header_height = 100
    height = header_height + row_height * len(analysis["ranking"]) + 30
    image = Image.new("RGB", (width, height), (18, 19, 23))
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    body_font = _font(20, bold=False)
    bold_font = _font(20, bold=True)
    small_font = _font(16, bold=False)

    draw.text((24, 18), "Controlled LoRA automatic ranking", font=title_font, fill=(248, 248, 250))
    draw.text(
        (24, 60),
        f"Winner: {analysis['winner']}  |  confidence: {analysis['confidence']}",
        font=small_font,
        fill=(195, 202, 214),
    )

    columns = [24, 90, 640, 790, 925, 1060]
    labels = ["#", "LoRA", "Combined", "Identity", "Detected", "Consistency"]
    y = header_height
    draw.rectangle((16, y, width - 16, y + row_height), fill=(42, 46, 56))
    for x, label in zip(columns, labels):
        draw.text((x, y + 14), label, font=bold_font, fill=(255, 220, 110))
    y += row_height

    for index, item in enumerate(analysis["ranking"]):
        fill = (31, 34, 42) if index % 2 == 0 else (25, 28, 35)
        draw.rectangle((16, y, width - 16, y + row_height), fill=fill)
        identity = f"{item['identity_score']:.2f}" if item.get("identity_score") is not None else "n/a"
        values = [
            str(item["rank"]),
            _short_lora_label(item["lora_label"]),
            f"{item['combined_score']:.2f}",
            identity,
            f"{item.get('detection_rate', 0.0):.1f}%",
            f"{item['consistency_score']:.2f}",
        ]
        for x, value in zip(columns, values):
            draw.text((x, y + 14), value, font=body_font, fill=(238, 240, 245))
        y += row_height

    output = run_root / "AUTO_RANKING.png"
    image.save(output, format="PNG")
    return output



def _analyze_run(run_id, reference_folder, analysis_mode):
    output_root = Path(folder_paths.get_output_directory())
    safe_run = _safe_slug(run_id, "run")
    run_root = _safe_child(output_root / "LoRA_Test_Grids", safe_run)
    if not run_root.exists():
        raise FileNotFoundError(f"Run not found: {run_root}")

    manifests = sorted(run_root.glob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError("No completed manifest was found for this run.")

    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [dict(item) for item in manifest["entries"]]
    prompt_count = int(manifest["prompt_count"])
    lora_count = int(manifest["lora_count"])

    template = _robust_reference_template(reference_folder)
    for entry in entries:
        p = int(entry["prompt_index"])
        l = int(entry["lora_index"])
        entry["cell_path"] = str(run_root / "_cells" / f"p{p:03d}_l{l:03d}.png")
        entry["metrics"] = _quality_metrics(entry["cell_path"])
        embedding = _face_embedding(entry["cell_path"], template["face_app"])
        entry["face_detected"] = embedding is not None
        entry["identity_similarity"] = _identity_similarity(embedding, template)

    # A small objective-quality tie-breaker, normalized only within each prompt.
    for p in range(prompt_count):
        group = [x for x in entries if int(x["prompt_index"]) == p]
        sharp = [x["metrics"]["sharpness"] for x in group]
        clipping = [x["metrics"]["clipping"] for x in group]
        for item in group:
            item["quality_score"] = float((
                0.70 * _minmax(sharp, item["metrics"]["sharpness"])
                + 0.30 * _minmax(clipping, item["metrics"]["clipping"], inverse=True)
            ) * 100.0)

    lora_labels = [
        next(item["lora_label"] for item in entries if int(item["lora_index"]) == l)
        for l in range(lora_count)
    ]

    ranking = []
    for l, label in enumerate(lora_labels):
        group = [x for x in entries if int(x["lora_index"]) == l]
        identities = [float(x["identity_similarity"]) for x in group if x["identity_similarity"] is not None]
        detection_rate = len(identities) / max(1, prompt_count)
        quality_score = float(np.mean([x["quality_score"] for x in group]))
        if identities:
            mean_similarity = float(np.mean(identities))
            median_similarity = float(np.median(identities))
            worst_quintile = float(np.quantile(identities, 0.20))
            identity_std = float(np.std(identities))
            consistency = max(0.0, 1.0 - identity_std * 3.5)
            # Identity dominates. Worst-case performance and detection prevent one easy portrait from winning.
            raw = (
                0.58 * mean_similarity
                + 0.22 * worst_quintile
                + 0.10 * median_similarity
                + 0.07 * detection_rate
                + 0.03 * consistency
            )
            combined = raw * 95.0 + quality_score * 0.05
        else:
            mean_similarity = median_similarity = worst_quintile = None
            consistency = 0.0
            combined = -100.0

        ranking.append({
            "lora_index": l,
            "lora_label": label,
            "identity_score": None if mean_similarity is None else mean_similarity * 100.0,
            "median_identity_score": None if median_similarity is None else median_similarity * 100.0,
            "worst_quintile_score": None if worst_quintile is None else worst_quintile * 100.0,
            "detection_rate": detection_rate * 100.0,
            "quality_score": quality_score,
            "consistency_score": consistency * 100.0,
            "combined_score": float(combined),
        })

    ranking.sort(key=lambda item: item["combined_score"], reverse=True)
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    winner = ranking[0]["lora_label"] if ranking else "n/a"
    gap = (ranking[0]["combined_score"] - ranking[1]["combined_score"]) if len(ranking) > 1 else 0.0
    if prompt_count >= 8 and gap >= 2.0 and ranking[0]["detection_rate"] >= 85:
        confidence = "high"
    elif prompt_count >= 5 and gap >= 0.7:
        confidence = "medium"
    else:
        confidence = "low — inspect the grid"

    analysis_warning = (
        f"AntelopeV2 used {template['inliers']} inlier reference faces out of "
        f"{template['detected']} detected / {template['total']} files; "
        f"{template['rejected']} reference outliers rejected. "
        f"Reference cohesion: {template['cohesion']:.3f}."
    )

    analysis = {
        "plugin_version": PLUGIN_VERSION,
        "analyzer": "InsightFace AntelopeV2 (SCRFD-10GF + GlintR100)",
        "run_id": run_id,
        "winner": winner,
        "confidence": confidence,
        "analysis_warning": analysis_warning,
        "face_ai_used": True,
        "reference_folder": reference_folder,
        "prompt_count": prompt_count,
        "lora_count": lora_count,
        "ranking": ranking,
        "entries": entries,
    }

    (run_root / "AUTO_ANALYSIS.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (run_root / "AUTO_RANKING.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "rank", "lora", "combined_score", "mean_identity", "median_identity",
            "worst_quintile", "face_detection_rate", "consistency", "quality_tiebreaker"
        ])
        for item in ranking:
            writer.writerow([
                item["rank"], item["lora_label"], f"{item['combined_score']:.6f}",
                "" if item["identity_score"] is None else f"{item['identity_score']:.6f}",
                "" if item["median_identity_score"] is None else f"{item['median_identity_score']:.6f}",
                "" if item["worst_quintile_score"] is None else f"{item['worst_quintile_score']:.6f}",
                f"{item['detection_rate']:.6f}", f"{item['consistency_score']:.6f}",
                f"{item['quality_score']:.6f}",
            ])

    (run_root / "AUTO_REPORT.html").write_text(
        _analysis_report_html(run_id, analysis), encoding="utf-8"
    )
    ranking_png = _render_ranking_png(run_root, analysis)
    review_zip = run_root / "REVIEW_PACKAGE.zip"
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_root.rglob("*"):
            if path == review_zip or not path.is_file():
                continue
            archive.write(path, path.relative_to(run_root))

    return {
        "run_id": run_id,
        "winner": winner,
        "confidence": confidence,
        "warning": analysis_warning,
        "ranking": ranking,
        "report_url": f"/lorapromptqueue/v4/report?run_id={run_id}",
        "run_folder": str(run_root),
        "ranking_png": str(ranking_png),
        "review_zip": str(review_zip),
    }


class LoRATestIdentityLoader:
    def __init__(self):
        self._cached_path = None
        self._cached_mtime = None
        self._cached_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = _available_loras() or ["None"]
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "automation/LoRA testing"

    def load_lora(self, model, lora_name, strength_model):
        if lora_name in ("", "None") or float(strength_model) == 0.0:
            return (model,)

        get_path = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(get_path):
            lora_path = get_path("loras", lora_name)
        else:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA not found: {lora_name}")

        mtime = os.path.getmtime(lora_path)
        if (
            self._cached_lora is None
            or self._cached_path != lora_path
            or self._cached_mtime != mtime
        ):
            self._cached_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self._cached_path = lora_path
            self._cached_mtime = mtime

        model_lora, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            self._cached_lora,
            float(strength_model),
            0.0,
        )
        return (model_lora,)


class LoRATestGridCollector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "run_id": ("STRING", {"default": "manual_run"}),
                "prompt_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "lora_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "prompt_count": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "lora_count": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "prompt_label": ("STRING", {"default": "Prompt 1"}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "lora_label": ("STRING", {"default": "LoRA"}),
                "output_prefix": ("STRING", {"default": "LoRA_Test"}),
                "grid_mode": (
                    [
                        "per_prompt_and_master",
                        "master_only",
                        "per_prompt_only",
                        "off",
                    ],
                ),
                "cell_width": (
                    "INT",
                    {"default": 384, "min": 160, "max": 1024, "step": 16},
                ),
                "label_height": (
                    "INT",
                    {"default": 64, "min": 32, "max": 180, "step": 4},
                ),
                "font_size": (
                    "INT",
                    {"default": 22, "min": 10, "max": 64, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "collect"
    CATEGORY = "automation/LoRA testing"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def collect(
        self,
        images,
        run_id,
        prompt_index,
        lora_index,
        prompt_count,
        lora_count,
        prompt_label,
        prompt_text,
        lora_label,
        output_prefix,
        grid_mode,
        cell_width,
        label_height,
        font_size,
    ):
        output_root = Path(folder_paths.get_output_directory())
        safe_run = _safe_slug(run_id, "run")
        safe_prefix = _safe_slug(output_prefix, "LoRA_Test")
        run_root = output_root / "LoRA_Test_Grids" / safe_run
        cells_root = run_root / "_cells"
        cells_root.mkdir(parents=True, exist_ok=True)

        prompt_index = int(prompt_index)
        lora_index = int(lora_index)
        prompt_count = int(prompt_count)
        lora_count = int(lora_count)

        cell_path = cells_root / f"p{prompt_index:03d}_l{lora_index:03d}.png"
        meta_path = cells_root / f"p{prompt_index:03d}_l{lora_index:03d}.json"
        pil_image = _tensor_to_pil(images)

        with _GRID_LOCK:
            pil_image.save(cell_path, format="PNG")
            metadata = {
                "run_id": run_id,
                "prompt_index": prompt_index,
                "lora_index": lora_index,
                "prompt_count": prompt_count,
                "lora_count": lora_count,
                "prompt_label": prompt_label,
                "prompt_text": prompt_text,
                "lora_label": lora_label,
                "output_prefix": output_prefix,
            }
            meta_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            completed_ui = []
            row_paths = [
                cells_root / f"p{prompt_index:03d}_l{i:03d}.png"
                for i in range(lora_count)
            ]
            row_meta = [
                cells_root / f"p{prompt_index:03d}_l{i:03d}.json"
                for i in range(lora_count)
            ]
            row_complete = all(p.exists() for p in row_paths) and all(p.exists() for p in row_meta)

            if row_complete and grid_mode in ("per_prompt_and_master", "per_prompt_only"):
                metas = [json.loads(p.read_text(encoding="utf-8")) for p in row_meta]
                image_height = _scaled_image_height(row_paths, int(cell_width))
                title_height = max(58, int(font_size) * 3)
                grid = Image.new(
                    "RGB",
                    (
                        int(cell_width) * lora_count,
                        title_height + image_height + int(label_height),
                    ),
                    (15, 15, 17),
                )
                draw = ImageDraw.Draw(grid)
                title_font = _font(max(15, int(font_size) + 2), bold=True)
                title = metas[0].get("prompt_label") or f"Prompt {prompt_index + 1}"
                lines = _wrap_pixels(draw, title, title_font, grid.width - 30, max_lines=2)
                y = 8
                for line in lines:
                    draw.text((15, y), line, font=title_font, fill=(245, 245, 245))
                    y += int((int(font_size) + 2) * 1.2)

                for column, (path, meta) in enumerate(zip(row_paths, metas)):
                    cell = _make_labeled_cell(
                        path,
                        meta["lora_label"],
                        int(cell_width),
                        image_height,
                        int(label_height),
                        int(font_size),
                    )
                    grid.paste(cell, (column * int(cell_width), title_height))

                row_output = run_root / f"{safe_prefix}_P{prompt_index + 1:02d}_GRID.png"
                grid.save(row_output, format="PNG")
                completed_ui.append(_ui_image_entry(row_output, output_root))

            all_paths = [
                cells_root / f"p{p:03d}_l{l:03d}.png"
                for p in range(prompt_count)
                for l in range(lora_count)
            ]
            all_meta_paths = [
                cells_root / f"p{p:03d}_l{l:03d}.json"
                for p in range(prompt_count)
                for l in range(lora_count)
            ]
            all_complete = all(p.exists() for p in all_paths) and all(
                p.exists() for p in all_meta_paths
            )

            if all_complete:
                metas = {
                    (p, l): json.loads(
                        (cells_root / f"p{p:03d}_l{l:03d}.json").read_text(encoding="utf-8")
                    )
                    for p in range(prompt_count)
                    for l in range(lora_count)
                }

                if grid_mode in ("per_prompt_and_master", "master_only"):
                    image_height = _scaled_image_height(all_paths, int(cell_width))
                    row_header_width = max(180, int(cell_width * 0.48))
                    title_height = max(68, int(font_size) * 3)
                    row_height = image_height + int(label_height)
                    master = Image.new(
                        "RGB",
                        (
                            row_header_width + int(cell_width) * lora_count,
                            title_height + row_height * prompt_count,
                        ),
                        (13, 13, 15),
                    )
                    draw = ImageDraw.Draw(master)
                    title_font = _font(max(17, int(font_size) + 4), bold=True)
                    draw.text(
                        (16, 12),
                        f"{output_prefix} — controlled LoRA comparison",
                        font=title_font,
                        fill=(250, 250, 250),
                    )
                    small_font = _font(max(12, int(font_size) - 3), bold=False)
                    draw.text(
                        (16, 42),
                        f"{prompt_count} prompts × {lora_count} LoRAs",
                        font=small_font,
                        fill=(190, 190, 198),
                    )

                    row_font = _font(max(14, int(font_size)), bold=True)
                    row_body_font = _font(max(10, int(font_size) - 5), bold=False)

                    for p in range(prompt_count):
                        row_y = title_height + p * row_height
                        draw.rectangle(
                            (0, row_y, row_header_width, row_y + row_height),
                            fill=(28, 28, 32) if p % 2 == 0 else (34, 34, 38),
                        )
                        meta = metas[(p, 0)]
                        draw.text(
                            (12, row_y + 12),
                            f"P{p + 1:02d}",
                            font=row_font,
                            fill=(255, 215, 90),
                        )
                        excerpt = re.sub(r"\s+", " ", meta.get("prompt_text") or "").strip()
                        excerpt_lines = _wrap_pixels(
                            draw,
                            excerpt,
                            row_body_font,
                            row_header_width - 24,
                            max_lines=8,
                        )
                        text_y = row_y + 46
                        for line in excerpt_lines:
                            draw.text(
                                (12, text_y),
                                line,
                                font=row_body_font,
                                fill=(220, 220, 225),
                            )
                            text_y += max(12, int((int(font_size) - 5) * 1.2))

                        for l in range(lora_count):
                            cell = _make_labeled_cell(
                                cells_root / f"p{p:03d}_l{l:03d}.png",
                                metas[(p, l)]["lora_label"],
                                int(cell_width),
                                image_height,
                                int(label_height),
                                int(font_size),
                            )
                            master.paste(
                                cell,
                                (row_header_width + l * int(cell_width), row_y),
                            )

                    master_output = run_root / f"{safe_prefix}_MASTER_GRID.png"
                    master.save(master_output, format="PNG")
                    completed_ui = [_ui_image_entry(master_output, output_root)]

                manifest = {
                    "plugin_version": PLUGIN_VERSION,
                    "run_id": run_id,
                    "output_prefix": output_prefix,
                    "prompt_count": prompt_count,
                    "lora_count": lora_count,
                    "entries": [
                        metas[(p, l)]
                        for p in range(prompt_count)
                        for l in range(lora_count)
                    ],
                }
                manifest_path = run_root / f"{safe_prefix}_manifest.json"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                complete_marker = run_root / ".matrix_complete"
                if not complete_marker.exists():
                    complete_marker.write_text("complete", encoding="utf-8")
                    if PromptServer is not None:
                        PromptServer.instance.send_sync(
                            "lorapromptqueue.matrix_complete",
                            {
                                "run_id": run_id,
                                "run_folder": str(run_root),
                                "manifest": str(manifest_path),
                            },
                        )

        if completed_ui:
            return {"ui": {"images": completed_ui}}
        return {"ui": {"text": [f"Collected P{prompt_index + 1} / LoRA {lora_index + 1}"]}}


class LoRAPromptQueueControllerV4:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Prompt one\n\n---PROMPT---\n\nPrompt two",
                    },
                ),
                "prompt_separator": ("STRING", {"default": "---PROMPT---"}),
                "prompt_parse_mode": (
                    ["separator_blocks", "one_nonempty_line_per_prompt"],
                ),
                "prompt_file_name": ("STRING", {"default": ""}),
                "selected_loras": ("STRING", {"multiline": True, "default": ""}),
                "excluded_lora_match": ("STRING", {"default": "krea2_turbo"}),
                "prompt_subset_mode": (
                    ["all", "first_n", "random_n"],
                ),
                "prompt_limit": (
                    "INT",
                    {"default": 10, "min": 1, "max": 10000},
                ),
                "prompt_sample_seed": (
                    "INT",
                    {"default": 20260710, "min": 0, "max": 0x7FFFFFFF},
                ),
                "identity_loader_node_id": (
                    "INT",
                    {"default": 19, "min": 0, "max": 100000},
                ),
                "positive_prompt_node_id": (
                    "INT",
                    {"default": 13, "min": 0, "max": 100000},
                ),
                "ksampler_node_id": (
                    "INT",
                    {"default": 9, "min": 0, "max": 100000},
                ),
                "save_node_id": (
                    "INT",
                    {"default": 10, "min": 0, "max": 100000},
                ),
                "grid_collector_node_id": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100000},
                ),
                "seed_step_per_prompt": (
                    "INT",
                    {"default": 1, "min": 0, "max": 1000000000},
                ),
                "output_prefix": ("STRING", {"default": "Krea2_Controlled_Test"}),
                "grid_mode": (
                    [
                        "per_prompt_and_master",
                        "master_only",
                        "per_prompt_only",
                        "off",
                    ],
                ),
                "grid_cell_width": (
                    "INT",
                    {"default": 384, "min": 160, "max": 1024, "step": 16},
                ),
                "grid_label_height": (
                    "INT",
                    {"default": 64, "min": 32, "max": 180, "step": 4},
                ),
                "grid_font_size": (
                    "INT",
                    {"default": 22, "min": 10, "max": 64, "step": 1},
                ),
                "auto_analyze_after_run": ("BOOLEAN", {"default": True}),
                "auto_open_report": ("BOOLEAN", {"default": True}),
                "analysis_mode": (
                    ["antelopev2_identity"],
                ),
                "analysis_reference_folder": (
                    "STRING",
                    {"default": "lora_reference"},
                ),
                "last_run_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "automation/LoRA testing"

    def noop(self, **kwargs):
        return ()


NODE_CLASS_MAPPINGS = {
    "LoRATestIdentityLoader": LoRATestIdentityLoader,
    "LoRATestGridCollector": LoRATestGridCollector,
    "LoRAPromptQueueControllerV4": LoRAPromptQueueControllerV4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRATestIdentityLoader": "LoRA Test — Identity Loader",
    "LoRATestGridCollector": "LoRA Test — Grid Collector",
    "LoRAPromptQueueControllerV4": "LoRA × Prompt Test Controller V4",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]


if PromptServer is not None and web is not None:
    @PromptServer.instance.routes.get("/lorapromptqueue/v4/loras")
    async def lora_prompt_queue_list_loras(request):
        return web.json_response({
            "version": PLUGIN_VERSION,
            "loras": _available_loras(),
        })

    @PromptServer.instance.routes.post("/lorapromptqueue/v4/analyze")
    async def lora_prompt_queue_analyze(request):
        try:
            payload = await request.json()
            result = _analyze_run(
                payload.get("run_id", ""),
                payload.get("reference_folder", "lora_reference"),
                payload.get("analysis_mode", "identity_if_available"),
            )
            return web.json_response({"ok": True, **result})
        except Exception as exc:
            return web.json_response(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status=400,
            )


    @PromptServer.instance.routes.get("/lorapromptqueue/v4/analyzer_status")
    async def lora_prompt_queue_analyzer_status(request):
        return web.json_response(read_analyzer_status())

    @PromptServer.instance.routes.post("/lorapromptqueue/v4/install_analyzer")
    async def lora_prompt_queue_install_analyzer(request):
        ensure_analyzer_async(force=True)
        return web.json_response({"ok": True, "status": read_analyzer_status()})

    @PromptServer.instance.routes.get("/lorapromptqueue/v4/report")
    async def lora_prompt_queue_report(request):
        try:
            output_root = Path(folder_paths.get_output_directory())
            run_id = request.query.get("run_id", "")
            run_root = _safe_child(
                output_root / "LoRA_Test_Grids",
                _safe_slug(run_id, "run"),
            )
            report = run_root / "AUTO_REPORT.html"
            if not report.exists():
                raise FileNotFoundError("Analysis report not found.")
            return web.FileResponse(report)
        except Exception as exc:
            return web.Response(
                text=f"Report error: {type(exc).__name__}: {exc}",
                status=404,
            )

    @PromptServer.instance.routes.get("/lorapromptqueue/v4/asset")
    async def lora_prompt_queue_asset(request):
        try:
            output_root = Path(folder_paths.get_output_directory())
            run_id = request.query.get("run_id", "")
            relative = request.query.get("path", "")
            run_root = _safe_child(
                output_root / "LoRA_Test_Grids",
                _safe_slug(run_id, "run"),
            )
            asset = _safe_child(run_root, relative)
            if not asset.exists() or not asset.is_file():
                raise FileNotFoundError("Asset not found.")
            return web.FileResponse(asset)
        except Exception as exc:
            return web.Response(
                text=f"Asset error: {type(exc).__name__}: {exc}",
                status=404,
            )


# LoRA Lab V6 is a dashboard-first product. Legacy nodes and routes remain available
# so existing workflows continue to load, while new runs no longer depend on a
# graph, numeric node IDs, or browser-side widget mutation.
try:
    from .lora_lab import (
        LAB_NODE_CLASS_MAPPINGS,
        LAB_NODE_DISPLAY_NAME_MAPPINGS,
    )

    NODE_CLASS_MAPPINGS.update(LAB_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LAB_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as exc:
    import traceback

    print(f"[LoRA Lab] V6 failed to load: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
