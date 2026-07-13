from __future__ import annotations

import importlib.util
import math
import sys
import threading
from pathlib import Path

import numpy as np


MODEL_FOLDER = "adaface_vit_base_kprpe_webface12m"
ALIGNER_FOLDER = "dfa_mobilenet"
MODEL_REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
ALIGNER_REPO = "minchul/cvlface_DFA_mobilenet"

_LOCK = threading.RLock()
_ANALYZER = None


def model_roots() -> tuple[Path, Path]:
    import folder_paths

    root = Path(folder_paths.models_dir) / "cvlface"
    return root / MODEL_FOLDER, root / ALIGNER_FOLDER


def ready_details() -> tuple[bool, dict]:
    model_root, aligner_root = model_roots()
    required = [
        model_root / "pretrained_model" / "model.pt",
        model_root / "pretrained_model" / "model.yaml",
        aligner_root / "pretrained_model" / "model.pt",
        aligner_root / "pretrained_model" / "model.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return not missing, {
        "name": "CVLFace ViT-KP-RPE AdaFace WebFace12M",
        "model_root": str(model_root),
        "aligner_root": str(aligner_root),
        "missing": missing,
    }


def _load_package(alias: str, package_root: Path, package_name: str):
    existing = sys.modules.get(alias)
    if existing is not None:
        return existing
    init_path = package_root / package_name / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        alias,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CVLFace package: {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _largest_face(faces):
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0]))
        * max(0.0, float(face.bbox[3] - face.bbox[1])),
    )


def _face_crop(image: np.ndarray, face) -> tuple[np.ndarray, float]:
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    side = max(32.0, max(x2 - x1, y2 - y1) * 1.55)
    left, top = int(math.floor(cx - side * 0.5)), int(math.floor(cy - side * 0.5))
    right, bottom = int(math.ceil(cx + side * 0.5)), int(math.ceil(cy + side * 0.5))
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Empty face crop")
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA if crop.shape[0] > 256 else cv2.INTER_CUBIC)
    return crop, side


class CVLFaceAnalyzer:
    def __init__(self):
        import torch
        import yaml
        from omegaconf import OmegaConf

        ready, details = ready_details()
        if not ready:
            raise FileNotFoundError(f"CVLFace files are missing: {details['missing']}")
        model_root, aligner_root = model_roots()
        model_package = _load_package("loralab_cvl_models", model_root, "models")
        aligner_package = _load_package("loralab_cvl_aligners", aligner_root, "aligners")

        model_config = OmegaConf.create(yaml.safe_load((model_root / "pretrained_model" / "model.yaml").read_text(encoding="utf-8")))
        aligner_config = OmegaConf.create(yaml.safe_load((aligner_root / "pretrained_model" / "model.yaml").read_text(encoding="utf-8")))
        self.model = model_package.get_model(model_config)
        self.model.load_state_dict_from_path(str(model_root / "pretrained_model" / "model.pt"))
        self.aligner = aligner_package.get_aligner(aligner_config)
        self.aligner.load_state_dict_from_path(str(aligner_root / "pretrained_model" / "model.pt"))
        self.model.eval()
        self.aligner.eval()
        self.device = torch.device("cpu")

    def ensure_device(self):
        import torch

        target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device != target:
            self.model.to(target)
            self.aligner.to(target)
            self.device = target
        return target

    def release_cuda(self):
        import torch

        if self.device.type == "cuda":
            self.model.to("cpu")
            self.aligner.to("cpu")
            self.device = torch.device("cpu")
            torch.cuda.empty_cache()

    def embedding(self, image_path, face_app) -> dict | None:
        import cv2
        import torch

        image = cv2.imread(str(image_path))
        if image is None:
            return None
        face = _largest_face(face_app.get(image))
        if face is None:
            return None
        crop, face_side = _face_crop(image, face)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(127.5).sub_(1.0).unsqueeze(0)
        device = self.ensure_device()
        tensor = tensor.to(device)
        with torch.inference_mode():
            aligned, _orig_landmarks, aligned_landmarks, align_score, _theta, _bbox = self.aligner(tensor)
            output = self.model(aligned, aligned_landmarks)
        output = output.float().reshape(output.shape[0], -1)
        raw_norm = float(torch.linalg.vector_norm(output, dim=1)[0].item())
        if raw_norm <= 1e-9:
            return None
        embedding = (output[0] / raw_norm).detach().cpu().numpy().astype(np.float32)
        detector_score = float(getattr(face, "det_score", 0.0) or 0.0)
        alignment_score = float(align_score.reshape(-1)[0].item())
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        size_score = float(np.clip(face_side / 180.0, 0.0, 1.0))
        sharpness_score = float(np.clip(math.log1p(sharpness) / math.log(501.0), 0.0, 1.0))
        quality = float(np.clip(0.35 * detector_score + 0.35 * alignment_score + 0.15 * size_score + 0.15 * sharpness_score, 0.0, 1.0))
        return {
            "embedding": embedding,
            "feature_norm": raw_norm,
            "quality": quality,
            "detector_score": detector_score,
            "alignment_score": alignment_score,
            "face_size": face_side,
            "sharpness": sharpness,
        }


def get_analyzer() -> CVLFaceAnalyzer:
    global _ANALYZER
    with _LOCK:
        if _ANALYZER is None:
            _ANALYZER = CVLFaceAnalyzer()
        return _ANALYZER


def release_cuda():
    with _LOCK:
        if _ANALYZER is not None:
            _ANALYZER.release_cuda()


def robust_template(records: list[dict]) -> dict:
    if len(records) < 2:
        raise RuntimeError("CVLFace needs at least two detectable reference faces.")
    embeddings = np.stack([record["embedding"] for record in records], axis=0)
    similarity = embeddings @ embeddings.T
    cohesion = np.median(similarity, axis=1)
    median = float(np.median(cohesion))
    mad = float(np.median(np.abs(cohesion - median)))
    keep = cohesion >= max(0.10, median - 2.5 * max(mad, 0.01))
    if int(np.sum(keep)) < 2:
        keep = np.ones(len(records), dtype=bool)
    inliers = embeddings[keep]
    quality = np.asarray([record["quality"] for record in records], dtype=np.float64)[keep]
    q_low, q_high = float(np.quantile(quality, 0.10)), float(np.quantile(quality, 0.90))
    normalized_quality = np.clip((quality - q_low) / max(1e-6, q_high - q_low), 0.0, 1.0)
    weights = 0.25 + 0.75 * normalized_quality
    centroid = np.average(inliers, axis=0, weights=weights)
    centroid /= max(1e-12, float(np.linalg.norm(centroid)))
    return {
        "centroid": centroid.astype(np.float32),
        "reference_embeddings": inliers.astype(np.float32),
        "weights": weights.astype(np.float32),
        "detected": len(records),
        "inliers": int(np.sum(keep)),
        "rejected": int(len(records) - np.sum(keep)),
        "cohesion": float(np.average(inliers @ centroid, weights=weights)),
        "mean_quality": float(np.mean(quality)),
    }


def similarity(record: dict | None, template: dict) -> dict | None:
    if record is None:
        return None
    embedding = record["embedding"]
    centroid_score = float(embedding @ template["centroid"])
    scores = np.sort(template["reference_embeddings"] @ embedding)[::-1]
    top_k = scores[: min(5, len(scores))]
    if len(top_k) >= 4:
        top_k = top_k[1:]  # remove one pose-copy maximum
    reference_score = float(np.mean(top_k))
    identity = 0.70 * centroid_score + 0.30 * reference_score
    return {
        "similarity": identity,
        "centroid_similarity": centroid_score,
        "trimmed_topk_similarity": reference_score,
        "quality": float(record["quality"]),
        "feature_norm": float(record["feature_norm"]),
        "detector_score": float(record["detector_score"]),
        "alignment_score": float(record["alignment_score"]),
    }
