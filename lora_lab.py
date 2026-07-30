from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import math
import os
import re
import threading
import time
import uuid
import zipfile
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

try:
    from .model_adapters import MODEL_PROFILES
except ImportError:
    from model_adapters import MODEL_PROFILES

try:
    from aiohttp import ClientSession, ClientTimeout, web
    from server import PromptServer
except Exception:
    ClientSession = None
    ClientTimeout = None
    web = None
    PromptServer = None


LAB_VERSION = "7.2.0"
BASELINE_VALUE = "__LORALAB_BASELINE__"
RUN_FOLDER = "LoRA_Lab"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_QUEUE_TASKS: dict[str, asyncio.Task] = {}
_WATCH_TASKS: dict[str, asyncio.Task] = {}
_RUN_LOCK = threading.RLock()


PROFILES = MODEL_PROFILES


PRESETS = {
    "quick": {
        "id": "quick",
        "name": "Quick screen",
        "description": "6 prompt categories, one matched seed. Use for every saved checkpoint.",
        "prompt_count": 6,
        "seeds": [20260710],
        "width": 768,
        "height": 768,
    },
    "standard": {
        "id": "standard",
        "name": "Standard comparison",
        "description": "8 prompt categories, two matched seeds at native 1024. Good default for finalists.",
        "prompt_count": 8,
        "seeds": [20260710, 20260711],
        "width": 1024,
        "height": 1024,
    },
    "deep": {
        "id": "deep",
        "name": "Deep final",
        "description": "8 prompt categories, three matched seeds. Use only for top 2–4 candidates.",
        "prompt_count": 8,
        "seeds": [20260710, 20260711, 20260712],
        "width": 1024,
        "height": 1024,
    },
}


DEFAULT_PROMPTS = [
    {
        "category": "anchor",
        "label": "Neutral anchor",
        "text": "{subject}, neutral head-and-shoulders studio portrait, facing the camera, relaxed closed-mouth expression, hair tucked behind one ear, plain mid-gray background, soft even daylight, natural skin texture",
    },
    {
        "category": "three_quarter",
        "label": "Three-quarter view",
        "text": "{subject}, realistic chest-up portrait from a three-quarter angle, looking back toward the camera, simple dark crew-neck shirt, softly blurred indoor background, gentle window light",
    },
    {
        "category": "profile",
        "label": "Profile structure",
        "text": "{subject}, clean side-profile portrait facing left, chin-length dark brown bob without bangs, minimal makeup, white shirt, pale seamless studio background, crisp softbox lighting",
    },
    {
        "category": "expression",
        "label": "Open smile",
        "text": "{subject}, candid waist-up photograph laughing with a broad natural smile and visible teeth, copper-red hair tied back, casual denim jacket, outdoor café in soft afternoon light",
    },
    {
        "category": "full_body",
        "label": "Full body",
        "text": "{subject}, full-body editorial photograph walking toward the camera, tailored navy suit and white sneakers, modern concrete gallery, balanced body proportions, diffuse skylight",
    },
    {
        "category": "hard_light",
        "label": "Hard lighting",
        "text": "{subject}, close cinematic portrait under hard directional side light, short black pixie haircut, serious expression, black turtleneck, deep blue night background, realistic shadow detail",
    },
    {
        "category": "occlusion",
        "label": "Glasses and hat",
        "text": "{subject}, realistic upper-body street photograph wearing clear eyeglasses and a knitted beanie, shoulder-length auburn hair, olive coat and scarf, light snowfall, overcast winter daylight",
    },
    {
        "category": "color_shift",
        "label": "Unseen styling",
        "text": "{subject}, polished beauty portrait with a sleek violet shoulder-length hairstyle and no bangs, subtle silver eye makeup, ivory blouse, warm beige background, large diffused key light",
    },
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_slug(value: object, fallback: str = "item") -> str:
    value = str(value or "").strip().replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:160] or fallback


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _run_root(run_id: str) -> Path:
    return Path(folder_paths.get_output_directory()) / RUN_FOLDER / _safe_slug(run_id, "run")


def _run_file(run_id: str) -> Path:
    return _run_root(run_id) / "LAB_RUN.json"


def _read_run(run_id: str) -> dict:
    path = _run_file(run_id)
    if not path.exists():
        raise FileNotFoundError(f"LoRA Lab run not found: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_run(run: dict) -> None:
    run["updated_at"] = _now_iso()
    _atomic_json(_run_file(run["run_id"]), run)


def _available_loras() -> list[str]:
    try:
        return sorted(folder_paths.get_filename_list("loras"), key=str.lower)
    except Exception:
        return []


def _available_files(folder: str) -> list[str]:
    try:
        return sorted(folder_paths.get_filename_list(folder), key=str.lower)
    except Exception:
        return []


def _extract_step(filename: str) -> int | None:
    stem = Path(filename).stem
    patterns = [
        r"(?:save|step)[_-]?(\d{3,7})",
        r"[_-]0*(\d{4,7})(?:$|[_-])",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, stem, flags=re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass
    return None


def _extract_training_progress(filename: str) -> dict:
    stem = Path(filename).stem
    match = re.search(r"(?:save|backup)[_-](\d+)[_-](\d+)[_-](\d+)", stem, flags=re.IGNORECASE)
    if not match:
        return {"step": _extract_step(filename), "epoch": None, "epoch_step": None}
    return {"step": int(match.group(1)), "epoch": int(match.group(2)), "epoch_step": int(match.group(3))}


def _lora_group(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_20\d\d-\d\d-\d\d[_-]\d\d[-:]\d\d[-:]\d\d.*$", "", stem)
    stem = re.sub(r"(?:[_-](?:save|step)[_-]?\d+.*)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_-]0+\d{4,7}$", "", stem)
    return stem.strip("_- ") or Path(filename).stem


def _lora_label(filename: str, strength: float | None = None) -> str:
    if filename == BASELINE_VALUE:
        return "Baseline · no LoRA"
    step = _extract_step(filename)
    stem = Path(filename).stem
    base = f"step {step}" if step is not None else stem
    if strength is not None:
        base += f" · {float(strength):.2f}"
    return base


def _lora_catalog() -> list[dict]:
    catalog = []
    for filename in _available_loras():
        progress = _extract_training_progress(filename)
        path = folder_paths.get_full_path("loras", filename)
        stat = Path(path).stat() if path and Path(path).exists() else None
        catalog.append({
            "filename": filename,
            "label": _lora_label(filename),
            "step": progress["step"],
            "epoch": progress["epoch"],
            "epoch_step": progress["epoch_step"],
            "group": _lora_group(filename),
            "size_mb": round(stat.st_size / 1024 / 1024, 1) if stat else None,
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)) if stat else None,
        })
    return catalog


def _preferred_turbo_lora() -> str | None:
    names = _available_loras()
    preferred = [
        "krea2_turbo_lora_rank_64_bf16(1).safetensors",
        "krea2_turbo_lora_rank_64_bf16.safetensors",
    ]
    for target in preferred:
        for name in names:
            if name.lower() == target.lower():
                return name
    return next((name for name in names if "krea2" in name.lower() and "turbo" in name.lower() and "lora" in name.lower()), None)


def _normalize_turbo_lora(payload: dict, available: set[str]) -> dict:
    raw = payload.get("turbo_lora") if isinstance(payload.get("turbo_lora"), dict) else {}
    enabled = bool(raw.get("enabled", False))
    filename = str(raw.get("filename") or _preferred_turbo_lora() or "")
    strength = float(raw.get("strength", 1.0))
    if enabled and not filename:
        raise FileNotFoundError("Krea 2 Turbo LoRA is not installed.")
    if enabled and filename not in available:
        raise FileNotFoundError(f"Turbo LoRA is not installed: {filename}")
    if strength < -2.0 or strength > 2.0:
        raise ValueError("Turbo LoRA strength must be between -2 and 2.")
    return {"enabled": enabled, "filename": filename, "strength": strength}


def _normalize_aux_loras(payload: dict, available: set[str]) -> list[dict]:
    raw = payload.get("aux_loras") or []
    if not isinstance(raw, list):
        raise ValueError("Always-on auxiliary LoRAs must be a list.")
    normalized = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Auxiliary LoRA {index + 1} must be an object.")
        enabled = bool(item.get("enabled", True))
        filename = str(item.get("filename") or "").strip()
        strength = float(item.get("strength", 1.0))
        if not enabled:
            continue
        if not filename:
            raise ValueError(f"Auxiliary LoRA {index + 1} has no file selected.")
        if filename not in available:
            raise FileNotFoundError(f"Auxiliary LoRA is not installed: {filename}")
        if filename in seen:
            raise ValueError(f"Auxiliary LoRA is listed more than once: {filename}")
        if strength < -2.0 or strength > 2.0:
            raise ValueError("Auxiliary LoRA strengths must be between -2 and 2.")
        seen.add(filename)
        normalized.append({"filename": filename, "strength": strength})
    if len(normalized) > 8:
        raise ValueError("Maximum eight always-on auxiliary LoRAs per run.")
    return normalized


def _model_filename(profile: dict) -> str:
    names = _available_files(profile.get("model_folder", "diffusion_models"))
    lower = [(name, name.lower()) for name in names]
    for needle in profile["model_contains"]:
        for name, value in lower:
            if needle in value:
                return name
    raise FileNotFoundError(f"No installed diffusion model matches {profile['model_contains']}")


def _profile_file(profile: dict, key: str, folder: str) -> str:
    needles = profile.get(key) or []
    names = _available_files(folder)
    for needle in needles:
        for name in names:
            if str(needle).lower() in name.lower():
                return name
    raise FileNotFoundError(f"No installed {folder} file matches {needles}")


def _model_patch_catalog() -> list[dict]:
    """Return installed nodes that accept and return MODEL; used by advanced UI."""
    try:
        import nodes
    except Exception:
        return []
    result = []
    for class_type, node_class in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            input_types = node_class.INPUT_TYPES()
            required = input_types.get("required", {})
            optional = input_types.get("optional", {})
            model_spec = required.get("model") or optional.get("model")
            returns = tuple(getattr(node_class, "RETURN_TYPES", ()))
            if model_spec and model_spec[0] == "MODEL" and returns and returns[0] == "MODEL":
                result.append({
                    "class_type": class_type,
                    "display_name": getattr(node_class, "TITLE", None) or class_type,
                })
        except Exception:
            continue
    return sorted(result, key=lambda item: item["class_type"].lower())


def _normalize_model_patches(value: object) -> list[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("MODEL patch chain must be a JSON array.")
    try:
        import nodes
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect ComfyUI nodes: {exc}") from exc
    patches = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"MODEL patch {index + 1} must be an object.")
        class_type = str(item.get("class_type") or "").strip()
        node_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            raise ValueError(f"MODEL patch node is not installed: {class_type}")
        input_types = node_class.INPUT_TYPES()
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})
        model_spec = required.get("model") or optional.get("model")
        returns = tuple(getattr(node_class, "RETURN_TYPES", ()))
        if not model_spec or model_spec[0] != "MODEL" or not returns or returns[0] != "MODEL":
            raise ValueError(f"{class_type} is not a MODEL-to-MODEL node.")
        inputs = item.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ValueError(f"Inputs for {class_type} must be an object.")
        if "model" in inputs:
            raise ValueError(f"Do not set model input for {class_type}; LoRA Lab wires it automatically.")
        patches.append({"class_type": class_type, "inputs": inputs})
    if len(patches) > 16:
        raise ValueError("Maximum 16 MODEL patch nodes per run.")
    return patches


def _preferred_filename(folder: str, needles: list[str]) -> str:
    names = list(folder_paths.get_filename_list(folder))
    for needle in needles:
        for name in names:
            if needle.lower() in name.lower():
                return name
    raise FileNotFoundError(f"No {folder} file matches: {needles}")


def _reference_folders() -> list[dict]:
    root = Path(folder_paths.get_input_directory())
    results = []
    if not root.exists():
        return results
    for directory, subdirs, files in os.walk(root):
        relative = Path(directory).relative_to(root)
        if len(relative.parts) > 3:
            subdirs[:] = []
            continue
        count = sum(Path(name).suffix.lower() in IMAGE_EXTENSIONS for name in files)
        if count >= 2:
            results.append({"folder": relative.as_posix() or ".", "count": count})
    return sorted(results, key=lambda item: (-item["count"], item["folder"].lower()))


def _hardware() -> dict:
    gpu = "CPU"
    vram_gb = 0.0
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = properties.name
        vram_gb = round(properties.total_memory / 1024 ** 3, 1)
    return {"gpu": gpu, "vram_gb": vram_gb, "cuda": torch.cuda.is_available()}


class LoRALabIdentityLoader:
    def __init__(self):
        self._cached_path = None
        self._cached_mtime = None
        self._cached_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": ([BASELINE_VALUE, *_available_loras()],),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "automation/LoRA Lab"
    DESCRIPTION = "Internal LoRA Lab adapter. Supports a true no-LoRA baseline and model-only identity LoRAs."

    def load_lora(self, model, lora_name, strength_model):
        if lora_name == BASELINE_VALUE or abs(float(strength_model)) < 1e-12:
            return (model,)
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"LoRA not found: {lora_name}")
        mtime = Path(path).stat().st_mtime_ns
        if path != self._cached_path or mtime != self._cached_mtime:
            self._cached_lora = comfy.utils.load_torch_file(path, safe_load=True)
            self._cached_path = path
            self._cached_mtime = mtime
        loaded, _ = comfy.sd.load_lora_for_models(
            model, None, self._cached_lora, float(strength_model), 0.0
        )
        return (loaded,)


LAB_NODE_CLASS_MAPPINGS = {"LoRALabIdentityLoader": LoRALabIdentityLoader}
LAB_NODE_DISPLAY_NAME_MAPPINGS = {"LoRALabIdentityLoader": "LoRA Lab · Identity / Baseline Loader"}


def _normalize_prompts(payload: dict, trigger: str) -> list[dict]:
    raw = payload.get("prompts") or DEFAULT_PROMPTS
    prompts = []
    subject_class = str(payload.get("subject_class") or "").strip()
    subject = " ".join(part for part in (trigger.strip(), subject_class) if part)
    for index, item in enumerate(raw):
        if isinstance(item, dict) and item.get("enabled") is False:
            continue
        if isinstance(item, str):
            label = f"Prompt {index + 1}"
            category = f"custom_{index + 1}"
            text = item
        elif isinstance(item, dict):
            label = str(item.get("label") or f"Prompt {index + 1}").strip()
            category = _safe_slug(item.get("category") or label, f"prompt_{index + 1}")
            text = str(item.get("text") or "")
        else:
            continue
        text = text.replace("{trigger}", trigger.strip()).replace("{class}", subject_class).replace("{subject}", subject)
        text = text.strip()
        if text:
            prompts.append({"label": label[:100], "category": category, "text": text})
    if not prompts:
        raise ValueError("At least one non-empty prompt is required.")
    if len(prompts) > 24:
        raise ValueError("Maximum 24 prompts per run.")
    return prompts


def _normalize_seeds(value: object) -> list[int]:
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value.strip())
    if not isinstance(value, list):
        value = [value]
    seeds = []
    for item in value:
        if item in (None, ""):
            continue
        seed = int(item)
        if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"Seed outside supported range: {seed}")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        seeds = [20260710]
    if len(seeds) > 5:
        raise ValueError("Maximum five matched seeds per run.")
    return seeds


def _normalize_dimension(value: object, default: int) -> int:
    value = int(value or default)
    if value < 512 or value > 2048:
        raise ValueError("Width and height must be between 512 and 2048.")
    return max(512, int(round(value / 8) * 8))


def _build_candidates(payload: dict, available: set[str]) -> list[dict]:
    selected = payload.get("selected_loras") or []
    if isinstance(selected, str):
        selected = [line.strip() for line in selected.splitlines() if line.strip()]
    selected = list(dict.fromkeys(map(str, selected)))
    missing = [name for name in selected if name not in available]
    if missing:
        raise FileNotFoundError(f"Selected LoRA files are not installed: {missing[:5]}")
    mode = str(payload.get("mode") or "compare")
    include_baseline = bool(payload.get("include_baseline", True))
    candidates = []
    if mode == "stack_compare":
        if len(selected) != 1:
            raise ValueError("Raw versus Turbo comparison requires exactly one identity LoRA.")
        if include_baseline:
            candidates.extend([
                {"filename": BASELINE_VALUE, "label": "Control · Raw only", "strength": 0.0, "baseline": True, "step": None, "stack_turbo": False},
                {"filename": BASELINE_VALUE, "label": "Control · Raw + Turbo LoRA", "strength": 0.0, "baseline": True, "step": None, "stack_turbo": True},
            ])
        step = _extract_step(selected[0])
        candidates.extend([
            {"filename": selected[0], "label": f"{_lora_label(selected[0])} · Raw", "strength": float(payload.get("common_strength", 1.0)), "baseline": False, "step": step, "stack_turbo": False},
            {"filename": selected[0], "label": f"{_lora_label(selected[0])} · Raw + Turbo", "strength": float(payload.get("common_strength", 1.0)), "baseline": False, "step": step, "stack_turbo": True},
        ])
    elif mode == "enhancer_compare":
        if len(selected) != 1:
            raise ValueError("Enhancer comparison requires exactly one identity LoRA.")
        if include_baseline:
            candidates.append({"filename": BASELINE_VALUE, "label": "Control · no identity LoRA", "strength": 0.0, "baseline": True, "step": None, "enhancer_variant": "off"})
        step = _extract_step(selected[0])
        strength = float(payload.get("common_strength", 1.0))
        candidates.extend([
            {"filename": selected[0], "label": f"{_lora_label(selected[0])} · enhancer off", "strength": strength, "baseline": False, "step": step, "enhancer_variant": "off"},
            {"filename": selected[0], "label": f"{_lora_label(selected[0])} · enhancer standard", "strength": strength, "baseline": False, "step": step, "enhancer_variant": "standard"},
            {"filename": selected[0], "label": f"{_lora_label(selected[0])} · enhancer advanced", "strength": strength, "baseline": False, "step": step, "enhancer_variant": "advanced"},
        ])
    elif include_baseline:
        candidates.append({
            "filename": BASELINE_VALUE,
            "label": _lora_label(BASELINE_VALUE),
            "strength": 0.0,
            "baseline": True,
            "step": None,
        })
    if mode == "strength":
        if len(selected) != 1:
            raise ValueError("Strength sweep requires exactly one LoRA.")
        strengths = payload.get("strengths") or [0.65, 0.8, 0.95, 1.1, 1.25]
        for strength in strengths:
            strength = float(strength)
            if strength < -2.0 or strength > 2.0:
                raise ValueError("Strength sweep values must be between -2 and 2.")
            candidates.append({
                "filename": selected[0],
                "label": _lora_label(selected[0], strength),
                "strength": strength,
                "baseline": False,
                "step": _extract_step(selected[0]),
            })
    elif mode not in {"stack_compare", "enhancer_compare"}:
        strength = float(payload.get("common_strength", 1.0))
        if strength < -2.0 or strength > 2.0:
            raise ValueError("LoRA strength must be between -2 and 2.")
        for filename in selected:
            candidates.append({
                "filename": filename,
                "label": _lora_label(filename),
                "strength": strength,
                "baseline": False,
                "step": _extract_step(filename),
            })
    real = [candidate for candidate in candidates if not candidate["baseline"]]
    if not real:
        raise ValueError("Select at least one LoRA candidate.")
    if any(float(candidate.get("strength", 0.0)) < -2.0 or float(candidate.get("strength", 0.0)) > 2.0 for candidate in candidates):
        raise ValueError("LoRA strength must be between -2 and 2.")
    if len(candidates) > 33:
        raise ValueError("Maximum 32 LoRA variants plus baseline per run.")
    labels = [item["label"] for item in candidates]
    if len(set(labels)) != len(labels):
        for index, item in enumerate(candidates):
            if labels.count(item["label"]) > 1:
                item["label"] = f"{item['label']} · C{index + 1}"
    return candidates


def _make_plan(payload: dict) -> dict:
    profile_id = str(payload.get("profile") or "krea2_turbo")
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile_id}")
    profile = dict(PROFILES[profile_id])
    workflow_adapter = str(payload.get("workflow_adapter") or "native")
    if workflow_adapter not in {"native", "api_template"}:
        raise ValueError("Workflow adapter must be native or api_template.")
    api_workflow = _validate_api_template(payload.get("api_workflow")) if workflow_adapter == "api_template" else None
    trigger = str(payload.get("trigger") or "").strip()
    prompts = _normalize_prompts(payload, trigger)
    seeds = _normalize_seeds(payload.get("seeds"))
    width = _normalize_dimension(payload.get("width"), 1024)
    height = _normalize_dimension(payload.get("height"), 1024)
    available_loras = set(_available_loras())
    candidates = _build_candidates(payload, available_loras)
    mode = str(payload.get("mode") or "compare")
    turbo_lora = _normalize_turbo_lora(payload, available_loras)
    aux_loras = _normalize_aux_loras(payload, available_loras)
    if mode == "stack_compare":
        turbo_lora["enabled"] = True
        if not turbo_lora["filename"] or turbo_lora["filename"] not in available_loras:
            raise FileNotFoundError("Raw versus Turbo comparison requires an installed Krea 2 Turbo LoRA.")
    if turbo_lora["enabled"]:
        if any(candidate["filename"] == turbo_lora["filename"] for candidate in candidates):
            raise ValueError("Turbo LoRA toggle already applies this LoRA. Do not also select it as a candidate.")
        if any(item["filename"] == turbo_lora["filename"] for item in aux_loras):
            raise ValueError("Turbo LoRA is also in the always-on auxiliary stack. Remove the duplicate.")
        if mode != "stack_compare":
            for candidate in candidates:
                if candidate.get("baseline"):
                    candidate["label"] = "Control · Turbo LoRA only"
    candidate_files = {candidate["filename"] for candidate in candidates if not candidate.get("baseline")}
    duplicate_candidates = sorted(candidate_files & {item["filename"] for item in aux_loras})
    if duplicate_candidates:
        raise ValueError(f"Candidate LoRAs cannot also be always-on auxiliaries: {duplicate_candidates[:3]}")
    scenarios = []
    for prompt_index, prompt in enumerate(prompts):
        for seed_index, seed in enumerate(seeds):
            scenarios.append({
                "scenario_index": len(scenarios),
                "prompt_index": prompt_index,
                "seed_index": seed_index,
                "seed": seed,
                **prompt,
            })
    total = len(scenarios) * len(candidates)
    if total > 500:
        raise ValueError(f"Run requests {total} cells; maximum is 500. Use staged testing.")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = _safe_slug(payload.get("output_prefix") or "LoRA_Lens", "LoRA_Lens")
    run_id = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    model_folder = profile.get("model_folder", "diffusion_models")
    installed_models = _available_files(model_folder)
    installed_clips = _available_files("text_encoders")
    installed_vaes = _available_files("vae")
    if workflow_adapter == "native":
        model_name = str(payload.get("model_name") or _model_filename(profile))
        clip_name = str(payload.get("clip_name") or _profile_file(profile, "clip_contains", "text_encoders"))
        clip_name_2 = str(payload.get("clip_name_2") or (_profile_file(profile, "clip_2_contains", "text_encoders") if profile.get("clip_2_contains") else ""))
        vae_name = str(payload.get("vae_name") or _profile_file(profile, "vae_contains", "vae"))
    else:
        model_name = str(payload.get("model_name") or "")
        clip_name = str(payload.get("clip_name") or "")
        clip_name_2 = str(payload.get("clip_name_2") or "")
        vae_name = str(payload.get("vae_name") or "")
    if workflow_adapter == "native" and model_name not in installed_models:
        raise FileNotFoundError(f"Model is not installed in {model_folder}: {model_name}")
    model_lower = model_name.lower()
    if workflow_adapter == "native" and turbo_lora["enabled"] and not profile.get("supports_acceleration_lora"):
        raise ValueError(f"{profile['name']} does not support the Krea 2 acceleration LoRA toggle.")
    if turbo_lora["enabled"] and "turbo" in model_lower:
        raise ValueError("A distilled model plus its acceleration LoRA would apply acceleration twice. Select the matching base model.")
    if workflow_adapter == "native" and clip_name not in installed_clips:
        raise FileNotFoundError(f"Text encoder is not installed: {clip_name}")
    if workflow_adapter == "native" and profile.get("adapter") == "split_dual" and clip_name_2 not in installed_clips:
        raise FileNotFoundError(f"Second text encoder is not installed: {clip_name_2}")
    if workflow_adapter == "native" and vae_name not in installed_vaes:
        raise FileNotFoundError(f"VAE is not installed: {vae_name}")
    advanced = payload.get("advanced") if isinstance(payload.get("advanced"), dict) else {}
    profile.update({
        "steps": int(advanced.get("steps", profile["steps"])),
        "cfg": float(advanced.get("cfg", profile["cfg"])),
        "sampler": str(advanced.get("sampler", profile["sampler"])),
        "scheduler": str(advanced.get("scheduler", profile["scheduler"])),
        "negative_mode": str(advanced.get("negative_mode", profile["negative_mode"])),
    })
    if turbo_lora["enabled"]:
        profile.update({
            "steps": 8,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "beta",
            "negative_mode": "zero",
        })
    if profile["steps"] < 1 or profile["steps"] > 1000:
        raise ValueError("Steps must be between 1 and 1000.")
    if profile["cfg"] < 0 or profile["cfg"] > 100:
        raise ValueError("CFG must be between 0 and 100.")
    if profile["sampler"] not in comfy.samplers.KSampler.SAMPLERS:
        raise ValueError(f"Unknown sampler: {profile['sampler']}")
    if profile["scheduler"] not in comfy.samplers.KSampler.SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {profile['scheduler']}")
    if profile["negative_mode"] not in {"zero", "encode"}:
        raise ValueError("Negative conditioning mode must be zero or encode.")
    model_patches = _normalize_model_patches(payload.get("model_patches"))
    if mode == "enhancer_compare":
        enhancer_patches = {
            "off": [],
            "standard": _normalize_model_patches([{"class_type": "ComfyUI-Krea2T-Enhancer", "inputs": {"enabled": True, "strength": 1.0, "debug": False}}]),
            "advanced": _normalize_model_patches([{"class_type": "Krea2T-Enhancer-Advanced", "inputs": {"enabled": True, "strength": 1.0, "text_scale": 1.5, "debug": False}}]),
        }
        for candidate in candidates:
            variant = candidate.get("enhancer_variant", "off")
            candidate["model_patches"] = [*enhancer_patches[variant], *model_patches]
    run = {
        "schema": 1,
        "lab_version": LAB_VERSION,
        "run_id": run_id,
        "name": str(payload.get("name") or prefix),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "planned",
        "objective": str(payload.get("objective") or "custom"),
        "mode": mode,
        "profile": profile,
        "workflow_adapter": workflow_adapter,
        "api_workflow": api_workflow,
        "api_output_node_id": str(payload.get("api_output_node_id") or ""),
        "api_output_index": int(payload.get("api_output_index") or 0),
        "model_name": model_name,
        "clip_name": clip_name,
        "clip_name_2": clip_name_2,
        "vae_name": vae_name,
        "model_patches": model_patches,
        "turbo_lora": turbo_lora,
        "aux_loras": aux_loras,
        "compatibility": {
            "model_family": "Custom API workflow" if workflow_adapter == "api_template" else profile.get("family", "Custom"),
            "variant": "Imported" if workflow_adapter == "api_template" else profile.get("variant", "Custom"),
            "adapter": workflow_adapter if workflow_adapter == "api_template" else profile.get("adapter", "split_single"),
            "validated": True,
            "turbo_double_apply": False,
            "aux_lora_count": len(aux_loras),
            "model_patch_count": len(model_patches),
        },
        "width": width,
        "height": height,
        "trigger": trigger,
        "subject_class": str(payload.get("subject_class") or "").strip(),
        "negative_prompt": str(payload.get("negative_prompt") or "deformed face, duplicate person, malformed hands"),
        "reference_folder": str(payload.get("reference_folder") or "lora_reference"),
        "grid_mode": str(payload.get("grid_mode") or "off"),
        "output_prefix": prefix,
        "prompts": prompts,
        "seeds": seeds,
        "scenarios": scenarios,
        "candidates": candidates,
        "scenario_count": len(scenarios),
        "candidate_count": len(candidates),
        "expected_cells": total,
        "submitted_prompt_ids": [],
        "submitted_jobs": {},
        "queue_errors": [],
        "auto_cleanup": True,
        "estimate": {
            "jobs": total,
            "seconds": int(total * (28.0 if (turbo_lora["enabled"] and profile["steps"] == 8) else profile["seconds_per_cell_4090"] * profile["steps"] / max(1, PROFILES[profile_id]["steps"])) * (width * height) / (1024 * 1024)),
            "storage_mb": int(math.ceil(total * width * height * 3.0 / 1024 / 1024 * 0.42)),
        },
    }
    root = _run_root(run_id)
    root.mkdir(parents=True, exist_ok=False)
    _write_run(run)
    return run


def _job_key(scenario_index: int, candidate_index: int) -> str:
    return f"p{scenario_index:03d}_l{candidate_index:03d}"


def _replace_template_values(value, replacements: dict):
    if isinstance(value, dict):
        return {key: _replace_template_values(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_template_values(item, replacements) for item in value]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def _validate_api_template(value: object) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value:
        raise ValueError("The API workflow must be a non-empty JSON object.")
    if len(value) > 500:
        raise ValueError("The API workflow exceeds the 500-node safety limit.")
    for node_id, node in value.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"Node {node_id} is not in ComfyUI API format. Export with 'Save (API Format)'.")
    if not any(node.get("class_type") == "LoRALabIdentityLoader" for node in value.values()):
        raise ValueError("Custom workflows must contain one LoRA Lab · Identity / Baseline Loader node.")
    return deepcopy(value)


def _custom_job_prompt(run: dict, scenario: dict, candidate: dict, candidate_index: int) -> dict:
    profile = run["profile"]
    replacements = {
        "{{PROMPT}}": scenario["text"], "{{NEGATIVE_PROMPT}}": run["negative_prompt"],
        "{{MODEL}}": run.get("model_name", ""), "{{CLIP}}": run.get("clip_name", ""),
        "{{CLIP_2}}": run.get("clip_name_2", ""), "{{VAE}}": run.get("vae_name", ""),
        "{{SEED}}": int(scenario["seed"]), "{{STEPS}}": int(profile["steps"]),
        "{{CFG}}": float(profile["cfg"]), "{{SAMPLER}}": profile["sampler"],
        "{{SCHEDULER}}": profile["scheduler"], "{{WIDTH}}": int(run["width"]),
        "{{HEIGHT}}": int(run["height"]),
    }
    workflow = _replace_template_values(run["api_workflow"], replacements)
    identity_nodes = []
    for node_id, node in workflow.items():
        if node.get("class_type") == "LoRALabIdentityLoader":
            node["inputs"]["lora_name"] = candidate["filename"]
            node["inputs"]["strength_model"] = float(candidate["strength"])
            identity_nodes.append((str(node_id), node))
    numeric_ids = [int(key) for key in workflow if str(key).isdigit()]
    next_node_id = max(numeric_ids, default=0) + 1
    for _, identity_node in identity_nodes:
        model_link = identity_node["inputs"].get("model")
        if not isinstance(model_link, list) or len(model_link) != 2:
            raise ValueError("Each LoRA Lab Identity Loader must have a connected MODEL input.")
        for aux in run.get("aux_loras") or []:
            aux_id = str(next_node_id)
            next_node_id += 1
            workflow[aux_id] = {
                "class_type": "LoRALabIdentityLoader",
                "inputs": {
                    "model": model_link,
                    "lora_name": aux["filename"],
                    "strength_model": float(aux["strength"]),
                },
            }
            model_link = [aux_id, 0]
        identity_node["inputs"]["model"] = model_link
    image_link = None
    output_node_id = str(run.get("api_output_node_id") or "").strip()
    if output_node_id:
        image_link = [output_node_id, int(run.get("api_output_index", 0))]
    else:
        for node in workflow.values():
            if node.get("class_type") in {"SaveImage", "PreviewImage"}:
                candidate_link = node.get("inputs", {}).get("images")
                if isinstance(candidate_link, list) and len(candidate_link) == 2:
                    image_link = candidate_link
                    break
    if image_link is None:
        raise ValueError("Custom workflow output was not found. Add PreviewImage/SaveImage or enter an IMAGE output node ID.")
    collector_id = str(next_node_id)
    workflow[collector_id] = {
        "class_type": "LoRATestGridCollector",
        "inputs": {
            "images": image_link, "run_id": run["run_id"],
            "prompt_index": int(scenario["scenario_index"]), "lora_index": int(candidate_index),
            "prompt_count": int(run["scenario_count"]), "lora_count": int(run["candidate_count"]),
            "prompt_label": f"{scenario['label']} · seed {scenario['seed']}",
            "prompt_text": scenario["text"], "lora_label": candidate["label"],
            "output_prefix": run["output_prefix"], "grid_mode": run["grid_mode"],
            "cell_width": 384, "label_height": 58, "font_size": 20,
        },
    }
    return workflow


def _job_prompt(run: dict, scenario: dict, candidate: dict, candidate_index: int) -> dict:
    if run.get("workflow_adapter") == "api_template":
        return _custom_job_prompt(run, scenario, candidate, candidate_index)
    profile = run["profile"]
    positive_id = "5"
    if profile["negative_mode"] == "zero":
        negative_node = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [positive_id, 0]}}
    else:
        negative_node = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": run["negative_prompt"], "clip": ["2", 0]},
        }
    label = candidate["label"]
    prompt_label = f"{scenario['label']} · seed {scenario['seed']}"
    clip_node = {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": run["clip_name"],
            "clip_name2": run["clip_name_2"],
            "type": profile["clip_type"],
            "device": "default",
        },
    } if profile.get("adapter") == "split_dual" else {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": run["clip_name"], "type": profile["clip_type"], "device": "default"},
    }
    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": run["model_name"], "weight_dtype": "default"},
        },
        "2": clip_node,
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": run["vae_name"]}},
        "4": {
            "class_type": "LoRALabIdentityLoader",
            "inputs": {
                "model": ["1", 0],
                "lora_name": candidate["filename"],
                "strength_model": float(candidate["strength"]),
            },
        },
        positive_id: {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": scenario["text"], "clip": ["2", 0]},
        },
        "6": negative_node,
        "7": {
            "class_type": profile.get("latent", "EmptyLatentImage"),
            "inputs": {"width": run["width"], "height": run["height"], "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "seed": int(scenario["seed"]),
                "steps": int(profile["steps"]),
                "cfg": float(profile["cfg"]),
                "sampler_name": profile["sampler"],
                "scheduler": profile["scheduler"],
                "positive": [positive_id, 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "LoRATestGridCollector",
            "inputs": {
                "images": ["9", 0],
                "run_id": run["run_id"],
                "prompt_index": int(scenario["scenario_index"]),
                "lora_index": int(candidate_index),
                "prompt_count": int(run["scenario_count"]),
                "lora_count": int(run["candidate_count"]),
                "prompt_label": prompt_label,
                "prompt_text": scenario["text"],
                "lora_label": label,
                "output_prefix": run["output_prefix"],
                "grid_mode": run["grid_mode"],
                "cell_width": 384,
                "label_height": 58,
                "font_size": 20,
            },
        },
    }
    next_patch_id = 11
    loader_model_link = ["1", 0]
    sampling_patch = profile.get("model_sampling")
    if sampling_patch:
        sampling_id = str(next_patch_id)
        sampling_inputs = dict(sampling_patch.get("inputs") or {})
        sampling_inputs["model"] = loader_model_link
        workflow[sampling_id] = {"class_type": sampling_patch["class_type"], "inputs": sampling_inputs}
        loader_model_link = [sampling_id, 0]
        workflow["4"]["inputs"]["model"] = loader_model_link
        next_patch_id += 1
    turbo_lora = run.get("turbo_lora") or {}
    use_turbo_lora = bool(candidate.get("stack_turbo")) if "stack_turbo" in candidate else bool(turbo_lora.get("enabled"))
    if use_turbo_lora:
        turbo_node_id = str(next_patch_id)
        workflow[turbo_node_id] = {
            "class_type": "LoRALabIdentityLoader",
            "inputs": {
                "model": loader_model_link,
                "lora_name": turbo_lora["filename"],
                "strength_model": float(turbo_lora.get("strength", 1.0)),
            },
        }
        workflow["4"]["inputs"]["model"] = [turbo_node_id, 0]
        loader_model_link = [turbo_node_id, 0]
        next_patch_id += 1
    for aux in run.get("aux_loras") or []:
        aux_node_id = str(next_patch_id)
        workflow[aux_node_id] = {
            "class_type": "LoRALabIdentityLoader",
            "inputs": {
                "model": loader_model_link,
                "lora_name": aux["filename"],
                "strength_model": float(aux["strength"]),
            },
        }
        loader_model_link = [aux_node_id, 0]
        workflow["4"]["inputs"]["model"] = loader_model_link
        next_patch_id += 1
    model_link = ["4", 0]
    candidate_patches = candidate.get("model_patches") if "model_patches" in candidate else run.get("model_patches")
    for patch_index, patch in enumerate(candidate_patches or []):
        node_id = str(next_patch_id + patch_index)
        inputs = dict(patch.get("inputs") or {})
        inputs["model"] = model_link
        workflow[node_id] = {"class_type": patch["class_type"], "inputs": inputs}
        model_link = [node_id, 0]
    workflow["8"]["inputs"]["model"] = model_link
    return workflow


def _cell_path(run_id: str, scenario_index: int, candidate_index: int) -> Path:
    legacy = Path(folder_paths.get_output_directory()) / "LoRA_Test_Grids" / _safe_slug(run_id, "run")
    return legacy / "_cells" / f"p{scenario_index:03d}_l{candidate_index:03d}.png"


def _legacy_run_root(run_id: str) -> Path:
    return Path(folder_paths.get_output_directory()) / "LoRA_Test_Grids" / _safe_slug(run_id, "run")


def _completed_keys(run: dict) -> set[str]:
    complete = set()
    for p in range(run["scenario_count"]):
        for l in range(run["candidate_count"]):
            if _cell_path(run["run_id"], p, l).exists():
                complete.add(_job_key(p, l))
    return complete


def _queue_depth(payload: dict) -> int:
    return len(payload.get("queue_running") or []) + len(payload.get("queue_pending") or [])


def _owned_queue_ids(queue: dict, run_id: str) -> tuple[set[str], set[str]]:
    running_ids = set()
    pending_ids = set()
    for key, destination in (("queue_running", running_ids), ("queue_pending", pending_ids)):
        for item in queue.get(key) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            extra_data = item[3] if isinstance(item[3], dict) else {}
            if str(extra_data.get("loralab_run_id") or "") == run_id:
                destination.add(str(item[1]))
    return running_ids, pending_ids


def _request_resource_cleanup() -> bool:
    analyzer_released = False
    try:
        from . import release_analyzer_resources
        release_analyzer_resources()
        analyzer_released = True
    except Exception:
        pass
    if PromptServer is None:
        return analyzer_released
    try:
        queue = PromptServer.instance.prompt_queue
        queue.set_flag("unload_models", True)
        queue.set_flag("free_memory", True)
        return True
    except Exception:
        return analyzer_released


async def _stop_run_queue(run_id: str, base_url: str) -> dict:
    task = _QUEUE_TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    run = _read_run(run_id)
    known_ids = {str(item) for item in run.get("submitted_prompt_ids", []) if item}
    running_ids = set()
    pending_ids = set()
    cancel_error = None
    if ClientSession is not None:
        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            async def post_json(path: str, payload: dict) -> int:
                async with session.post(f"{base_url}{path}", json=payload) as response:
                    await response.read()
                    return response.status

            try:
                async with session.get(f"{base_url}/queue") as response:
                    queue = await response.json()
                running_ids, pending_ids = _owned_queue_ids(queue, run_id)
                all_ids = sorted(known_ids | running_ids | pending_ids)
                if all_ids:
                    status = await post_json("/api/jobs/cancel", {"job_ids": all_ids})
                    if status >= 400:
                        raise RuntimeError(f"Batch cancellation returned HTTP {status}")
            except Exception as exc:
                cancel_error = f"{type(exc).__name__}: {exc}"
                if pending_ids:
                    with contextlib.suppress(Exception):
                        await post_json("/queue", {"delete": sorted(pending_ids)})
                for prompt_id in running_ids:
                    with contextlib.suppress(Exception):
                        await post_json("/interrupt", {"prompt_id": prompt_id})

            remaining_running = set(running_ids)
            remaining_pending = set(pending_ids)
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.25)
                try:
                    async with session.get(f"{base_url}/queue") as response:
                        queue = await response.json()
                    remaining_running, remaining_pending = _owned_queue_ids(queue, run_id)
                except Exception:
                    break
                if not remaining_running and not remaining_pending:
                    break
                if remaining_pending:
                    with contextlib.suppress(Exception):
                        await post_json("/queue", {"delete": sorted(remaining_pending)})
                for prompt_id in remaining_running:
                    with contextlib.suppress(Exception):
                        await post_json("/interrupt", {"prompt_id": prompt_id})
            with contextlib.suppress(Exception):
                await post_json("/free", {"unload_models": True, "free_memory": True})
    else:
        remaining_running = running_ids
        remaining_pending = pending_ids

    cleanup_requested = _request_resource_cleanup()
    return {
        "cancelled_prompt_ids": sorted(known_ids | running_ids | pending_ids),
        "remaining_prompt_ids": sorted(remaining_running | remaining_pending),
        "cleanup_requested": cleanup_requested,
        "cancel_error": cancel_error,
    }


async def _queue_run(run_id: str, base_url: str, client_id: str | None = None) -> None:
    timeout = ClientTimeout(total=60) if ClientTimeout else None
    try:
        async with ClientSession(timeout=timeout) as session:
            while True:
                with _RUN_LOCK:
                    run = _read_run(run_id)
                if run["status"] in {"paused", "cancelled"}:
                    break
                completed = _completed_keys(run)
                jobs = []
                for scenario in run["scenarios"]:
                    p = int(scenario["scenario_index"])
                    for l, candidate in enumerate(run["candidates"]):
                        key = _job_key(p, l)
                        if key in completed or key in run.get("submitted_jobs", {}):
                            continue
                        jobs.append((key, scenario, candidate, l))
                if not jobs:
                    with _RUN_LOCK:
                        run = _read_run(run_id)
                        if len(_completed_keys(run)) >= run["expected_cells"]:
                            run["status"] = "complete"
                        elif run["status"] not in {"paused", "cancelled"}:
                            run["status"] = "running"
                        _write_run(run)
                    break
                try:
                    async with session.get(f"{base_url}/queue") as response:
                        queue = await response.json()
                    if _queue_depth(queue) >= 3:
                        await asyncio.sleep(0.8)
                        continue
                except Exception:
                    await asyncio.sleep(0.8)
                key, scenario, candidate, candidate_index = jobs[0]
                prompt = _job_prompt(run, scenario, candidate, candidate_index)
                body = {
                    "prompt": prompt,
                    "client_id": client_id,
                    "extra_data": {
                        "loralab_run_id": run_id,
                        "loralab_job_key": key,
                    },
                }
                try:
                    async with session.post(f"{base_url}/prompt", json=body) as response:
                        result = await response.json()
                        if response.status >= 400:
                            raise RuntimeError(result.get("error") or f"HTTP {response.status}")
                    with _RUN_LOCK:
                        run = _read_run(run_id)
                        prompt_id = str(result["prompt_id"])
                        run.setdefault("submitted_prompt_ids", []).append(prompt_id)
                        run.setdefault("submitted_jobs", {})[key] = prompt_id
                        run["status"] = "queueing"
                        _write_run(run)
                    if PromptServer is not None:
                        PromptServer.instance.send_sync("loralab.progress", {"run_id": run_id, "submitted": len(run["submitted_jobs"]), "total": run["expected_cells"]})
                except Exception as exc:
                    with _RUN_LOCK:
                        run = _read_run(run_id)
                        run.setdefault("queue_errors", []).append({"job": key, "error": f"{type(exc).__name__}: {exc}", "time": _now_iso()})
                        run["queue_errors"] = run["queue_errors"][-50:]
                        run["status"] = "queue_error"
                        _write_run(run)
                    await asyncio.sleep(2.0)
                    break
    finally:
        _QUEUE_TASKS.pop(run_id, None)


def _start_queue_task(run_id: str, base_url: str, client_id: str | None) -> bool:
    existing = _QUEUE_TASKS.get(run_id)
    if existing is not None and not existing.done():
        return False
    task = asyncio.create_task(_queue_run(run_id, base_url.rstrip("/"), client_id))
    _QUEUE_TASKS[run_id] = task
    return True


def _run_progress(run: dict, include_cells: bool = True) -> dict:
    completed = _completed_keys(run)
    total = int(run["expected_cells"])
    if len(completed) >= total and run["status"] not in {"cancelled"}:
        run["status"] = "complete"
        if run.get("auto_cleanup") and not run.get("resources_released_at"):
            run["resources_released_at"] = _now_iso()
            run["resource_cleanup_requested"] = _request_resource_cleanup()
        _write_run(run)
    cells = []
    if include_cells:
        for p in range(run["scenario_count"]):
            for l in range(run["candidate_count"]):
                path = _cell_path(run["run_id"], p, l)
                if not path.exists():
                    continue
                cells.append({
                    "prompt_index": p,
                    "lora_index": l,
                    "key": _job_key(p, l),
                    "candidate": run["candidates"][l],
                    "scenario": run["scenarios"][p],
                    "asset_url": f"/loralab/v1/asset?run_id={run['run_id']}&path=_cells/{path.name}",
                })
    elapsed = max(0, time.time() - time.mktime(time.strptime(run["created_at"][:19], "%Y-%m-%dT%H:%M:%S"))) if run.get("created_at") else 0
    remaining = None
    if completed and len(completed) < total:
        remaining = int(elapsed / len(completed) * (total - len(completed)))
    return {
        "completed": len(completed),
        "submitted": len(run.get("submitted_jobs", {})),
        "total": total,
        "percent": round(100.0 * len(completed) / max(1, total), 1),
        "status": run["status"],
        "remaining_seconds": remaining,
        "cells": cells,
    }


def _list_runs(limit: int = 30) -> list[dict]:
    root = Path(folder_paths.get_output_directory()) / RUN_FOLDER
    if not root.exists():
        return []
    rows = []
    files = sorted(root.glob("*/LAB_RUN.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:limit]:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            progress = _run_progress(run, include_cells=False)
            analysis_path = path.parent / "LAB_ANALYSIS.json"
            winner = None
            decisive = None
            if analysis_path.exists():
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                winner = analysis.get("winner")
                decisive = analysis.get("decisive")
            rows.append({
                "run_id": run["run_id"],
                "name": run.get("name"),
                "created_at": run.get("created_at"),
                "profile": run.get("profile", {}).get("name"),
                "mode": run.get("mode"),
                "candidate_count": run.get("candidate_count"),
                "scenario_count": run.get("scenario_count"),
                "progress": progress,
                "winner": winner,
                "decisive": decisive,
            })
        except Exception:
            continue
    return rows


def _ratings_path(run_id: str) -> Path:
    return _run_root(run_id) / "LAB_RATINGS.json"


def _read_ratings(run_id: str) -> dict:
    path = _ratings_path(run_id)
    if not path.exists():
        return {"schema": 1, "run_id": run_id, "ratings": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _watchers_path() -> Path:
    return Path(folder_paths.get_output_directory()) / RUN_FOLDER / "LAB_WATCHERS.json"


def _read_watchers() -> dict:
    path = _watchers_path()
    if not path.exists():
        return {"schema": 1, "watchers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("watchers"), dict):
            data["watchers"] = {}
        return data
    except Exception:
        return {"schema": 1, "watchers": {}}


def _write_watchers(data: dict) -> None:
    _atomic_json(_watchers_path(), data)


def _watcher_public(watcher: dict) -> dict:
    return {
        "watcher_id": watcher["watcher_id"],
        "group": watcher["group"],
        "active": bool(watcher.get("active")),
        "interval_seconds": int(watcher.get("interval_seconds", 15)),
        "known_count": len(watcher.get("known_files") or []),
        "observed_count": len(watcher.get("observed") or {}),
        "run_ids": list(watcher.get("run_ids") or []),
        "last_scan": watcher.get("last_scan"),
        "last_new_file": watcher.get("last_new_file"),
        "errors": list(watcher.get("errors") or [])[-5:],
    }


async def _watch_loop(watcher_id: str, base_url: str) -> None:
    try:
        while True:
            data = _read_watchers()
            watcher = data["watchers"].get(watcher_id)
            if not watcher or not watcher.get("active"):
                break
            watcher["last_scan"] = _now_iso()
            known = set(watcher.get("known_files") or [])
            observed = watcher.setdefault("observed", {})
            ready = []
            for item in _lora_catalog():
                filename = item["filename"]
                if item["group"] != watcher["group"] or filename in known:
                    continue
                path = folder_paths.get_full_path("loras", filename)
                if not path or not Path(path).is_file():
                    continue
                signature = f"{Path(path).stat().st_size}:{Path(path).stat().st_mtime_ns}"
                if observed.get(filename) == signature:
                    ready.append(item)
                else:
                    observed[filename] = signature
            ready.sort(key=lambda item: (item.get("step") is None, item.get("step") or 0, item["filename"]))
            _write_watchers(data)
            for item in ready:
                data = _read_watchers()
                watcher = data["watchers"].get(watcher_id)
                if not watcher or not watcher.get("active"):
                    return
                filename = item["filename"]
                if filename in watcher.get("known_files", []):
                    continue
                payload = json.loads(json.dumps(watcher["template"]))
                payload.update({
                    "mode": "compare",
                    "objective": "best_checkpoint",
                    "selected_loras": [filename],
                    "include_baseline": True,
                    "name": f"Watch · {watcher['group']} · {_lora_label(filename)}",
                    "output_prefix": f"Watch_{_safe_slug(watcher['group'])}",
                })
                try:
                    run = _make_plan(payload)
                    watcher.setdefault("known_files", []).append(filename)
                    watcher.setdefault("run_ids", []).append(run["run_id"])
                    watcher["last_new_file"] = filename
                    watcher.get("observed", {}).pop(filename, None)
                    _write_watchers(data)
                    _start_queue_task(run["run_id"], base_url, None)
                except Exception as exc:
                    watcher.setdefault("errors", []).append({"at": _now_iso(), "file": filename, "error": f"{type(exc).__name__}: {exc}"})
                    if filename not in watcher.setdefault("known_files", []):
                        watcher["known_files"].append(filename)
                    _write_watchers(data)
            await asyncio.sleep(max(10, min(300, int(watcher.get("interval_seconds", 15)))))
    except asyncio.CancelledError:
        pass
    finally:
        _WATCH_TASKS.pop(watcher_id, None)


def _start_watch_task(watcher_id: str, base_url: str) -> bool:
    task = _WATCH_TASKS.get(watcher_id)
    if task and not task.done():
        return False
    _WATCH_TASKS[watcher_id] = asyncio.create_task(_watch_loop(watcher_id, base_url))
    return True


def _resume_watch_tasks(base_url: str) -> None:
    for watcher_id, watcher in _read_watchers().get("watchers", {}).items():
        if watcher.get("active"):
            _start_watch_task(watcher_id, base_url)


def _tournament_path(run_id: str) -> Path:
    return _run_root(run_id) / "LAB_TOURNAMENT.json"


def _read_tournament(run_id: str) -> dict | None:
    path = _tournament_path(run_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_tournament_round(scenario: dict) -> None:
    contenders = list(scenario.get("contenders") or [])
    scenario["pairs"] = [[contenders[i], contenders[i + 1]] for i in range(0, len(contenders) - 1, 2)]
    scenario["pair_index"] = 0
    scenario["advancing"] = [contenders[-1]] if len(contenders) % 2 else []


def _advance_tournament(tournament: dict) -> None:
    scenarios = tournament["scenarios"]
    while tournament["scenario_cursor"] < len(scenarios):
        scenario = scenarios[tournament["scenario_cursor"]]
        if scenario.get("champion") is not None:
            tournament["scenario_cursor"] += 1
            continue
        if scenario["pair_index"] < len(scenario["pairs"]):
            return
        contenders = list(scenario["advancing"])
        if len(contenders) == 1:
            scenario["champion"] = contenders[0]
            tournament["scenario_cursor"] += 1
            continue
        scenario["round"] += 1
        scenario["contenders"] = contenders
        _prepare_tournament_round(scenario)
    tournament["status"] = "complete"
    tournament["completed_at"] = _now_iso()


def _analysis_path(run_id: str) -> Path:
    return _run_root(run_id) / "LAB_ANALYSIS.json"


def _load_analysis(run_id: str) -> dict:
    path = _analysis_path(run_id)
    if not path.exists():
        raise RuntimeError("Run automatic analysis before starting blind tournament.")
    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_similarity(analysis: dict, scenario_index: int, candidate_index: int) -> float:
    for entry in analysis.get("entries") or []:
        if int(entry.get("prompt_index", -1)) == scenario_index and int(entry.get("lora_index", -1)) == candidate_index:
            return float(entry.get("identity_similarity") or 0.0)
    return 0.0


def _analysis_metric(analysis: dict, scenario_index: int, candidate_index: int, key: str) -> float | None:
    for entry in analysis.get("entries") or []:
        if int(entry.get("prompt_index", -1)) == scenario_index and int(entry.get("lora_index", -1)) == candidate_index:
            value = entry.get(key)
            return float(value) if value is not None else None
    return None


def _start_tournament(
    run_id: str,
    include_baseline: bool = True,
    human_weight: float = 0.5,
    candidate_indices: list[int] | None = None,
    round_number: int = 1,
    stage: str = "initial",
    previous_rounds: list[dict] | None = None,
) -> dict:
    run = _read_run(run_id)
    if len(_completed_keys(run)) < run["expected_cells"]:
        raise RuntimeError("Run must be complete before blind tournament.")
    _load_analysis(run_id)
    if candidate_indices is None:
        candidate_indices = [
            index for index, candidate in enumerate(run["candidates"])
            if include_baseline or not candidate.get("baseline")
        ]
    else:
        candidate_indices = list(dict.fromkeys(int(index) for index in candidate_indices))
        if any(index < 0 or index >= len(run["candidates"]) for index in candidate_indices):
            raise ValueError("Blind tournament contains a candidate outside this run.")
    if len(candidate_indices) < 2:
        raise ValueError("Blind tournament needs at least two candidates. Include baseline when testing one LoRA.")
    scenarios = []
    round_number = max(1, int(round_number))
    stable_seed = (
        sum((index + 1) * ord(char) for index, char in enumerate(run_id))
        + round_number * 104729
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(stable_seed)
    base_order = [int(value) for value in rng.permutation(candidate_indices)]
    for scenario_index in range(run["scenario_count"]):
        shift = scenario_index % len(base_order)
        contenders = base_order[shift:] + base_order[:shift]
        if scenario_index % 2:
            contenders = [value for i in range(0, len(contenders), 2) for value in reversed(contenders[i:i + 2])]
        scenario = {
            "scenario_index": scenario_index,
            "round": 1,
            "contenders": contenders,
            "pairs": [],
            "pair_index": 0,
            "advancing": [],
            "champion": None,
        }
        _prepare_tournament_round(scenario)
        scenarios.append(scenario)
    tournament = {
        "schema": 2,
        "run_id": run_id,
        "status": "active",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "round_number": round_number,
        "stage": stage if stage in {"initial", "runoff"} else "initial",
        "previous_rounds": list(previous_rounds or []),
        "include_baseline": bool(include_baseline),
        "human_weight": max(0.0, min(1.0, float(human_weight))),
        "candidate_indices": candidate_indices,
        "scenario_cursor": 0,
        "scenarios": scenarios,
        "comparisons": [],
    }
    _advance_tournament(tournament)
    _atomic_json(_tournament_path(run_id), tournament)
    return tournament


def _tournament_next_match(tournament: dict, run: dict) -> dict | None:
    if tournament.get("status") != "active":
        return None
    _advance_tournament(tournament)
    if tournament.get("status") != "active":
        return None
    scenario_state = tournament["scenarios"][tournament["scenario_cursor"]]
    left, right = scenario_state["pairs"][scenario_state["pair_index"]]
    scenario_index = int(scenario_state["scenario_index"])
    scenario = run["scenarios"][scenario_index]
    return {
        "match_id": f"t{int(tournament.get('round_number', 1)):02d}_s{scenario_index:03d}_r{scenario_state['round']:02d}_m{scenario_state['pair_index']:02d}",
        "scenario_index": scenario_index,
        "scenario_number": scenario_index + 1,
        "scenario_total": run["scenario_count"],
        "round": scenario_state["round"],
        "prompt_label": scenario["label"],
        "prompt_text": scenario["text"],
        "seed": scenario["seed"],
        "left": {"asset_url": f"/loralab/v1/asset?run_id={run['run_id']}&path=_cells/{_cell_path(run['run_id'], scenario_index, left).name}"},
        "right": {"asset_url": f"/loralab/v1/asset?run_id={run['run_id']}&path=_cells/{_cell_path(run['run_id'], scenario_index, right).name}"},
    }


def _normalized_scores(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if abs(high - low) < 1e-12:
        return {key: 50.0 for key in values}
    return {key: (value - low) / (high - low) * 100.0 for key, value in values.items()}


def _winner_label(rows: list[dict], score_key: str, tolerance: float = 1e-9) -> str:
    if not rows:
        return "No candidate"
    best = max(float(row.get(score_key) or 0.0) for row in rows)
    winners = [row["candidate"]["label"] for row in rows if abs(float(row.get(score_key) or 0.0) - best) <= tolerance]
    return winners[0] if len(winners) == 1 else f"Tie: {', '.join(winners)}"


def _bradley_terry_scores(candidate_indices: list[int], comparisons: list[dict], choice_key: str = "human_choice") -> dict[int, float]:
    positions = {candidate: index for index, candidate in enumerate(candidate_indices)}
    count = len(candidate_indices)
    wins = np.zeros(count, dtype=np.float64)
    games = np.zeros((count, count), dtype=np.float64)
    for comparison in comparisons:
        choice = comparison.get(choice_key)
        if not choice or choice == "skip":
            continue
        left, right = positions[int(comparison["left"])], positions[int(comparison["right"])]
        effective = "right" if choice == "left_broken" else "left" if choice == "right_broken" else choice
        score_left = 1.0 if effective == "left" else 0.0 if effective == "right" else 0.5
        wins[left] += score_left
        wins[right] += 1.0 - score_left
        games[left, right] += 1.0
        games[right, left] += 1.0
    ability = np.ones(count, dtype=np.float64)
    for _ in range(100):
        updated = np.zeros(count, dtype=np.float64)
        for i in range(count):
            denominator = sum(games[i, j] / max(1e-12, ability[i] + ability[j]) for j in range(count) if j != i)
            updated[i] = wins[i] / denominator if denominator > 0 and wins[i] > 0 else 1e-6
        updated /= max(1e-12, float(np.mean(updated)))
        if float(np.max(np.abs(updated - ability))) < 1e-8:
            ability = updated
            break
        ability = updated
    normalized = _normalized_scores({candidate: float(ability[index]) for candidate, index in positions.items()})
    return normalized


def _tournament_public(tournament: dict | None, run: dict) -> dict | None:
    if tournament is None:
        return None
    _advance_tournament(tournament)
    candidate_count = len(tournament["candidate_indices"])
    expected = run["scenario_count"] * max(0, candidate_count - 1)
    result = {
        "status": tournament["status"],
        "round_number": int(tournament.get("round_number", 1)),
        "stage": tournament.get("stage", "initial"),
        "candidate_indices": list(tournament["candidate_indices"]),
        "include_baseline": tournament.get("include_baseline", True),
        "human_weight": tournament.get("human_weight", 0.5),
        "completed": len(tournament["comparisons"]),
        "total": expected,
        "percent": round(100.0 * len(tournament["comparisons"]) / max(1, expected), 1),
        "next_match": _tournament_next_match(tournament, run),
        "can_undo": bool(tournament["comparisons"]),
        "previous_rounds": [
            {
                "round_number": int(item.get("round_number", index + 1)),
                "stage": item.get("stage", "initial"),
                "completed_at": item.get("completed_at"),
                "candidate_count": len(item.get("candidate_indices") or []),
                "comparison_count": len(item.get("comparisons") or []),
                "human_winner": item.get("human_winner", "No candidate"),
                "combined_winner": item.get("combined_winner", "No candidate"),
                "candidate_labels": [
                    run["candidates"][int(candidate_index)]["label"]
                    for candidate_index in item.get("candidate_indices") or []
                    if 0 <= int(candidate_index) < len(run["candidates"])
                ],
            }
            for index, item in enumerate(tournament.get("previous_rounds") or [])
        ],
    }
    if tournament.get("status") != "complete":
        return result

    ratings = {index: 1500.0 for index in tournament["candidate_indices"]}
    overall_ratings = {index: 1500.0 for index in tournament["candidate_indices"]}
    overall_vote_count = 0
    agreement_count = 0
    agreement_total = 0
    category_agreement = defaultdict(lambda: {"same": 0, "total": 0})
    side_counts = {index: {"left": 0, "right": 0} for index in tournament["candidate_indices"]}
    issue_counts = {index: {"artifact": 0, "identity_failure": 0} for index in tournament["candidate_indices"]}
    analyzer_agreement = {"ensemble": {"same": 0, "total": 0}, "kprpe": {"same": 0, "total": 0}, "antelopev2": {"same": 0, "total": 0}}
    for comparison in tournament["comparisons"]:
        left, right = int(comparison["left"]), int(comparison["right"])
        side_counts[left]["left"] += 1
        side_counts[right]["right"] += 1
        issue_counts[left]["artifact"] += int(bool(comparison.get("artifact_left")))
        issue_counts[right]["artifact"] += int(bool(comparison.get("artifact_right")))
        issue_counts[left]["identity_failure"] += int(bool(comparison.get("identity_failure_left")))
        issue_counts[right]["identity_failure"] += int(bool(comparison.get("identity_failure_right")))
        human_choice = comparison["human_choice"]
        if human_choice == "skip":
            continue
        expected_left = 1.0 / (1.0 + 10.0 ** ((ratings[right] - ratings[left]) / 400.0))
        effective_choice = "right" if human_choice == "left_broken" else "left" if human_choice == "right_broken" else human_choice
        actual_left = 1.0 if effective_choice == "left" else 0.0 if effective_choice == "right" else 0.5
        change = 24.0 * (actual_left - expected_left)
        ratings[left] += change
        ratings[right] -= change
        overall_choice = comparison.get("overall_choice")
        if overall_choice and overall_choice != "skip":
            overall_vote_count += 1
            overall_expected_left = 1.0 / (1.0 + 10.0 ** ((overall_ratings[right] - overall_ratings[left]) / 400.0))
            overall_actual_left = 1.0 if overall_choice == "left" else 0.0 if overall_choice == "right" else 0.5
            overall_change = 24.0 * (overall_actual_left - overall_expected_left)
            overall_ratings[left] += overall_change
            overall_ratings[right] -= overall_change
        if comparison.get("agreement") is not None:
            agreement_total += 1
            agreement_count += int(bool(comparison.get("agreement")))
            category = run["scenarios"][int(comparison["scenario_index"])]["category"]
            category_agreement[category]["total"] += 1
            category_agreement[category]["same"] += int(bool(comparison.get("agreement")))
        if int(comparison.get("vote_schema") or 1) >= 2 and human_choice in {"left", "right", "tie"}:
            model_choices = {"ensemble": comparison.get("ai_choice")}
            for name, left_key, right_key in (
                ("kprpe", "kprpe_left_score", "kprpe_right_score"),
                ("antelopev2", "antelope_left_score", "antelope_right_score"),
            ):
                left_value, right_value = comparison.get(left_key), comparison.get(right_key)
                if left_value is None or right_value is None:
                    continue
                difference = float(left_value) - float(right_value)
                model_choices[name] = "tie" if abs(difference) <= 0.015 else "left" if difference > 0 else "right"
            for name, model_choice in model_choices.items():
                if model_choice not in {"left", "right", "tie"}:
                    continue
                analyzer_agreement[name]["total"] += 1
                analyzer_agreement[name]["same"] += int(model_choice == human_choice)
    human_scores = _normalized_scores(ratings)
    overall_scores = _normalized_scores(overall_ratings) if overall_vote_count else {}
    bt_scores = _bradley_terry_scores(tournament["candidate_indices"], tournament["comparisons"])
    overall_bt_scores = _bradley_terry_scores(tournament["candidate_indices"], tournament["comparisons"], "overall_choice")
    analysis = _load_analysis(run["run_id"])
    automatic_raw = {}
    all_rank_rows = list(analysis.get("ranking") or [])
    if analysis.get("baseline"):
        all_rank_rows.append(analysis["baseline"])
    for row in all_rank_rows:
        index = int(row["candidate_index"])
        if index in ratings:
            automatic_raw[index] = float(row.get("automatic_score") or 0.0)
    automatic_scores = _normalized_scores(automatic_raw)
    human_weight = float(tournament.get("human_weight", 0.5))
    champion_counts = defaultdict(int)
    for scenario in tournament["scenarios"]:
        if scenario.get("champion") is not None:
            champion_counts[int(scenario["champion"])] += 1
    standings = []
    for index in tournament["candidate_indices"]:
        automatic = automatic_scores.get(index, 0.0)
        human = human_scores.get(index, 0.0)
        standings.append({
            "candidate_index": index,
            "candidate": run["candidates"][index],
            "human_elo": ratings[index],
            "human_score": human,
            "bradley_terry_score": bt_scores.get(index, 0.0),
            "overall_elo": overall_ratings[index] if overall_vote_count else None,
            "overall_score": overall_scores.get(index) if overall_vote_count else None,
            "overall_bradley_terry_score": overall_bt_scores.get(index) if overall_vote_count else None,
            "automatic_score": automatic,
            "combined_score": human_weight * human + (1.0 - human_weight) * automatic,
            "scenario_wins": champion_counts[index],
            "left_count": side_counts[index]["left"],
            "right_count": side_counts[index]["right"],
            "artifact_count": issue_counts[index]["artifact"],
            "identity_failure_count": issue_counts[index]["identity_failure"],
        })
    standings.sort(key=lambda row: (row["combined_score"], row["human_elo"]), reverse=True)
    for rank, row in enumerate(standings, 1):
        row["rank"] = rank
    result.update({
        "agreement_count": agreement_count,
        "agreement_total": agreement_total,
        "agreement_rate": agreement_count / max(1, agreement_total),
        "category_agreement": [{"category": category, **values, "rate": values["same"] / max(1, values["total"])} for category, values in sorted(category_agreement.items())],
        "analyzer_agreement": {name: {**values, "rate": values["same"] / max(1, values["total"])} for name, values in analyzer_agreement.items()},
        "standings": standings,
        "human_winner": _winner_label(standings, "human_elo", tolerance=0.01),
        "overall_winner": _winner_label(standings, "overall_elo", tolerance=0.01) if overall_vote_count else "No overall votes",
        "overall_vote_count": overall_vote_count,
        "automatic_winner": _winner_label(standings, "automatic_score", tolerance=0.01),
        "combined_winner": _winner_label(standings, "combined_score", tolerance=0.01),
    })
    return result


def _archive_tournament_round(tournament: dict, run: dict) -> dict:
    public = _tournament_public(tournament, run)
    if not public or public.get("status") != "complete":
        raise RuntimeError("Complete the current blind tournament before starting a runoff.")
    return {
        "round_number": int(tournament.get("round_number", 1)),
        "stage": tournament.get("stage", "initial"),
        "created_at": tournament.get("created_at"),
        "completed_at": tournament.get("completed_at"),
        "candidate_indices": list(tournament.get("candidate_indices") or []),
        "comparisons": list(tournament.get("comparisons") or []),
        "scenarios": list(tournament.get("scenarios") or []),
        "human_winner": public.get("human_winner"),
        "automatic_winner": public.get("automatic_winner"),
        "combined_winner": public.get("combined_winner"),
        "standings": list(public.get("standings") or []),
    }


def _start_tournament_runoff(run_id: str, finalist_count: int = 3) -> dict:
    run = _read_run(run_id)
    existing = _read_tournament(run_id)
    if not existing or existing.get("status") != "complete":
        raise RuntimeError("Complete the current blind tournament before starting a runoff.")
    public = _tournament_public(existing, run)
    eligible = [
        row for row in public.get("standings") or []
        if not row.get("candidate", {}).get("baseline")
    ]
    if len(eligible) < 2:
        raise RuntimeError("A finalist runoff needs at least two non-baseline checkpoints.")
    finalist_count = max(2, min(int(finalist_count), min(4, len(eligible))))
    finalists = [int(row["candidate_index"]) for row in eligible[:finalist_count]]
    previous_rounds = list(existing.get("previous_rounds") or [])
    previous_rounds.append(_archive_tournament_round(existing, run))
    return _start_tournament(
        run_id,
        include_baseline=False,
        human_weight=float(existing.get("human_weight", 0.5)),
        candidate_indices=finalists,
        round_number=int(existing.get("round_number", 1)) + 1,
        stage="runoff",
        previous_rounds=previous_rounds,
    )


def _vote_tournament(run_id: str, choice: str, match_id: str, flags: dict | None = None) -> dict:
    if choice not in {"left", "right", "tie", "left_broken", "right_broken", "skip"}:
        raise ValueError("Tournament choice must be left, right, tie, broken-side, or skip.")
    flags = flags or {}
    overall_choice = flags.get("overall_choice")
    if overall_choice not in {None, "", "left", "right", "tie", "skip"}:
        raise ValueError("Overall preference must be left, right, tie, skip, or empty.")
    run = _read_run(run_id)
    tournament = _read_tournament(run_id)
    if not tournament or tournament.get("status") != "active":
        raise RuntimeError("No active blind tournament.")
    match = _tournament_next_match(tournament, run)
    if not match or match["match_id"] != match_id:
        raise RuntimeError("Match changed. Reload and vote on current pair.")
    scenario_state = tournament["scenarios"][tournament["scenario_cursor"]]
    left, right = scenario_state["pairs"][scenario_state["pair_index"]]
    analysis = _load_analysis(run_id)
    scenario_index = int(scenario_state["scenario_index"])
    left_score = _analysis_similarity(analysis, scenario_index, left)
    right_score = _analysis_similarity(analysis, scenario_index, right)
    kprpe_left = _analysis_metric(analysis, scenario_index, left, "kprpe_similarity")
    kprpe_right = _analysis_metric(analysis, scenario_index, right, "kprpe_similarity")
    antelope_left = _analysis_metric(analysis, scenario_index, left, "antelope_similarity")
    antelope_right = _analysis_metric(analysis, scenario_index, right, "antelope_similarity")
    if abs(left_score - right_score) <= 0.015:
        ai_choice = "tie"
    else:
        ai_choice = "left" if left_score > right_score else "right"
    if choice == "left":
        advancing = left
    elif choice == "right":
        advancing = right
    elif choice == "left_broken":
        advancing = right
    elif choice == "right_broken":
        advancing = left
    elif ai_choice == "left":
        advancing = left
    elif ai_choice == "right":
        advancing = right
    else:
        advancing = min(left, right)
    comparison = {
        "vote_schema": 2,
        "human_criterion": "identity",
        "match_id": match_id,
        "scenario_index": scenario_index,
        "round": scenario_state["round"],
        "left": left,
        "right": right,
        "human_choice": choice,
        "overall_choice": overall_choice or None,
        "ai_choice": ai_choice,
        "ai_left_score": left_score,
        "ai_right_score": right_score,
        "kprpe_left_score": kprpe_left,
        "kprpe_right_score": kprpe_right,
        "antelope_left_score": antelope_left,
        "antelope_right_score": antelope_right,
        "agreement": choice == ai_choice if choice in {"left", "right", "tie"} else None,
        "artifact_left": bool(flags.get("artifact_left")),
        "artifact_right": bool(flags.get("artifact_right")),
        "identity_failure_left": bool(flags.get("identity_failure_left")),
        "identity_failure_right": bool(flags.get("identity_failure_right")),
        "voted_at": _now_iso(),
    }
    tournament["comparisons"].append(comparison)
    scenario_state["advancing"].append(advancing)
    scenario_state["pair_index"] += 1
    tournament["updated_at"] = _now_iso()
    _advance_tournament(tournament)
    _atomic_json(_tournament_path(run_id), tournament)
    return tournament


def _undo_tournament(run_id: str) -> dict:
    existing = _read_tournament(run_id)
    if not existing or not existing.get("comparisons"):
        raise RuntimeError("No blind vote to undo.")
    replay = list(existing["comparisons"][:-1])
    rebuilt = _start_tournament(
        run_id,
        bool(existing.get("include_baseline", True)),
        float(existing.get("human_weight", 0.5)),
        candidate_indices=list(existing.get("candidate_indices") or []),
        round_number=int(existing.get("round_number", 1)),
        stage=existing.get("stage", "initial"),
        previous_rounds=list(existing.get("previous_rounds") or []),
    )
    for comparison in replay:
        run = _read_run(run_id)
        current = _tournament_next_match(rebuilt, run)
        if not current:
            break
        rebuilt = _vote_tournament(
            run_id,
            comparison["human_choice"],
            current["match_id"],
            {
                "artifact_left": comparison.get("artifact_left"),
                "artifact_right": comparison.get("artifact_right"),
                "identity_failure_left": comparison.get("identity_failure_left"),
                "identity_failure_right": comparison.get("identity_failure_right"),
                "overall_choice": comparison.get("overall_choice"),
            },
        )
    rebuilt["undo_count"] = int(existing.get("undo_count", 0)) + 1
    rebuilt["updated_at"] = _now_iso()
    _atomic_json(_tournament_path(run_id), rebuilt)
    return rebuilt


def _quantile(values: list[float], q: float, default: float = 0.0) -> float:
    return float(np.quantile(values, q)) if values else default


def _bootstrap_ci(values: list[float], rng: np.random.Generator, iterations: int = 2500) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    samples = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _analyze_sync(run_id: str, reference_folder: str | None = None) -> dict:
    from . import _face_embedding, _get_face_app, _identity_similarity, _quality_metrics, _robust_reference_template, release_analyzer_resources
    from .cvlface_analyzer import get_analyzer as _get_cvlface, ready_details as _cvlface_ready, release_cuda as _release_cvlface, robust_template as _cvlface_template, similarity as _cvlface_similarity

    run = _read_run(run_id)
    if len(_completed_keys(run)) < run["expected_cells"]:
        raise RuntimeError("Run is incomplete. Retry missing cells before analysis.")
    reference_folder = reference_folder or run.get("reference_folder")
    template = _robust_reference_template(reference_folder)
    face_app = template.get("face_app") or _get_face_app()
    cvlface = None
    cvl_template = None
    cvlface_error = None
    cvl_ready, cvl_details = _cvlface_ready()
    if cvl_ready:
        try:
            cvlface = _get_cvlface()
            cvl_records = []
            for path in template.get("image_paths", []):
                record = cvlface.embedding(path, face_app)
                if record is not None:
                    cvl_records.append(record)
            cvl_template = _cvlface_template(cvl_records)
        except Exception as exc:
            cvlface_error = f"{type(exc).__name__}: {exc}"
            cvlface = None
            cvl_template = None
    else:
        cvlface_error = f"Model files unavailable: {', '.join(cvl_details.get('missing') or [])}"
    ratings = _read_ratings(run_id).get("ratings", {})
    entries = []
    by_candidate: dict[int, list[dict]] = defaultdict(list)
    try:
        for p, scenario in enumerate(run["scenarios"]):
            for l, candidate in enumerate(run["candidates"]):
                path = _cell_path(run_id, p, l)
                embedding = _face_embedding(path, face_app)
                antelope_similarity = _identity_similarity(embedding, template) if embedding is not None else None
                cvl_record = cvlface.embedding(path, face_app) if cvlface is not None else None
                cvl_metrics = _cvlface_similarity(cvl_record, cvl_template) if cvl_record is not None and cvl_template is not None else None
                quality = _quality_metrics(path)
                rating = ratings.get(_job_key(p, l), {})
                entry = {
                    "prompt_index": p,
                    "lora_index": l,
                    "key": _job_key(p, l),
                    "candidate": candidate,
                    "scenario": scenario,
                    "face_detected": antelope_similarity is not None or cvl_metrics is not None,
                    "identity_similarity": None,
                    "antelope_similarity": antelope_similarity,
                    "kprpe_similarity": cvl_metrics.get("similarity") if cvl_metrics else None,
                    "identity_confidence": cvl_metrics.get("quality") if cvl_metrics else None,
                    "kprpe_metrics": cvl_metrics,
                    "quality_metrics": quality,
                    "rating": rating,
                    "asset_url": f"/loralab/v1/asset?run_id={run_id}&path=_cells/{path.name}",
                }
                entries.append(entry)
                by_candidate[l].append(entry)
    finally:
        _release_cvlface()
        template.pop("face_app", None)
        face_app = None
        cvlface = None
        release_analyzer_resources()

    def calibrate(values: list[float | None]) -> tuple[list[float | None], dict]:
        finite = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
        if not len(finite):
            return [None for _ in values], {"median": None, "scale": None}
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        scale = max(0.01, 1.4826 * mad, float(np.std(finite)) * 0.35)
        calibrated = []
        for value in values:
            if value is None:
                calibrated.append(None)
                continue
            z_score = float(np.clip((float(value) - center) / scale, -6.0, 6.0))
            calibrated.append(float(1.0 / (1.0 + math.exp(-z_score))))
        return calibrated, {"median": center, "scale": scale}

    antelope_calibrated, antelope_calibration = calibrate([entry.get("antelope_similarity") for entry in entries])
    kprpe_calibrated, kprpe_calibration = calibrate([entry.get("kprpe_similarity") for entry in entries])
    for index, entry in enumerate(entries):
        antelope_value = antelope_calibrated[index]
        kprpe_value = kprpe_calibrated[index]
        entry["antelope_calibrated"] = antelope_value
        entry["kprpe_calibrated"] = kprpe_value
        if kprpe_value is not None and antelope_value is not None:
            entry["identity_similarity"] = 0.65 * kprpe_value + 0.35 * antelope_value
            entry["analyzer_coverage"] = "ensemble"
        elif kprpe_value is not None:
            entry["identity_similarity"] = kprpe_value
            entry["analyzer_coverage"] = "kprpe_only"
        elif antelope_value is not None:
            entry["identity_similarity"] = antelope_value
            entry["analyzer_coverage"] = "antelope_only"
        else:
            entry["identity_similarity"] = None
            entry["analyzer_coverage"] = "missing"

    rng = np.random.default_rng(20260710)
    baseline_index = next((i for i, item in enumerate(run["candidates"]) if item.get("baseline")), None)
    baseline_scores = {}
    if baseline_index is not None:
        for entry in by_candidate[baseline_index]:
            baseline_scores[entry["prompt_index"]] = float(entry["identity_similarity"] or 0.0)

    ranking = []
    bootstrap_means: dict[int, np.ndarray] = {}
    iterations = 3000
    for index, candidate in enumerate(run["candidates"]):
        group = by_candidate[index]
        scores = [float(entry["identity_similarity"] or 0.0) for entry in group]
        detected_scores = [float(entry["identity_similarity"]) for entry in group if entry["identity_similarity"] is not None]
        detected = len(detected_scores)
        antelope_scores = [float(entry["antelope_similarity"]) for entry in group if entry.get("antelope_similarity") is not None]
        kprpe_scores = [float(entry["kprpe_similarity"]) for entry in group if entry.get("kprpe_similarity") is not None]
        confidence_scores = [float(entry["identity_confidence"]) for entry in group if entry.get("identity_confidence") is not None]
        detection_rate = 100.0 * detected / max(1, len(group))
        mean_similarity = float(np.mean(scores)) if scores else 0.0
        median_similarity = float(np.median(scores)) if scores else 0.0
        floor_similarity = _quantile(scores, 0.20)
        standard_deviation = float(np.std(scores)) if scores else 1.0
        consistency = max(0.0, 100.0 - standard_deviation * 300.0)
        ci_low, ci_high = _bootstrap_ci(scores, rng)
        array = np.asarray(scores, dtype=np.float64)
        if len(array):
            bootstrap_means[index] = rng.choice(array, size=(iterations, len(array)), replace=True).mean(axis=1)
        gains = []
        if baseline_scores and not candidate.get("baseline"):
            gains = [float(entry["identity_similarity"] or 0.0) - baseline_scores.get(entry["prompt_index"], 0.0) for entry in group]
        category_scores = defaultdict(list)
        for entry in group:
            category_scores[entry["scenario"]["category"]].append(float(entry["identity_similarity"] or 0.0))
        rating_rows = [entry["rating"] for entry in group if entry.get("rating")]
        human_components = []
        artifact_count = 0
        for rating in rating_rows:
            values = []
            weights = []
            for key, weight in (("identity", 0.55), ("quality", 0.20), ("adherence", 0.25)):
                if rating.get(key) is not None:
                    values.append(float(rating[key]) * weight)
                    weights.append(weight)
            if weights:
                human_components.append(sum(values) / sum(weights) / 5.0 * 100.0)
            artifact_count += int(bool(rating.get("artifact")))
        human_score = float(np.mean(human_components)) if human_components else None
        if human_score is not None:
            human_score = max(0.0, human_score - 15.0 * artifact_count / max(1, len(rating_rows)))
        automatic_score = (
            0.60 * mean_similarity * 100.0
            + 0.25 * floor_similarity * 100.0
            + 0.10 * detection_rate
            + 0.05 * consistency
        )
        final_score = automatic_score
        if human_score is not None and len(human_components) >= max(3, math.ceil(len(group) * 0.5)):
            final_score = 0.55 * automatic_score + 0.45 * human_score
        ranking.append({
            "candidate_index": index,
            "candidate": candidate,
            "baseline": bool(candidate.get("baseline")),
            "automatic_score": automatic_score,
            "human_score": human_score,
            "human_rating_count": len(human_components),
            "final_score": final_score,
            "mean_similarity": mean_similarity,
            "mean_antelope_similarity": float(np.mean(antelope_scores)) if antelope_scores else None,
            "mean_kprpe_similarity": float(np.mean(kprpe_scores)) if kprpe_scores else None,
            "mean_identity_confidence": float(np.mean(confidence_scores)) if confidence_scores else None,
            "median_similarity": median_similarity,
            "floor_similarity": floor_similarity,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "detection_rate": detection_rate,
            "detected": detected,
            "missing_faces": len(group) - detected,
            "consistency": consistency,
            "identity_gain_vs_baseline": float(np.mean(gains)) if gains else None,
            "gain_ci": _bootstrap_ci(gains, rng) if gains else None,
            "category_scores": {key: float(np.mean(value)) for key, value in category_scores.items()},
        })

    real_ranking = [item for item in ranking if not item["baseline"]]
    real_ranking.sort(key=lambda item: item["final_score"], reverse=True)
    if real_ranking:
        candidate_indices = [item["candidate_index"] for item in real_ranking]
        stacked = np.stack([bootstrap_means[index] for index in candidate_indices], axis=1)
        winners = np.argmax(stacked, axis=1)
        for rank, item in enumerate(real_ranking, start=1):
            item["rank"] = rank
            item["probability_best"] = float(np.mean(winners == (rank - 1)))
    baseline = next((item for item in ranking if item["baseline"]), None)
    decisive = False
    paired_ci = None
    probability_top_beats_second = None
    if len(real_ranking) == 1:
        decisive = len(by_candidate[real_ranking[0]["candidate_index"]]) >= 6
    elif len(real_ranking) >= 2:
        first = real_ranking[0]["candidate_index"]
        second = real_ranking[1]["candidate_index"]
        first_scores = [float(entry["identity_similarity"] or 0.0) for entry in by_candidate[first]]
        second_scores = [float(entry["identity_similarity"] or 0.0) for entry in by_candidate[second]]
        paired = [a - b for a, b in zip(first_scores, second_scores)]
        paired_ci = _bootstrap_ci(paired, rng)
        probability_top_beats_second = float(np.mean(bootstrap_means[first] > bootstrap_means[second]))
        decisive = paired_ci[0] > 0 and probability_top_beats_second >= 0.90

    if not real_ranking:
        winner = "No candidate"
        confidence = "none"
    elif decisive:
        winner = real_ranking[0]["candidate"]["label"]
        confidence = "high" if real_ranking[0].get("probability_best", 0) >= 0.90 else "medium"
    else:
        winner = "No decisive winner"
        confidence = "low — finalists overlap"
    recommendation = "Use top candidate. Then run a strength sweep around its current value." if decisive else "Keep top 2–3 as finalists. Run Standard or Deep with more matched seeds; do blind ratings before choosing."
    model_agreements = []
    model_category_agreement = defaultdict(lambda: {"same": 0, "total": 0})
    if cvl_template is not None:
        for scenario_index, scenario in enumerate(run["scenarios"]):
            scenario_entries = [entry for entry in entries if entry["prompt_index"] == scenario_index and entry.get("antelope_similarity") is not None and entry.get("kprpe_similarity") is not None]
            if len(scenario_entries) < 2:
                continue
            antelope_winner = max(scenario_entries, key=lambda entry: float(entry["antelope_similarity"]))["lora_index"]
            kprpe_winner = max(scenario_entries, key=lambda entry: float(entry["kprpe_similarity"]))["lora_index"]
            same = antelope_winner == kprpe_winner
            model_agreements.append(same)
            bucket = model_category_agreement[scenario["category"]]
            bucket["total"] += 1
            bucket["same"] += int(same)
    analysis = {
        "schema": 1,
        "lab_version": LAB_VERSION,
        "run_id": run_id,
        "created_at": _now_iso(),
        "analyzer": "CVLFace ViT-KP-RPE AdaFace WebFace12M + InsightFace AntelopeV2" if cvl_template is not None else "InsightFace AntelopeV2 (CVLFace unavailable)",
        "analyzer_status": {
            "primary": "CVLFace ViT-KP-RPE AdaFace WebFace12M",
            "secondary": "InsightFace AntelopeV2",
            "ensemble_weights": {"kprpe": 0.65, "antelopev2": 0.35} if cvl_template is not None else {"antelopev2": 1.0},
            "quality_affects_identity_rank": False,
            "cvlface_error": cvlface_error,
            "calibration": {"kprpe": kprpe_calibration, "antelopev2": antelope_calibration},
            "model_agreement_rate": float(np.mean(model_agreements)) if model_agreements else None,
            "model_agreement_count": int(np.sum(model_agreements)) if model_agreements else 0,
            "model_agreement_total": len(model_agreements),
            "category_agreement": [{"category": key, **value, "rate": value["same"] / max(1, value["total"])} for key, value in sorted(model_category_agreement.items())],
        },
        "reference_folder": reference_folder,
        "reference": {
            "total": template["total"],
            "detected": template["detected"],
            "inliers": template["inliers"],
            "rejected": template["rejected"],
            "cohesion": template["cohesion"],
            "mean_quality": template.get("mean_quality"),
            "kprpe": {
                "detected": cvl_template["detected"],
                "inliers": cvl_template["inliers"],
                "rejected": cvl_template["rejected"],
                "cohesion": cvl_template["cohesion"],
                "mean_quality": cvl_template["mean_quality"],
            } if cvl_template is not None else None,
        },
        "winner": winner,
        "decisive": decisive,
        "confidence": confidence,
        "recommendation": recommendation,
        "paired_top_two_ci": paired_ci,
        "probability_top_beats_second": probability_top_beats_second,
        "ranking": real_ranking,
        "baseline": baseline,
        "entries": entries,
        "metric_notes": {
            "identity": "Robustly calibrated 65% KP-RPE + 35% AntelopeV2 identity ensemble. Compare only within the same run.",
            "missing_face": "Missing face scores as zero and is also reported separately.",
            "quality": "Technical metrics are displayed but never allowed to overturn identity ranking.",
            "confidence": "Paired bootstrap over matched prompt/seed scenarios. Overlapping finalists produce no forced winner.",
        },
    }
    _atomic_json(_run_root(run_id) / "LAB_ANALYSIS.json", analysis)
    csv_path = _run_root(run_id) / "LAB_RANKING.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "candidate", "step", "strength", "final_score", "automatic_score", "human_score", "mean_ensemble_identity", "mean_kprpe_cosine", "mean_antelope_cosine", "mean_face_confidence", "identity_ci_low", "identity_ci_high", "identity_gain_vs_baseline", "detection_rate", "missing_faces", "probability_best"])
        for item in real_ranking:
            candidate = item["candidate"]
            writer.writerow([item["rank"], candidate["filename"], candidate.get("step"), candidate["strength"], item["final_score"], item["automatic_score"], item.get("human_score"), item["mean_similarity"], item.get("mean_kprpe_similarity"), item.get("mean_antelope_similarity"), item.get("mean_identity_confidence"), item["ci_low"], item["ci_high"], item.get("identity_gain_vs_baseline"), item["detection_rate"], item["missing_faces"], item.get("probability_best")])
    run["status"] = "analyzed"
    run["winner"] = winner
    run["decisive"] = decisive
    _write_run(run)
    return analysis


def _evidence_bundle(run_id: str) -> tuple[bytes, str]:
    run = _read_run(run_id)
    root = _run_root(run_id)
    analysis = _load_analysis(run_id)
    tournament = _read_tournament(run_id)
    public_tournament = _tournament_public(tournament, run) if tournament else None
    analyzer_status = analysis.get("analyzer_status") or {}
    lines = [
        f"# LoRA Lab evidence — {run.get('name') or run_id}",
        "",
        f"- Run: `{run_id}`",
        f"- Created: {run.get('created_at') or 'unknown'}",
        f"- Profile: {run.get('profile') or 'unknown'}",
        f"- Analyser: {analysis.get('analyzer') or 'unknown'}",
        f"- Winner: {analysis.get('winner') or 'unknown'}",
        f"- Confidence: {analysis.get('confidence') or 'unknown'}",
        f"- Model agreement: {analyzer_status.get('model_agreement_count', 0)}/{analyzer_status.get('model_agreement_total', 0)} prompt winners",
        "",
        "## Automatic ranking",
        "",
        "| Rank | Candidate | Final | Ensemble | KP-RPE cosine | Antelope cosine | Face coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in analysis.get("ranking") or []:
        candidate = item.get("candidate") or {}
        lines.append(
            f"| {item.get('rank', '')} | {candidate.get('label', candidate.get('filename', ''))} | "
            f"{float(item.get('final_score') or 0):.2f} | {float(item.get('mean_similarity') or 0):.3f} | "
            f"{float(item.get('mean_kprpe_similarity') or 0):.3f} | {float(item.get('mean_antelope_similarity') or 0):.3f} | "
            f"{float(item.get('detection_rate') or 0):.0f}% |"
        )
    if public_tournament:
        lines.extend([
            "",
            "## Human validation",
            "",
            f"- Identity winner: {public_tournament.get('human_winner') or 'not available'}",
            f"- Overall-preference winner: {public_tournament.get('overall_winner') or 'not available'}",
            f"- AI winner: {public_tournament.get('automatic_winner') or 'not available'}",
            f"- Identity/AI agreement: {public_tournament.get('agreement_count', 0)}/{public_tournament.get('agreement_total', 0)}",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        analysis.get("metric_notes", {}).get("identity", ""),
        "",
        "Face quality is diagnostic confidence only; it does not add identity points.",
    ])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("README.md", "\n".join(lines))
        for name in ("LAB_RUN.json", "LAB_ANALYSIS.json", "LAB_RANKING.csv", "LAB_TOURNAMENT.json", "LAB_RATINGS.json"):
            path = root / name
            if path.is_file():
                archive.write(path, name)
    return buffer.getvalue(), f"{_safe_slug(run.get('name') or run_id)}_evidence.zip"


def _bootstrap_payload() -> dict:
    try:
        import nodes
        installed_node_types = set(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        installed_node_types = set()
    profiles = []
    for profile in PROFILES.values():
        item = dict(profile)
        missing = []
        try:
            item["installed_model"] = _model_filename(profile)
        except Exception as exc:
            missing.append(str(exc))
        try:
            item["installed_clip"] = _profile_file(profile, "clip_contains", "text_encoders")
        except Exception as exc:
            missing.append(str(exc))
        if profile.get("clip_2_contains"):
            try:
                item["installed_clip_2"] = _profile_file(profile, "clip_2_contains", "text_encoders")
            except Exception as exc:
                missing.append(str(exc))
        try:
            item["installed_vae"] = _profile_file(profile, "vae_contains", "vae")
        except Exception as exc:
            missing.append(str(exc))
        if profile.get("sampler") not in comfy.samplers.KSampler.SAMPLERS:
            missing.append(f"Sampler is unavailable: {profile.get('sampler')}")
        if profile.get("scheduler") not in comfy.samplers.KSampler.SCHEDULERS:
            missing.append(f"Scheduler is unavailable: {profile.get('scheduler')}")
        required_nodes = {"UNETLoader", "VAELoader", "KSampler", profile.get("latent")}
        required_nodes.add("DualCLIPLoader" if profile.get("adapter") == "split_dual" else "CLIPLoader")
        if profile.get("model_sampling"):
            required_nodes.add(profile["model_sampling"]["class_type"])
        for node_type in sorted(required_nodes - installed_node_types):
            missing.append(f"ComfyUI node is unavailable: {node_type}")
        item["missing"] = missing
        item["available"] = not missing
        if missing:
            item["error"] = " · ".join(missing)
        profiles.append(item)
    try:
        from .analyzer_installer import read_status
        analyzer = read_status()
    except Exception as exc:
        analyzer = {"state": "error", "message": str(exc)}
    return {
        "ok": True,
        "version": LAB_VERSION,
        "hardware": _hardware(),
        "profiles": profiles,
        "model_families": sorted({item["family"] for item in profiles}),
        "diffusion_models": _available_files("diffusion_models"),
        "text_encoders": _available_files("text_encoders"),
        "vaes": _available_files("vae"),
        "samplers": list(comfy.samplers.KSampler.SAMPLERS),
        "schedulers": list(comfy.samplers.KSampler.SCHEDULERS),
        "model_patch_nodes": _model_patch_catalog(),
        "turbo_lora": {
            "available": _preferred_turbo_lora() is not None,
            "filename": _preferred_turbo_lora(),
        },
        "presets": list(PRESETS.values()),
        "default_prompts": DEFAULT_PROMPTS,
        "loras": _lora_catalog(),
        "references": _reference_folders(),
        "runs": _list_runs(),
        "analyzer": analyzer,
        "watchers": [_watcher_public(item) for item in _read_watchers().get("watchers", {}).values()],
        "defaults": {
            "profile": "krea2_turbo",
            "preset": "quick",
            "trigger": "",
            "subject_class": "",
            "common_strength": 1.0,
            "strengths": [0.65, 0.8, 0.95, 1.1, 1.25],
            "include_baseline": True,
            "reference_folder": "lora_reference",
            "turbo_lora": _preferred_turbo_lora(),
        },
    }


if PromptServer is not None and web is not None:
    routes = PromptServer.instance.routes

    @routes.get("/loralab/v1/bootstrap")
    async def loralab_bootstrap(request):
        try:
            _resume_watch_tasks(f"{request.scheme}://{request.host}")
            return web.json_response(_bootstrap_payload())
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @routes.post("/loralab/v1/plan")
    async def loralab_plan(request):
        try:
            payload = await request.json()
            run = _make_plan(payload)
            return web.json_response({"ok": True, "run": run})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/workflow-preview")
    async def loralab_workflow_preview(request):
        """Compile one representative job without queueing or loading a model."""
        try:
            payload = await request.json()
            run = _read_run(str(payload.get("run_id") or ""))
            scenario_index = int(payload.get("scenario_index") or 0)
            candidate_index = int(payload.get("candidate_index") or 0)
            scenario = run["scenarios"][scenario_index]
            candidate = run["candidates"][candidate_index]
            workflow = _job_prompt(run, scenario, candidate, candidate_index)
            import nodes
            required = sorted({node["class_type"] for node in workflow.values()})
            missing = sorted(set(required) - set(nodes.NODE_CLASS_MAPPINGS))
            return web.json_response({"ok": not missing, "workflow": workflow, "required_nodes": required, "missing_nodes": missing})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/start")
    async def loralab_start(request):
        try:
            payload = await request.json()
            run_id = str(payload.get("run_id") or "")
            with _RUN_LOCK:
                run = _read_run(run_id)
                if run["status"] == "cancelled":
                    run["submitted_jobs"] = {key: value for key, value in run.get("submitted_jobs", {}).items() if key in _completed_keys(run)}
                run["status"] = "queueing"
                _write_run(run)
            base_url = f"{request.scheme}://{request.host}"
            started = _start_queue_task(run_id, base_url, payload.get("client_id"))
            return web.json_response({"ok": True, "started": started, "run_id": run_id})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.get("/loralab/v1/run")
    async def loralab_run(request):
        try:
            run = _read_run(request.query.get("run_id", ""))
            progress = _run_progress(run, include_cells=True)
            analysis_path = _run_root(run["run_id"]) / "LAB_ANALYSIS.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else None
            tournament = _read_tournament(run["run_id"])
            return web.json_response({
                "ok": True,
                "run": run,
                "progress": progress,
                "analysis": analysis,
                "ratings": _read_ratings(run["run_id"]),
                "tournament": _tournament_public(tournament, run),
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)

    @routes.get("/loralab/v1/runs")
    async def loralab_runs(request):
        return web.json_response({"ok": True, "runs": _list_runs(int(request.query.get("limit", 30)))})

    @routes.get("/loralab/v1/watch")
    async def loralab_watch_get(request):
        data = _read_watchers()
        return web.json_response({"ok": True, "watchers": [_watcher_public(item) for item in data.get("watchers", {}).values()]})

    @routes.post("/loralab/v1/watch")
    async def loralab_watch_post(request):
        try:
            payload = await request.json()
            action = str(payload.get("action") or "start")
            data = _read_watchers()
            if action == "start":
                group = str(payload.get("group") or "").strip()
                template = payload.get("template")
                if not group:
                    raise ValueError("Choose a checkpoint series/group before starting watcher.")
                if not isinstance(template, dict):
                    raise ValueError("Watcher requires a run template.")
                current = [item["filename"] for item in _lora_catalog() if item["group"] == group]
                if not current:
                    raise ValueError(f"No installed checkpoints belong to group: {group}")
                watcher_id = f"watch_{_safe_slug(group)}"
                watcher = {
                    "watcher_id": watcher_id,
                    "group": group,
                    "active": True,
                    "created_at": _now_iso(),
                    "interval_seconds": max(10, min(300, int(payload.get("interval_seconds", 15)))),
                    "known_files": current,
                    "observed": {},
                    "run_ids": [],
                    "errors": [],
                    "template": template,
                }
                data["watchers"][watcher_id] = watcher
                _write_watchers(data)
                _start_watch_task(watcher_id, f"{request.scheme}://{request.host}")
            elif action in {"stop", "delete"}:
                watcher_id = str(payload.get("watcher_id") or "")
                watcher = data["watchers"].get(watcher_id)
                if not watcher:
                    raise FileNotFoundError(f"Watcher not found: {watcher_id}")
                watcher["active"] = False
                watcher["stopped_at"] = _now_iso()
                task = _WATCH_TASKS.get(watcher_id)
                if task and not task.done():
                    task.cancel()
                if action == "delete":
                    data["watchers"].pop(watcher_id, None)
                _write_watchers(data)
            else:
                raise ValueError(f"Unknown watcher action: {action}")
            return web.json_response({"ok": True, "watchers": [_watcher_public(item) for item in data["watchers"].values()]})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/register")
    async def loralab_register(request):
        try:
            payload = await request.json()
            run = _read_run(str(payload.get("run_id") or ""))
            ids = payload.get("prompt_ids") or ([payload.get("prompt_id")] if payload.get("prompt_id") else [])
            for prompt_id in ids:
                if prompt_id not in run["submitted_prompt_ids"]:
                    run["submitted_prompt_ids"].append(prompt_id)
            _write_run(run)
            return web.json_response({"ok": True})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/status")
    async def loralab_status(request):
        try:
            payload = await request.json()
            run_id = str(payload.get("run_id") or "")
            action = str(payload.get("action") or "pause")
            run = _read_run(run_id)
            base_url = f"{request.scheme}://{request.host}"
            if action == "pause":
                run["status"] = "paused"
                _write_run(run)
            elif action in {"resume", "retry"}:
                if action == "retry":
                    completed = _completed_keys(run)
                    run["submitted_jobs"] = {key: value for key, value in run.get("submitted_jobs", {}).items() if key in completed}
                    run["queue_errors"] = []
                run["status"] = "queueing"
                _write_run(run)
                _start_queue_task(run_id, base_url, payload.get("client_id"))
            elif action in {"cancel", "stop"}:
                run["status"] = "cancelled"
                _write_run(run)
                stop_result = await _stop_run_queue(run_id, base_url)
                run = _read_run(run_id)
                run["stopped_at"] = _now_iso()
                run["stop_result"] = stop_result
                run["resources_released_at"] = _now_iso()
                run["resource_cleanup_requested"] = bool(stop_result["cleanup_requested"])
                _write_run(run)
            elif action == "free":
                requested = _request_resource_cleanup()
                if ClientSession is not None:
                    async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                        with contextlib.suppress(Exception):
                            async with session.post(f"{base_url}/free", json={"unload_models": True, "free_memory": True}) as response:
                                await response.read()
                run["resources_released_at"] = _now_iso()
                run["resource_cleanup_requested"] = requested
                _write_run(run)
            else:
                raise ValueError(f"Unknown action: {action}")
            return web.json_response({
                "ok": True,
                "status": run["status"],
                "stop_result": run.get("stop_result"),
                "resource_cleanup_requested": run.get("resource_cleanup_requested", False),
            })
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/analyze")
    async def loralab_analyze(request):
        try:
            payload = await request.json()
            analysis = await asyncio.to_thread(_analyze_sync, str(payload.get("run_id") or ""), payload.get("reference_folder"))
            return web.json_response({"ok": True, "analysis": analysis})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/rating")
    async def loralab_rating(request):
        try:
            payload = await request.json()
            run_id = str(payload.get("run_id") or "")
            run = _read_run(run_id)
            p = int(payload["prompt_index"])
            l = int(payload["lora_index"])
            if p < 0 or p >= run["scenario_count"] or l < 0 or l >= run["candidate_count"]:
                raise ValueError("Rating cell is outside this run.")
            rating = {}
            for key in ("identity", "quality", "adherence"):
                value = payload.get(key)
                if value not in (None, ""):
                    value = int(value)
                    if value < 1 or value > 5:
                        raise ValueError(f"{key} must be 1–5.")
                    rating[key] = value
            rating["artifact"] = bool(payload.get("artifact", False))
            rating["updated_at"] = _now_iso()
            data = _read_ratings(run_id)
            data["ratings"][_job_key(p, l)] = rating
            _atomic_json(_ratings_path(run_id), data)
            return web.json_response({"ok": True, "rating": rating})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.post("/loralab/v1/tournament")
    async def loralab_tournament(request):
        try:
            payload = await request.json()
            run_id = str(payload.get("run_id") or "")
            action = str(payload.get("action") or "start")
            run = _read_run(run_id)
            if action == "start":
                tournament = _start_tournament(
                    run_id,
                    bool(payload.get("include_baseline", True)),
                    float(payload.get("human_weight", 0.5)),
                )
            elif action == "reset":
                existing = _read_tournament(run_id)
                tournament = _start_tournament(
                    run_id,
                    bool(existing.get("include_baseline", True)) if existing else bool(payload.get("include_baseline", True)),
                    float(existing.get("human_weight", 0.5)) if existing else float(payload.get("human_weight", 0.5)),
                    candidate_indices=list(existing.get("candidate_indices") or []) if existing else None,
                    round_number=int(existing.get("round_number", 1)) if existing else 1,
                    stage=existing.get("stage", "initial") if existing else "initial",
                    previous_rounds=list(existing.get("previous_rounds") or []) if existing else None,
                )
            elif action == "runoff":
                tournament = _start_tournament_runoff(
                    run_id,
                    int(payload.get("finalist_count", 3)),
                )
            elif action == "vote":
                tournament = _vote_tournament(
                    run_id,
                    str(payload.get("choice") or ""),
                    str(payload.get("match_id") or ""),
                    {
                        "artifact_left": payload.get("artifact_left"),
                        "artifact_right": payload.get("artifact_right"),
                        "identity_failure_left": payload.get("identity_failure_left"),
                        "identity_failure_right": payload.get("identity_failure_right"),
                        "overall_choice": payload.get("overall_choice"),
                    },
                )
            elif action == "undo":
                tournament = _undo_tournament(run_id)
            elif action == "weight":
                tournament = _read_tournament(run_id)
                if not tournament:
                    raise RuntimeError("No blind tournament exists for this run.")
                tournament["human_weight"] = max(0.0, min(1.0, float(payload.get("human_weight", 0.5))))
                tournament["updated_at"] = _now_iso()
                _atomic_json(_tournament_path(run_id), tournament)
            else:
                raise ValueError(f"Unknown tournament action: {action}")
            return web.json_response({"ok": True, "tournament": _tournament_public(tournament, run)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @routes.get("/loralab/v1/export")
    async def loralab_export(request):
        try:
            payload, filename = await asyncio.to_thread(_evidence_bundle, request.query.get("run_id", ""))
            return web.Response(
                body=payload,
                content_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)

    @routes.get("/loralab/v1/asset")
    async def loralab_asset(request):
        try:
            run_id = request.query.get("run_id", "")
            relative = Path(request.query.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe asset path.")
            root = _legacy_run_root(run_id).resolve()
            path = (root / relative).resolve()
            if root != path and root not in path.parents:
                raise ValueError("Unsafe asset path.")
            if not path.is_file():
                raise FileNotFoundError(path.name)
            return web.FileResponse(path)
        except Exception as exc:
            return web.Response(text=f"Asset error: {type(exc).__name__}: {exc}", status=404)
