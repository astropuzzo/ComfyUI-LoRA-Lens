import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

PLUGIN_VERSION = "6.0.0"
MODEL_NAME = "antelopev2"
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
_REQUIRED_ONNX = {"glintr100.onnx", "scrfd_10g_bnkps.onnx"}
_LOCK = threading.Lock()
_THREAD = None
CVLFACE_MODEL_REPO = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
CVLFACE_ALIGNER_REPO = "minchul/cvlface_DFA_mobilenet"


def _plugin_dir():
    return Path(__file__).resolve().parent


def _comfy_root():
    # .../ComfyUI/custom_nodes/ComfyUI_LoRA_Prompt_Queue
    return _plugin_dir().parent.parent


def models_root():
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "insightface"
    except Exception:
        return _comfy_root() / "models" / "insightface"


def model_dir():
    return models_root() / "models" / MODEL_NAME


def status_path():
    return _plugin_dir() / "analyzer_status.json"


def _write_status(state, message, **extra):
    payload = {
        "version": PLUGIN_VERSION,
        "state": state,
        "message": message,
        "updated": time.time(),
        **extra,
    }
    try:
        status_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(f"[LoRA Test Analyzer] {state}: {message}", flush=True)
    return payload


def read_status():
    ready, details = analyzer_ready_details()
    if ready:
        return _write_status("ready", "CVLFace KP-RPE + AntelopeV2 ensemble is ready.", **details)
    try:
        if status_path().exists():
            payload = json.loads(status_path().read_text(encoding="utf-8"))
            payload.update(details)
            return payload
    except Exception:
        pass
    return {"version": PLUGIN_VERSION, "state": "missing", "message": "Analyzer is not installed.", **details}


def _module_present(name):
    return importlib.util.find_spec(name) is not None


def _model_ready():
    target = model_dir()
    if not target.exists(): return False
    present = {p.name for p in target.glob('*.onnx')}
    return _REQUIRED_ONNX.issubset(present)


def analyzer_ready_details():
    modules = {
        "insightface": _module_present("insightface"),
        "onnxruntime": _module_present("onnxruntime"),
        "cv2": _module_present("cv2"),
    }
    model = _model_ready()
    providers = []
    if modules["onnxruntime"]:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:
            providers = []
    antelope_ready = all(modules.values()) and model
    cvl_root = _comfy_root() / "models" / "cvlface"
    cvl_required = [
        cvl_root / "adaface_vit_base_kprpe_webface12m" / "pretrained_model" / "model.pt",
        cvl_root / "adaface_vit_base_kprpe_webface12m" / "pretrained_model" / "model.yaml",
        cvl_root / "dfa_mobilenet" / "pretrained_model" / "model.pt",
        cvl_root / "dfa_mobilenet" / "pretrained_model" / "model.yaml",
    ]
    cvlface_ready = all(path.is_file() for path in cvl_required)
    ready = antelope_ready and cvlface_ready
    return ready, {
        "modules": modules,
        "model_ready": model,
        "model_dir": str(model_dir()),
        "providers": providers,
        "gpu": "CUDAExecutionProvider" in providers,
        "antelope_ready": antelope_ready,
        "cvlface_ready": cvlface_ready,
        "cvlface_root": str(cvl_root),
    }


def _run_pip(args):
    command = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-input", *args]
    print("[LoRA Test Analyzer] running:", " ".join(command), flush=True)
    return subprocess.run(command, check=True)


def _ensure_pip():
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except Exception:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)


def _install_dependencies(force=False):
    ready, details = analyzer_ready_details()
    if all(details["modules"].values()) and not force:
        return
    _write_status("installing", "Installing AntelopeV2 analyzer dependencies with ComfyUI's Python...")
    _ensure_pip()
    _run_pip(["install", "--upgrade", "setuptools>=68", "wheel", "cython>=3.0"])
    packages = ["insightface==0.7.3", "opencv-python-headless>=4.9.0.80"]
    _run_pip(["install", "--upgrade", *packages])
    # Prefer CUDA. Fall back to CPU only if the GPU wheel cannot be installed.
    try:
        _run_pip(["install", "--upgrade", "onnxruntime-gpu>=1.19.0"])
    except Exception as gpu_error:
        _write_status("installing", f"GPU runtime install failed; using CPU fallback. {gpu_error}")
        _run_pip(["install", "--upgrade", "onnxruntime>=1.19.0"])
    importlib.invalidate_caches()


def _download_with_progress(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LoRA-Test-Analyzer/4.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        while True:
            block = response.read(1024 * 1024)
            if not block: break
            output.write(block)
            downloaded += len(block)
            if total:
                pct = int(downloaded * 100 / total)
                if pct // 5 != last_pct // 5:
                    last_pct = pct
                    _write_status("downloading", f"Downloading AntelopeV2 model: {pct}%", downloaded=downloaded, total=total)


def _install_model(force=False):
    target = model_dir()
    if _model_ready() and not force:
        return
    _write_status("downloading", "Downloading the official InsightFace AntelopeV2 model pack (about 344 MB)...")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="antelopev2_") as temp_name:
        temp = Path(temp_name)
        archive = temp / "antelopev2.zip"
        _download_with_progress(MODEL_URL, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError("Downloaded AntelopeV2 file is not a valid ZIP archive.")
        extracted = temp / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        candidates = []
        for directory in [extracted, *[p for p in extracted.rglob('*') if p.is_dir()]]:
            names = {p.name for p in directory.glob('*.onnx')}
            if _REQUIRED_ONNX.issubset(names):
                candidates.append(directory)
        if not candidates:
            raise RuntimeError("AntelopeV2 archive does not contain the expected recognition and detection models.")
        source = min(candidates, key=lambda p: len(p.parts))
        if target.exists(): shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for file in source.glob('*.onnx'):
            shutil.copy2(file, target / file.name)
    if not _model_ready():
        raise RuntimeError("AntelopeV2 model installation did not complete correctly.")


def _install_cvlface_models(force=False):
    root = _comfy_root() / "models" / "cvlface"
    model_target = root / "adaface_vit_base_kprpe_webface12m"
    aligner_target = root / "dfa_mobilenet"
    required = [model_target / "pretrained_model" / "model.pt", aligner_target / "pretrained_model" / "model.pt"]
    if all(path.is_file() for path in required) and not force:
        return
    _write_status("downloading", "Downloading CVLFace KP-RPE AdaFace and DFA aligner (about 465 MB)...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _run_pip(["install", "--upgrade", "huggingface-hub>=0.28", "transformers>=4.44", "omegaconf>=2.3", "timm>=1.0"])
        importlib.invalidate_caches()
        from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(CVLFACE_MODEL_REPO, local_dir=str(model_target), force_download=force)
    snapshot_download(CVLFACE_ALIGNER_REPO, local_dir=str(aligner_target), force_download=force)
    rpe_init = model_target / "models" / "vit_kprpe" / "RPE" / "__init__.py"
    if rpe_init.is_file():
        source = rpe_init.read_text(encoding="utf-8")
        marker = "# LORALAB_PURE_TORCH_RPE"
        if marker not in source:
            start = source.find("except ImportError:")
            end = source.find("\n\ndef build_rpe", start)
            if start >= 0 and end > start:
                replacement = 'except ImportError:\n    # LORALAB_PURE_TORCH_RPE\n    warnings.warn("rpe_ops is not built; using the pure-Torch KP-RPE fallback.")\n'
                rpe_init.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
    if not all(path.is_file() for path in required):
        raise RuntimeError("CVLFace model installation did not complete correctly.")


def ensure_analyzer_sync(force=False):
    with _LOCK:
        try:
            _install_dependencies(force=force)
            _install_model(force=force)
            _install_cvlface_models(force=force)
            # Clear stale imports and verify.
            importlib.invalidate_caches()
            ready, details = analyzer_ready_details()
            if not ready:
                raise RuntimeError(f"Analyzer verification failed: {details}")
            return _write_status("ready", "CVLFace KP-RPE + AntelopeV2 ensemble is ready.", **details)
        except Exception as exc:
            return _write_status("error", f"{type(exc).__name__}: {exc}")


def ensure_analyzer_async(force=False):
    global _THREAD
    ready, _ = analyzer_ready_details()
    if ready and not force:
        return None
    if _THREAD is not None and _THREAD.is_alive():
        return _THREAD
    def worker():
        ensure_analyzer_sync(force=force)
    _THREAD = threading.Thread(target=worker, name="LoRA-Test-AntelopeV2-Installer", daemon=True)
    _THREAD.start()
    return _THREAD
