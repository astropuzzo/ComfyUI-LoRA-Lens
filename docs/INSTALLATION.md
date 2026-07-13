# Installation and prerequisites

## Install the node pack

Place the repository at `ComfyUI/custom_nodes/ComfyUI-LoRA-Lens`, install `requirements.txt` with ComfyUI's Python, then restart ComfyUI.

### Stability Matrix

Open the package folder for ComfyUI, then install dependencies with its virtual environment:

```powershell
cd <StabilityMatrix>\Packages\ComfyUI
.\venv\Scripts\python.exe -m pip install -r .\custom_nodes\ComfyUI-LoRA-Lens\requirements.txt
```

### ComfyUI portable on Windows

```powershell
cd <ComfyUI-portable>
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-LoRA-Lens\requirements.txt
```

### Linux or a virtual environment

```bash
cd /path/to/ComfyUI
./venv/bin/python -m pip install -r custom_nodes/ComfyUI-LoRA-Lens/requirements.txt
```

## Face analyser

The analyser setup is automatic and non-blocking. It installs/downloads:

- CVLFace ViT KP-RPE with AdaFace trained on WebFace12M.
- InsightFace AntelopeV2.
- SCRFD/InsightFace face detection and alignment dependencies.

The dashboard reports `installing`, `ready`, or an actionable error. Restart ComfyUI after the first completed installation if the status does not refresh.

## Model prerequisites

LoRA Lens never downloads generation models automatically. The setup page lists missing files for the selected profile. Standard locations are:

```text
ComfyUI/models/diffusion_models
ComfyUI/models/text_encoders
ComfyUI/models/vae
ComfyUI/models/loras
```

Keep ComfyUI updated because Z-Image and newer model families require recent core nodes. Exact file names and preferred alternatives are shown in [Model adapters](MODEL_ADAPTERS.md).

