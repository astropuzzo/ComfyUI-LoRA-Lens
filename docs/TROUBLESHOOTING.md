# Troubleshooting

## The profile says prerequisites are missing

Read the full warning under the execution stack. It names each missing diffusion model, encoder, VAE, sampler, scheduler, or ComfyUI node. Put files in the correct model subfolder and refresh the dashboard. Update ComfyUI if a core node such as `ModelSamplingAuraFlow` is unavailable.

## An imported workflow is rejected

- Export with **Save (API Format)**, not the normal UI workflow format.
- Include exactly one or more `LoRALabIdentityLoader` nodes.
- Keep placeholders exact, including double braces and uppercase spelling.
- Add PreviewImage/SaveImage or specify a node ID whose selected output is `IMAGE`.
- Install every custom node used by the workflow.

## Turbo results look wrong

Do not apply acceleration twice. Use either a distilled Turbo/Schnell diffusion model or the compatible acceleration LoRA on its base model. LoRA Lens blocks the known Krea 2 double-application case.

## Stop does not appear immediate

Some model-loading and custom-node operations only observe ComfyUI's interrupt
flag at safe boundaries. **Stop run now** prevents all new submissions
immediately, removes pending prompts, repeatedly interrupts the active prompt
for up to eight seconds, and then requests VRAM cleanup. The active operation
may need to reach its next safe boundary before it disappears from the queue.

If memory remains reserved after the queue is empty, press **Release VRAM**.
CUDA tools may still display a small ComfyUI process allocation; reserved VRAM
is not the same as memory held by a loaded diffusion model.

## An auxiliary LoRA is rejected

- Every selected file must exist in `ComfyUI/models/loras`.
- A file may appear only once in the auxiliary stack.
- Do not also select an auxiliary file as a checkpoint candidate.
- Do not duplicate the Krea Turbo LoRA in the auxiliary stack.
- Strength must be between `-2` and `2`; at most eight auxiliaries are allowed.

## Face ranking feels random

- Use 8–20 clean holdout references of the same identity.
- Avoid group photos, tiny faces, heavy occlusion, duplicates, and training-set mistakes.
- Ensure the generated face is large enough to detect.
- Compare automatic ranking with the blind human tournament.
- Treat very low detector/quality confidence as insufficient evidence.

## InsightFace or CVLFace installation fails

Run the dependency command from [Installation](INSTALLATION.md) with the exact Python executable used by ComfyUI. Check free disk space, network access to Hugging Face, and CUDA/ONNX Runtime compatibility. Restart ComfyUI after fixing dependencies.

## ComfyUI starts with an import error

Inspect the terminal log for the first exception. A frequent cause is installing requirements into system Python rather than ComfyUI's environment. Re-run the documented command with the package's `venv` or portable Python.

When reporting a bug, include ComfyUI version, LoRA Lens version, OS, GPU/VRAM, selected profile, and the first complete traceback. Do not upload private reference images or model weights.
