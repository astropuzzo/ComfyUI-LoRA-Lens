# Changelog

## 7.2.1 - 2026-07-30

- Made revealed tournament winners and ranking candidates open their exact installed LoRA location.
- Added safe cross-platform file reveal with Explorer selection on Windows.

## 7.2.0 - 2026-07-30

- Added browser-local real-photo references beside blind tournament duels, with multi-image navigation and full-size viewing.
- Added top-2, top-3, and top-4 finalist runoffs with fresh bracket ordering.
- Preserved completed round summaries, comparisons, and standings in run evidence across repeated runoffs.
- Made undo and reset operate on the current tournament round without discarding archived rounds.

## 7.1.0 - 2026-07-24

- Rebuilt Stop as a run-owned cancellation flow that halts submission, removes
  pending prompts, interrupts the active prompt, and verifies queue removal.
- Added automatic ComfyUI model unloading and VRAM cleanup after completed and
  stopped runs.
- Added a manual **Release VRAM** action.
- Released persistent AntelopeV2 and CVLFace GPU resources after analysis.
- Added an ordered stack of up to eight always-on auxiliary LoRAs with
  independent strengths.
- Applied auxiliary stacks consistently to native and imported API workflows.

## 7.0.0 - 2026-07-13

- Renamed the product to ComfyUI LoRA Lens.
- Added categorized native model adapters for Krea 2, FLUX.1, Z-Image, and Anima.
- Added explicit base/deployment variants including Turbo, Schnell, Aesthetic, Raw, and Dev.
- Added universal ComfyUI API-workflow import with controlled placeholders.
- Added per-profile prerequisite inspection and non-executing workflow compilation checks.
- Preserved arbitrary custom-node model-patch chains.
- Removed all person-specific triggers, search terms, and reference-folder defaults.
- Retained dual face recognition, blind pairwise voting, full-size viewer, checkpoint watcher, and evidence export from 6.x.
