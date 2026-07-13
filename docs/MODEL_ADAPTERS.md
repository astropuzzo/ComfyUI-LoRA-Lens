# Model adapters

## Why adapters exist

Image-model families differ in loader nodes, text encoders, VAEs, latent formats, guidance, and sampling patches. LoRA Lens separates the evaluation engine from the workflow adapter so adding a model does not duplicate ranking, viewer, tournament, or history code.

## Native profiles

| Profile | Required generation files | Defaults |
|---|---|---|
| Krea 2 Raw | Krea 2 Raw diffusion model, Qwen3-VL 4B encoder, Qwen Image VAE | 52 steps, CFG 3.5, Euler/simple |
| Krea 2 Turbo | Krea 2 Turbo diffusion model, same encoder/VAE | 8 steps, CFG 1, Euler/simple |
| FLUX.1 Dev | `flux1-dev`, CLIP-L, T5XXL, `ae.safetensors` | 20 steps, CFG 1, Euler/simple |
| FLUX.1 Schnell | `flux1-schnell`, CLIP-L, T5XXL, `ae.safetensors` | 4 steps, CFG 1, Euler/simple |
| FLUX.1 Krea Dev | `flux1-krea-dev`, CLIP-L, T5XXL, `ae.safetensors` | 20 steps, CFG 1, Euler/simple |
| Z-Image Base | `z_image_bf16`, `qwen_3_4b`, `ae.safetensors` | 25 steps, CFG 4, res_multistep/simple |
| Z-Image Turbo | `z_image_turbo`, `qwen_3_4b`, `ae.safetensors` | 8 steps, CFG 1, res_multistep/simple |
| Anima Base | `anima-base-v1.0`, `qwen_3_06b_base`, Qwen Image VAE | 30 steps, CFG 4, er_sde/simple |
| Anima Aesthetic | matching Aesthetic diffusion model, same encoder/VAE | 30 steps, CFG 4, er_sde/simple |
| Anima Turbo | matching Turbo diffusion model, same encoder/VAE | 8 steps, CFG 1, Euler/simple |

Defaults follow the upstream model cards or official ComfyUI workflow templates and remain editable. Quantized filenames are detected by family patterns rather than one exact build name.

## Import any ComfyUI API workflow

This is the forward-compatibility path for new models, GGUF nodes, custom enhancers, ControlNet-like patches, or pipelines too specialized for a native adapter.

1. Build and test the graph normally in ComfyUI.
2. Replace the candidate LoRA loader with **LoRA Lab · Identity / Baseline Loader**.
3. Replace values you want LoRA Lens to control with exact placeholders.
4. Export with **Save (API Format)** and import the JSON in LoRA Lens.
5. Optionally enter the node ID and output index that produce the final `IMAGE`. Otherwise, the first PreviewImage/SaveImage input is detected.

Supported exact placeholders:

```text
{{PROMPT}}  {{NEGATIVE_PROMPT}}  {{MODEL}}  {{CLIP}}  {{CLIP_2}}
{{VAE}}     {{SEED}}             {{STEPS}}  {{CFG}}   {{SAMPLER}}
{{SCHEDULER}}  {{WIDTH}}         {{HEIGHT}}
```

The imported graph is preserved. LoRA Lens changes the placeholders and identity-LoRA node for each matched job, then attaches its collector to the selected image output. It validates API format and installed node types before GPU work begins.

