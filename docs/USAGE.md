# Using LoRA Lens

## A reliable checkpoint search

1. Run a **Quick screen** across spaced checkpoints, not every saved file.
2. Use one matched seed, 768 px, and a no-LoRA baseline.
3. Eliminate obvious underfit, overfit, or broken candidates.
4. Run a **Strength sweep** only on the strongest checkpoint.
5. Retest two to four finalists at native resolution with two or three seeds.
6. Complete blind identity and overall-preference votes before revealing automatic ranks.

More epochs do not automatically mean a better LoRA. The right stopping point depends on dataset size, repeats, batch/accumulation, learning rate, optimizer behavior, rank, and augmentation. LoRA Lens evaluates outputs; it does not pretend that a fixed epoch count is universally correct.

## Prompt placeholders

- `{trigger}`: exact trigger token or phrase.
- `{class}`: optional subject/class, blank unless supplied.
- `{subject}`: trigger plus optional class.

The resolved positive prompt is displayed under every prompt. No hidden word such as `woman` is added.

## Controls remain editable

Native profiles are starting points. Model file, encoders, VAE, steps, CFG, sampler, scheduler, negative conditioning, Turbo LoRA, and model-patch chain remain editable. A profile change applies official defaults once; it does not lock controls.

## Reference faces

Put 8–20 varied, high-quality, single-face holdout images in a folder under `ComfyUI/input`, then select that relative folder. Prefer references not used for training. Include realistic variation in pose, expression, lighting, and age while rejecting blurred, tiny, occluded, or wrong-identity faces.

## Results

Use arrow keys or on-screen arrows to inspect full-size outputs one by one. Rate identity and overall preference independently. The blind tournament presents matched images pairwise until a winner emerges, then compares human and automatic rankings.

