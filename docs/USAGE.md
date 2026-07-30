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

Native profiles are starting points. Model file, encoders, VAE, steps, CFG, sampler, scheduler, negative conditioning, Turbo LoRA, auxiliary LoRAs, and model-patch chain remain editable. A profile change applies official defaults once; it does not lock controls.

## Always-on auxiliary LoRAs

Add up to eight acceleration, style, detail, or compatibility LoRAs that must
be shared by every generated cell. Set an independent strength for each and
arrange them in execution order. LoRA Lens applies the stack before the
candidate checkpoint. The control column keeps the auxiliary stack but removes
the candidate, which isolates the candidate's actual contribution.

Do not add the same file as Turbo, auxiliary, and candidate LoRA. LoRA Lens
rejects these duplicate applications before GPU work starts.

## Stop and memory cleanup

**Pause submissions** prevents new prompts but leaves already submitted work in
ComfyUI. **Stop run now** is the destructive run-level stop: it cancels the
submission task, removes pending prompts owned by the run, interrupts its active
prompt, and requests ComfyUI model unload plus cache cleanup. Completed cells
remain available for inspection or a later retry.

LoRA Lens also requests cleanup when a run completes. Use **Release VRAM** to
repeat it manually. The face analyser releases both CVLFace and AntelopeV2 GPU
resources after every analysis.

## Reference faces

Put 8–20 varied, high-quality, single-face holdout images in a folder under `ComfyUI/input`, then select that relative folder. Prefer references not used for training. Include realistic variation in pose, expression, lighting, and age while rejecting blurred, tiny, occluded, or wrong-identity faces.

## Results

Use arrow keys or on-screen arrows to inspect full-size outputs one by one. Rate identity and overall preference independently. The blind tournament presents matched images pairwise until a winner emerges, then compares human and automatic rankings.

Before or during a blind tournament, use **Load photo** to keep one or more real identity references beside every duel. These files stay inside the browser session and are never uploaded or copied into the run. Use the reference arrows to switch photos, or click the photo for the full-size viewer.

When the first result is not convincing, choose the top 2–4 checkpoints and start a **finalist runoff**. The runoff reshuffles the bracket and collects a new set of votes from only those finalists. Earlier rounds remain archived in `LAB_TOURNAMENT.json` and in the evidence export; resetting a runoff resets only its current votes.
