# Contributing

Issues and pull requests are welcome. Keep changes focused, preserve existing run-file compatibility, and avoid adding model weights or personal datasets.

For a new native adapter:

1. Use an official model card or official ComfyUI workflow as the source.
2. Add declarative metadata to `model_adapters.py`.
3. Declare loader type, encoder type, latent node, optional model-sampling patch, prerequisites, and conservative defaults.
4. Verify the compiled workflow through `/loralab/v1/workflow-preview` without queueing GPU work.
5. Update the support table and cite the upstream license where relevant.

Before submitting:

```bash
python -m py_compile *.py
node --check web/lora_lab.js
```

Do not claim that one optimizer, rank, alpha, epoch count, or sampler is universally best. Recommendations must state the model family and evidence.

