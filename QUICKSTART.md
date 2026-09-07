# Quick Start Guide - Knowledge Distillation Pipeline

## Notebook Cell Order

The numbers below refer to the current top-to-bottom order in `pipeline.ipynb`.

1. **Cell 1:** Set `OPENROUTER_API_KEY` in the environment. No key is stored in the notebook.
2. **Cell 2:** Download or load the local Llama model.
3. **Cell 3:** Create the OpenRouter teacher client.
4. **Cell 4:** Load the mathematical function and prompt generators.
5. **Cells 5-10:** Setup and status cells. They do not generate data, call the API, copy models, or train.
6. **Cell 11:** Load the black-box distillation functions.
7. **Cell 13:** Run the schema validation check.

Run cells 1-4, 11, and 13 in that order before starting the workflow. Cells 12, 14, 17, 19, and 20 are not required for the normal workflow.

## Generate New Data

Set the API key outside the notebook. Then run cells 1-4, 11, and 13. In **cell 15**, uncomment and run:

```python
records = generate_distillation_data(deepseek, num_examples=200)
```

Each teacher request explicitly asks for JSON with both fields:

```json
{
  "answer_rationale": "step-by-step reasoning",
  "answer": "final answer only"
}
```

Invalid JSON or a missing or empty field is retried up to three times. Invalid records are never written. Successful generation creates these files in `data/`:

- `distillation_data.jsonl`: records with `input_prompt`, `answer_rationale`, and `answer`.
- `train_data.jsonl`: training split.
- `val_data.jsonl`: validation split, kept outside training.
- `multitask_train.jsonl` and `multitask_val.jsonl`: paired task records.

Each source example produces two records with the same input. `predict: <input prompt>` targets `answer`; `explain: <input prompt>` targets `answer_rationale`. Use `inference_prompts(input_prompt)` so inference uses the exact same prefixes.

## Make Model Copies

After loading the functions in cell 11, run **cell 18** once by uncommenting:

```python
model_variants = prepare_model_variants()
```

This creates independent `baseline`, `distilled`, and `last_layer` copies. The source model is not modified. Each copy is approximately 13 GB.

## Training Script

The Lightning training script is already available at `data/train_variants.py`.
Cell 16 only confirms its location. Do not use the older
`write_training_script()` helper, because it would overwrite the resumable
Lightning trainer. In `last_layer` mode, only `lm_head` is trainable.

## Train The Variants

Run these commands from the repository root after cell 15 has generated data:

```powershell
python .\data\train_variants.py --mode full --model .\models\llama-3.2-3b-distilled --output .\data\full_distilled
python .\data\train_variants.py --mode last_layer --model .\models\llama-3.2-3b-last-layer --output .\data\last_layer_distilled
```

The baseline copy is not trained. Do not pass `val_data.jsonl` or `multitask_val.jsonl` to the training script.

Training uses PyTorch Lightning and writes a checkpoint after every optimizer
update. To run a bounded session, add for example:

```powershell
python .\data\train_variants.py --mode full --model .\models\llama-3.2-3b-distilled --output .\data\full_distilled --max-time 02:00:00
```

The single checkpoint is stored as `data/full_distilled/checkpoints/last.ckpt`
and is overwritten after every optimizer update. CSV loss logs are stored in
`data/full_distilled/logs/`. A stable update-level loss file is also written to
`data/full_distilled/loss_history.csv`. Re-run the same command with `--resume auto` (the
default) to restore model weights, optimizer state, scheduler state, epoch,
global step, and random state. Use `--resume none` to deliberately start over.

The training dataloader uses a resumable random sampler. Its shuffled
permutation and current position are stored in `last.ckpt`, so restarting
mid-epoch continues with the next unconsumed batch instead of restarting the
epoch with a new shuffle order.

Because checkpoints are written every optimizer update, the output directory
can become large. Keep the checkpoint directory between sessions; removing it
removes the resumable training state.

## Troubleshooting

**OpenRouter error:** confirm `OPENROUTER_API_KEY` is set and optionally set `OPENROUTER_MODEL`.

**Missing data:** run cell 15 after cells 1-4, 11, and 13.

**Missing trainer:** restore `data/train_variants.py`; cell 16 does not generate it.

**CUDA out of memory:** reduce `--batch-size` or increase
`--gradient-accumulation`.
