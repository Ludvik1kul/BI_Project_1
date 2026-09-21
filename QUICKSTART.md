# Quick Start Guide

This repository contains a Llama 3.2 3B derivative-distillation experiment.
The notebook prepares data and evaluation; training is started separately with
`data/train_variants.py`.

## Prerequisites

Run all commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the notebook kernel is unavailable, or Transformers reports that
`device_map="auto"` requires another package, install the missing notebook
runtime dependencies in the active environment:

```powershell
python -m pip install ipykernel accelerate
```

The base Llama model is gated on Hugging Face. Accept the model license and
authenticate before running the model-loading cell. The model and each copy
require several gigabytes of disk space. Training the 3B model also requires
substantial RAM/VRAM, an estimated 24GB minimum.

Open the notebook with the repository root as its working directory:


## Recommended Notebook Run

Run the cells from the top through the helper/status cells in order:

1. Read the introductory markdown cell.
2. Run the imports cell.
3. Run the environment-key cell. `OPENROUTER_API_KEY` is only required to
  generate new teacher data.
4. Run the model-loading cell if the local base model is not already available.
  It downloads and loads `models/llama-3.2-3b` and performs a small chat test.
5. Run the teacher-client, function-generator, and orchestration cells.
6. Run the parser, strict-generation, and JSONL-writer definition cells.
7. Run the helper/status cells.

Do not use **Run All** with the current notebook. Two cells that are labelled
optional are currently executable: the new-data cell calls OpenRouter
immediately, and the legacy-migration cell expects a file that is not in this
repository. Skip both unless their prerequisites have been supplied.

The repository already contains `data/distillation_data.jsonl`,
`data/train_data.jsonl`, and `data/val_data.jsonl`, so the normal reproduction
path does not need an API call or the migration cell.

## Generate New Teacher Data

Only do this when `OPENROUTER_API_KEY` is set and you intend to replace the
checked-in dataset. Run the strict-generation definitions first, then run:

```python
records = generate_distillation_data(deepseek, num_examples=200)
```

The teacher is asked for JSON containing non-empty `answer_rationale` and
`answer` fields. Invalid responses are retried up to three times. The cell
writes:

- `data/distillation_data.jsonl`
- `data/train_data.jsonl`
- `data/val_data.jsonl`
- `data/multitask_train.jsonl`
- `data/multitask_val.jsonl`

Training expands each source record into two tasks: `predict:` targets the
final answer and `explain:` targets the rationale.

## Prepare Model Variants

Run the model-copy cell once after the orchestration definitions:

```python
model_variants = prepare_model_variants()
```

This creates independent copies at:

- `models/llama-3.2-3b-baseline`
- `models/llama-3.2-3b-distilled`
- `models/llama-3.2-3b-last-layer`

The source model is not modified. The baseline is not trained.

## Train

From the repository root, run one or both variants:

```powershell
python .\data\train_variants.py --mode full --model .\models\llama-3.2-3b-distilled --output .\data\full_distilled
python .\data\train_variants.py --mode last_layer --model .\models\llama-3.2-3b-last-layer --output .\data\last_layer_distilled
```

The default input is `data/train_data.jsonl`; validation data is not used by
the trainer. In `last_layer` mode, only `lm_head` is trainable.

Training writes a resumable checkpoint to `data/<variant>/checkpoints/last.ckpt`,
CSV logs under `data/<variant>/logs/`, a loss history CSV, and final model files
under `data/<variant>/final/`. Re-running the same command resumes automatically.
Use `--resume none` to deliberately restart. For a bounded run, add for example:

```powershell
python .\data\train_variants.py --mode full --model .\models\llama-3.2-3b-distilled --output .\data\full_distilled --max-time 02:00:00
```

## Evaluate And Compare

After the final model directories exist:

- Run the model chat widget cell to compare the untrained, full-distilled, and
  last-layer models interactively.
- Run the loss-plot cell to create `training_loss_curves.png`.
- Run the validation cell to evaluate all three models by difficulty and
  function type. It requires `data/val_data.jsonl` and both trained final
  directories.

In the notebook these plots are hardcoded to be certain runs, update with fitting paths.

