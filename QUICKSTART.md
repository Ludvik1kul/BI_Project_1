# Quick Start Guide - Knowledge Distillation Pipeline

## What You Have

A complete knowledge distillation pipeline with:
- ✓ DeepSeek API integration (teacher model)
- ✓ Llama 3.2 3B locally loaded (student model)
- ✓ Mathematical function generator
- ✓ Prompt generator with variations and noise
- ✓ Training data orchestrator
- ✓ Evaluation framework
- ✓ Fine-tuning configuration

## How to Use

### 1. Generate Training Data (200 examples)
In **Cell 6** of pipeline.ipynb, uncomment and run:
```python
data_gen = TrainingDataGenerator(deepseek, num_examples=200)
dataset = data_gen.generate_dataset()
train_path, val_path, full_path = data_gen.save_to_file()
```

**What happens:**
- Makes ~200 API calls to DeepSeek
- Creates varied prompts for mathematical functions
- Saves to `./data/`:
  - `training_data.jsonl` (full dataset)
  - `train_data.jsonl` (80% for training)
  - `val_data.jsonl` (20% for validation - **kept hidden from model**)
- **Time**: ~5-10 minutes
- **Cost**: ~$0.5-1.0

### 2. Evaluate Baseline (Optional but Recommended)
In **Cell 7**, uncomment and run:
```python
evaluator = EvaluationFramework(pipe, "./data/val_data.jsonl")
results_before = evaluator.evaluate_before_finetuning(sample_size=5)
```

**What happens:**
- Tests Llama on 5 random examples before any training
- Shows current (untrained) performance
- Provides baseline for comparison after fine-tuning
- **Time**: <1 minute

### 3. Generate Fine-tuning Script
In **Cell 7**, uncomment and run:
```python
script_path = finetuning_setup.generate_training_script()
```

**What happens:**
- Creates `./data/fine_tuning_script.py`
- Ready to run independently
- No need to modify anything

### 4. Run Fine-tuning (External Execution)
Open terminal and run:
```bash
cd c:\Users\A2610206\OneDrive\ -\ BI\ Norwegian\ Business\ School\ \(BIEDU\)\Documents\kode\Project-1\BI_Project_1
python ./data/fine_tuning_script.py
```

**What happens:**
- Loads training data from `train_data.jsonl`
- Trains Llama on 160 examples for 3 epochs
- Saves fine-tuned model to `./data/fine_tuned_model/`
- Logs progress to `./logs/`
- **Time**: 30-60 minutes (GPU dependent)
- **Memory**: ~8-12 GB VRAM

### 5. Evaluate Fine-tuned Model
After fine-tuning completes, in a new cell add:
```python
# Load fine-tuned model
from transformers import pipeline as hf_pipeline

fine_tuned_pipe = hf_pipeline(
    "text-generation",
    model="./data/fine_tuned_model/final",
    dtype=torch.bfloat16,
    device_map="auto",
)

# Evaluate
evaluator_after = EvaluationFramework(fine_tuned_pipe, "./data/val_data.jsonl")
results_after = evaluator_after.evaluate_after_finetuning(sample_size=5)
evaluator_after.save_results("./data/evaluation_results.json")
```

**What happens:**
- Tests fine-tuned Llama on same 5 examples
- Compares before/after outputs
- Shows improvement from distillation
- Saves detailed results to JSON
- **Time**: <1 minute

---

## Understanding the Data

### Alpaca Format Example
Each training example is JSON with:
```json
{
  "id": 42,
  "instruction": "Find the derivative of: f(x) = 3*x**3 + 2*x**2 (Step by step)",
  "input": "3*x**3 + 2*x**2",
  "output": "9*x**2 + 4*x",
  "difficulty": "easy",
  "derivation_type": "symbolic",
  "function_type": "polynomial",
  "split": "train"
}
```

### Key Points
- **instruction**: What the model is asked to do (with noise)
- **input**: The mathematical function
- **output**: DeepSeek's answer (ground truth)
- **difficulty**: Easy / Medium / Hard
- **derivation_type**: Symbolic OR Numerical (never mixed)
- **function_type**: polynomial / trig / exponential / logarithmic / composite
- **split**: "train" (used for learning) or "val" (hidden for testing)

---

## Explore the Data

### See sample examples
```python
explore_training_data(num_samples=10)
```

### Get statistics
```python
print_dataset_statistics()
```

**Output shows:**
- Total examples
- Train/Val split percentages
- Distribution by difficulty
- Distribution by derivation type
- Function types included

---

## Expected Results

### Before Fine-tuning
- Llama may give generic or incomplete answers
- No clear pattern in derivative computation
- Inconsistent formatting

### After Fine-tuning
- More accurate derivatives
- Consistent mathematical notation
- Better decomposition of complex functions
- Higher accuracy on validation set

### Measurement
- Before: Take note of outputs
- After: Compare to baseline
- Metric: Exact match or similarity to DeepSeek answers

---

## Important Notes

1. **Validation Set is Hidden**: The `val_data.jsonl` file is never shown to the model during training. Only `train_data.jsonl` is used. This ensures fair before/after comparison.

2. **Separate Prompts**: Symbolic and numerical derivatives are in separate prompts. Never mixed.

3. **Prompt Variations**: Each function is presented with different phrasings to increase robustness.

4. **No Training Loop Runs Automatically**: You control when fine-tuning happens (external script).

5. **Results Saved**: All evaluation results are saved to JSON for analysis.

---

## Troubleshooting

### "OpenRouter API error"
- Check OPENROUTER_API_KEY is set
- Verify API quota and rate limits
- Check internet connection

### "CUDA Out of Memory"
- Reduce per_device_train_batch_size to 2
- Set gradient_accumulation_steps to 4
- Use CPU mode (slow) for testing

### "File not found: training_data.jsonl"
- Run data generation first (Step 1)
- Check that .ipynb is run in correct directory

### "Model not found"
- Ensure Llama model is downloaded in ./models/llama-3.2-3b/
- Run the model loading cell first

---

## Next Steps After Fine-tuning

1. ✓ Compare before/after results
2. ✓ Save comparison to `evaluation_results.json`
3. ✓ Analyze which difficulty levels improved most
4. ✓ Consider retraining with more examples if improvement is marginal
5. ✓ Deploy fine-tuned model for production use

---

**Status**: Ready to execute
**Estimated Time**: 
- Data generation: 5-10 min
- Baseline evaluation: <1 min
- Fine-tuning: 30-60 min
- Final evaluation: <1 min
- **Total**: ~45 minutes to 70 minutes

