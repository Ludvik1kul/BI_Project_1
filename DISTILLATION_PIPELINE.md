# Knowledge Distillation Pipeline: DeepSeek → Llama
## Derivative Learning through Teacher-Student Framework

---

## Overview

This pipeline implements **knowledge distillation** where:
- **Teacher (DeepSeek)**: Generates high-quality solutions for mathematical derivative problems
- **Student (Llama 3.2 3B)**: Learns to solve similar problems through fine-tuning on teacher outputs
- **Evaluation**: Before/after fine-tuning comparison to measure learning improvement

---

## Architecture

### 1. **MathFunctionGenerator**
Generates mathematical functions of varying difficulty:
- **Easy**: Linear (degree 1-2) and simple geometric functions
- **Medium**: Polynomials (degree 2-4), trigonometric compositions
- **Hard**: High-degree polynomials (4-6), complex compositions

**Functions Generated:**
- Polynomials: $f(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_0$
- Trigonometric: $f(x) = \sin(x), \cos(x), \tan(x)$, with compositions
- Exponential: $f(x) = b^x, e^{ax}$
- Logarithmic: $f(x) = \ln(x), \log(x+c)$
- Composite: Products and sums of above

### 2. **PromptGenerator**
Creates varied prompts with "noise" for the same function:
- **Prompt Variations**: "Find", "Compute", "Derive", "Calculate", "Determine"
- **Noise Addition**: Random phrase additions like "(Show work)", "(Step by step)", etc.
- **Two Types**:
  - **Symbolic**: Request symbolic derivative output in LaTeX format
  - **Numerical**: Request numerical derivative at specific point

### 3. **TrainingDataGenerator**
Orchestrates data creation via DeepSeek API:
- Generates 200 examples (configurable)
- Distributed across difficulty levels and derivation types
- Uses DeepSeek to generate ground truth labels
- Saves in **Alpaca format** for fine-tuning

### 4. **EvaluationFramework**
Evaluates Llama's performance:
- **Before Fine-tuning**: Baseline performance on validation set
- **After Fine-tuning**: Performance after training
- Keeps evaluation set hidden from model during training
- Saves results with timestamps for analysis

### 5. **FineTuningSetup**
Configures full fine-tuning (no LoRA):
- Generates standalone Python script
- Configuration managed via JSON
- Ready to run independently

---

## Data Format (Alpaca)

Training data stored as JSONL in this format:

```json
{
  "id": 1,
  "instruction": "Find the derivative of: f(x) = 2*x**2 + 3*x",
  "input": "2*x**2 + 3*x",
  "output": "4*x + 3",
  "difficulty": "easy",
  "derivation_type": "symbolic",
  "function_type": "polynomial",
  "split": "train"
}
```

**Files Generated:**
- `training_data.jsonl` - Full dataset (200 examples)
- `train_data.jsonl` - Training split (~160 examples, 80%)
- `val_data.jsonl` - Validation split (~40 examples, 20%)
  - **Note**: Validation split kept separate, NOT revealed to model during training

---

## Execution Workflow

### Step 1: Generate Training Data
```python
data_gen = TrainingDataGenerator(deepseek, num_examples=200)
dataset = data_gen.generate_dataset()  # ~200 API calls
train_path, val_path, full_path = data_gen.save_to_file()
```
**Output**: Saves to `./data/` folder
- Creates 200 examples with varied prompts
- Each example includes both symbolic and numerical derivatives
- Split: 80% train, 20% validation (hidden)

### Step 2: Evaluate Baseline Llama
```python
evaluator = EvaluationFramework(pipe, "./data/val_data.jsonl")
results_before = evaluator.evaluate_before_finetuning(sample_size=5)
```
**Purpose**: Establish baseline performance before any training
- Tests on 5 random validation examples
- Captures Llama's initial capability
- Used for comparison after fine-tuning

### Step 3: Generate Fine-tuning Script
```python
script_path = finetuning_setup.generate_training_script()
# Output: ./data/fine_tuning_script.py
```
**Configuration Used:**
- Learning Rate: `2e-4`
- Epochs: `3`
- Batch Size: `4` (per device)
- Gradient Accumulation: `2`
- Warmup Steps: `100`
- Max Sequence Length: `512`
- FP16: Enabled (memory efficient)

### Step 4: Run Fine-tuning (External)
```bash
cd ./data
python fine_tuning_script.py
```
**Duration**: ~30-60 minutes on GPU (varies by hardware)
**Output**: Fine-tuned model saved to `./data/fine_tuned_model/`

### Step 5: Evaluate Fine-tuned Model
```python
# Load fine-tuned model
fine_tuned_pipe = pipeline(
    "text-generation",
    model="./data/fine_tuned_model/final",
    device_map="auto",
)

# Create new evaluator with fine-tuned model
evaluator_new = EvaluationFramework(fine_tuned_pipe, "./data/val_data.jsonl")
results_after = evaluator_new.evaluate_after_finetuning(sample_size=5)
evaluator_new.save_results("./data/evaluation_results.json")
```
**Purpose**: Measure improvement from fine-tuning

---

## Data Exploration

### View Sample Examples
```python
explore_training_data(data_path="./data/training_data.jsonl", num_samples=5)
```

### Print Dataset Statistics
```python
stats = print_dataset_statistics(data_path="./data/training_data.jsonl")
```

---

## Expected Outputs

### Before Fine-tuning (Baseline)
- Llama may generate generic or incomplete derivatives
- Inconsistent mathematical notation
- High variance in output quality

### After Fine-tuning (Improved)
- More accurate derivative calculations
- Consistent mathematical formatting (LaTeX)
- Better understanding of function decomposition
- Higher quality outputs on validation set

---

## File Structure

```
BI_Project_1/
├── pipeline.ipynb          # Main notebook with all components
├── DISTILLATION_PIPELINE.md # This documentation
├── models/
│   └── llama-3.2-3b/       # Llama model weights
├── data/
│   ├── training_data.jsonl  # Full dataset (200 examples)
│   ├── train_data.jsonl     # Training split (80%)
│   ├── val_data.jsonl       # Validation split (20%, hidden from training)
│   ├── fine_tuning_script.py # Standalone fine-tuning script
│   ├── evaluation_results.json # Before/after comparison
│   └── fine_tuned_model/    # Directory for fine-tuned weights
└── requirements.txt        # Project dependencies
```

---

## Key Design Decisions

### 1. **Alpaca Format**
- Standard format for LLM fine-tuning
- Easy to extend with new examples
- Compatible with HuggingFace Trainer

### 2. **Separate Symbolic/Numerical Prompts**
- NOT mixed in same prompt (as per requirements)
- Allows model to specialize per task type
- Better training signal

### 3. **Hidden Validation Set**
- Validation split kept separate (not revealed to model during training)
- Labeled with "split" field for later analysis
- Enables true before/after comparison

### 4. **Prompt Variations & Noise**
- Multiple phrasings for same function (robustness)
- Random "noise" additions (generalization)
- Models different student interaction patterns

### 5. **Full Fine-tuning (Not LoRA)**
- As requested: full parameter fine-tuning
- More flexibility for learning new derivative patterns
- Trade-off: higher memory usage (but manageable for 3B model)

---

## Configuration Details

### Function Distribution
- **Difficulty**: Easy (30%), Medium (40%), Hard (30%)
- **Derivation Type**: Symbolic (50%), Numerical (50%)
- **Function Type**: Polynomial (50%), Geometric (50%)

### Prompt Variations (7 symbolic + 5 numerical templates)
Each template has 5 possible noise additions:
- "" (no noise)
- " (Show work)"
- " (Step by step)"
- " (Be precise)"
- " Express in simplified form."

**Total prompt variations**: 12 templates × 5 noise options = 60+ unique prompts

### Training Hyperparameters
```python
{
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": True
}
```

---

## API Requirements

### OpenRouter/DeepSeek API
- Model: `deepseek/deepseek-v4-flash`
- API Key: Set in environment variable `OPENROUTER_API_KEY`
- Cost: ~$0.5-1.0 for full dataset (200 examples)

### Local Llama Model
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Hardware: GPU with 8GB+ VRAM (RTX 3060 or better)
- Alternative: CPU mode (slower, for testing only)

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch size or gradient accumulation steps in `FineTuningSetup.config`

### Issue: Slow API Calls
**Solution**: Check OpenRouter rate limits, add delays between calls

### Issue: Validation Set Mismatch
**Cause**: Model sees validation set during training
**Solution**: Ensure validation set is only used for `eval_steps` in trainer

### Issue: Low Improvement After Training
**Possible Causes**:
- Too few training examples (200 is minimum)
- Learning rate too low
- Model capacity too limited for task

**Solutions**:
- Increase to 500+ examples
- Increase learning rate to 5e-4
- Use larger base model (7B or 13B Llama)

---

## Future Improvements

1. **LoRA Fine-tuning**: More efficient parameter updates
2. **Larger Datasets**: 500-1000 examples for better learning
3. **Multiple Evaluation Metrics**: BLEU, ROUGE, mathematical correctness
4. **Automated Testing**: Symbolic verification of derivatives
5. **Model Ensembling**: Combine predictions from multiple fine-tuned versions
6. **Active Learning**: Prioritize hard examples for next iteration

---

## Citation & References

This pipeline combines:
- **Knowledge Distillation**: Hinton et al. (2015)
- **Supervised Fine-tuning**: Alpaca format (Stanford)
- **Mathematical Domain Adaptation**: Specialized prompts for calculus

---

**Created**: 2026-09-01
**Status**: Ready for deployment
**Author**: Knowledge Distillation Pipeline Generator
