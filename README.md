# SONAR-LLM Experiments

Professional implementation of SONAR-LLM training on RULER benchmark tasks.

## 📁 Project Structure

```
sonar_llm_experiments/
├── configs/                    # Configuration files
│   ├── niah_config.json       # NIAH task configuration
│   └── qa_config.json         # QA task configuration
├── scripts/                    # Python scripts
│   ├── text_cleaning.py       # Text cleaning utilities (FlagEmbedding-style)
│   ├── generate_niah_data.py  # NIAH dataset generation
│   ├── generate_qa_data.py    # QA dataset generation
│   ├── train_niah.py          # NIAH training script
│   └── train_qa.py            # QA training script
├── data/                       # Generated datasets
│   ├── niah/                  # NIAH datasets
│   └── qa/                    # QA datasets
├── models/                     # Trained models
│   ├── niah_model/            # NIAH checkpoints
│   └── qa_model/              # QA checkpoints
├── results/                    # Evaluation results
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Generate NIAH Dataset

```bash
cd scripts
python generate_niah_data.py \
  --num_samples 10000 \
  --context_length 512 \
  --needle_type_k words \
  --needle_type_v numbers \
  --output_dir ../data/niah \
  --output_name niah_10k_512.json
```

**Parameters:**
- `--num_samples`: Number of samples to generate
- `--context_length`: Target context length in characters
- `--needle_type_k`: Key type (`words`, `numbers`, `uuids`)
- `--needle_type_v`: Value type (`words`, `numbers`, `uuids`)
- `--depth_distribution`: Needle placement (`uniform`, `random`, `fixed`)

### 2. Generate QA Dataset

```bash
cd scripts
python generate_qa_data.py \
  --num_samples 1000 \
  --context_length 512 \
  --add_distractors \
  --output_dir ../data/qa \
  --output_name qa_1k_512.json
```

**Parameters:**
- `--num_samples`: Number of samples
- `--context_length`: Target context length
- `--add_distractors`: Add distractor paragraphs

### 3. Train NIAH Model

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 python train_niah.py \
  --data_path ../data/niah/niah_10k_512.json \
  --output_dir ../models/niah_model \
  --epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --grad_accum_steps 4 \
  --gpu_id 0
```

### 4. Train QA Model

```bash
cd scripts
CUDA_VISIBLE_DEVICES=1 python train_qa.py \
  --data_path ../data/qa/qa_1k_512.json \
  --output_dir ../models/qa_model \
  --epochs 3 \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --grad_accum_steps 4 \
  --gpu_id 1
```

## 📊 RULER Benchmark Metrics

### NIAH Task
- **Metric**: `string_match_all`
- Checks if ALL expected values are present in predictions
- Formula: `(correct_predictions / total) * 100`

### QA Task
- **Metric**: `string_match_part`
- Checks if AT LEAST ONE expected answer is present
- More lenient than `string_match_all`

## 🧹 Text Cleaning

Following [FlagEmbedding approach](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/main/eval_needle.py):

### For Data Generation (RULER-style):
- Remove excessive newlines (`\n\n\n` → `\n\n`)
- Remove excessive spaces
- Remove control characters (`\x00-\x1f`)
- Normalize whitespace

### For Predictions (Flag-style):
- Strip newlines
- Take first line only

## 📈 Training Pipeline

```
1. Data Generation
   ├─> generate_niah_data.py → data/niah/
   └─> generate_qa_data.py → data/qa/

2. Data Processing
   └─> text_cleaning.py (clean_text_ruler_style)

3. Training
   ├─> train_niah.py → models/niah_model/
   └─> train_qa.py → models/qa_model/

4. Evaluation
   └─> RULER metrics (string_match_all, string_match_part)

5. Results
   └─> final_results.json (loss, accuracy, examples)
```

## 🔧 Configuration Files

Edit `configs/*.json` to change hyperparameters without modifying code.

**Example** (modify learning rate):
```json
{
  "training": {
    "learning_rate": 1e-4  // Changed from 2e-5
  }
}
```

## 📝 Output Files

After training, each model directory contains:

```
models/niah_model/
├── checkpoint_step_100/
│   └── model.pt
├── checkpoint_step_200/
├── best_model/
│   └── model.pt           # Best performing checkpoint
└── final_results.json      # Evaluation results
```

**final_results.json** structure:
```json
{
  "loss": 3.0,
  "accuracy": 15.5,
  "num_evaluated": 1000,
  "examples": [
    {
      "key": "key_1234",
      "expected_value": "blue car",
      "prediction": "The value is blue car",
      "correct": true
    }
  ]
}
```

## 🎯 Best Practices

### For NIAH:
- Start with small dataset (1K samples) to test
- Use `uniform` depth distribution
- Learning rate: 2e-5 to 1e-4
- Context length: 256-512 chars for initial experiments

### For QA:
- Smaller dataset (1K samples) sufficient
- Higher learning rate (5e-5 to 1e-4)
- Add distractors to make it harder
- Monitor accuracy - should be >30% after training

## ⚠️ Known Issues

1. **Mixed Precision**: Currently disabled due to GradScaler compatibility issues
2. **Memory**: Precomputing embeddings requires significant RAM
3. **Speed**: ~3-4 it/s during training

## 📚 References

- [RULER Benchmark](https://github.com/hsiehjackson/RULER)
- [FlagEmbedding NIAH](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)
- [SONAR](https://github.com/facebookresearch/SONAR)

## 📧 Contact

For questions about this implementation, refer to the training logs and configuration files.

