# Usage Guide

## 🎯 Complete Workflow

### 1. Generate and Train NIAH

```bash
# Generate 10K samples with 512 char context
cd scripts
python generate_niah_data.py \
  --num_samples 10000 \
  --context_length 512 \
  --needle_type_k words \
  --needle_type_v numbers \
  --output_dir ../data/niah \
  --output_name niah_10k.json

# Train model
python train_niah.py \
  --data_path ../data/niah/niah_10k.json \
  --output_dir ../models/niah_model \
  --epochs 3 \
  --learning_rate 2e-5 \
  --gpu_id 0
```

### 2. Generate and Train QA

```bash
# Generate 1K samples
cd scripts
python generate_qa_data.py \
  --num_samples 1000 \
  --context_length 512 \
  --add_distractors \
  --output_dir ../data/qa \
  --output_name qa_1k.json

# Train model
python train_qa.py \
  --data_path ../data/qa/qa_1k.json \
  --output_dir ../models/qa_model \
  --epochs 3 \
  --learning_rate 5e-5 \
  --gpu_id 1
```

### 3. Evaluate Results

```bash
# Evaluate individual tasks
python evaluate_model.py \
  --niah_results ../models/niah_model/final_results.json \
  --qa_results ../models/qa_model/final_results.json

# Compare tasks
python evaluate_model.py --compare
```

## ⚡ Quick Start (One Command)

```bash
# NIAH
./run_niah.sh 0  # GPU 0

# QA
./run_qa.sh 1    # GPU 1
```

## 🔧 Advanced Configuration

### Custom NIAH Dataset

```bash
python generate_niah_data.py \
  --num_samples 5000 \
  --context_length 1024 \
  --needle_type_k uuids \
  --needle_type_v uuids \
  --depth_distribution random \
  --seed 123 \
  --output_dir ../data/niah \
  --output_name niah_custom.json
```

### Custom Training

```bash
python train_niah.py \
  --data_path ../data/niah/niah_custom.json \
  --output_dir ../models/niah_custom \
  --epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --grad_accum_steps 2 \
  --hidden_size 1024 \
  --num_layers 12 \
  --gpu_id 2
```

## 📊 Monitoring Training

```bash
# Watch training logs
tail -f ../models/niah_model/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor progress
tail -f ../models/niah_model/training.log | grep "Step\|Eval"
```

## 🎯 Expected Results

### NIAH Task
- **Easy** (context=256, simple values): 30-50% accuracy expected
- **Medium** (context=512, numbers): 10-30% accuracy expected
- **Hard** (context=1024, UUIDs): <10% accuracy expected

### QA Task
- **With distractors** (context=512): 40-60% accuracy expected
- **Without distractors**: 60-80% accuracy expected

## 💡 Tips

1. **Start small**: Test with 100 samples first
2. **Monitor GPU**: Use `nvidia-smi` to check memory
3. **Adjust LR**: If loss plateaus, try higher LR
4. **Check examples**: Look at predictions to debug
5. **Save checkpoints**: Resume from best performing step

## 🐛 Troubleshooting

**Problem**: Out of memory
**Solution**: Reduce batch_size or use gradient_checkpointing

**Problem**: Accuracy = 0%
**Solution**: 
- Increase learning rate
- Train longer
- Simplify task (fewer unique values)

**Problem**: Loss not decreasing
**Solution**:
- Check learning rate (too high/low)
- Check gradient clipping
- Verify data quality

## 📝 File Locations

After training, find:
- Checkpoints: `models/{task}_model/checkpoint_step_*/`
- Best model: `models/{task}_model/best_model/`
- Results: `models/{task}_model/final_results.json`
- Logs: Training output (stdout/stderr)

