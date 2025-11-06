#!/bin/bash
#
# Professional training script for NIAH task
# Usage: ./run_niah.sh [GPU_ID]
#

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sonar-llm

# Set GPU
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Configuration
DATA_DIR="data/niah"
MODEL_DIR="models/niah_model"
NUM_SAMPLES=10000
CONTEXT_LENGTH=512
NEEDLE_TYPE_K="words"
NEEDLE_TYPE_V="numbers"
EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=2e-5
GRAD_ACCUM=4

echo "======================================"
echo "SONAR-LLM NIAH Training Pipeline"
echo "======================================"
echo "GPU: $GPU_ID"
echo "Samples: $NUM_SAMPLES"
echo "Context: $CONTEXT_LENGTH chars"
echo "======================================"
echo ""

# Step 1: Generate dataset
echo "[1/2] Generating NIAH dataset..."
cd scripts
python generate_niah_data.py \
  --num_samples $NUM_SAMPLES \
  --context_length $CONTEXT_LENGTH \
  --needle_type_k $NEEDLE_TYPE_K \
  --needle_type_v $NEEDLE_TYPE_V \
  --output_dir ../$DATA_DIR \
  --output_name "niah_${NUM_SAMPLES}_${CONTEXT_LENGTH}.json"

if [ $? -ne 0 ]; then
    echo "✗ Dataset generation failed!"
    exit 1
fi

echo "✓ Dataset generated"
echo ""

# Step 2: Train model
echo "[2/2] Training NIAH model..."
python train_niah.py \
  --data_path ../$DATA_DIR/niah_${NUM_SAMPLES}_${CONTEXT_LENGTH}.json \
  --output_dir ../$MODEL_DIR \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --grad_accum_steps $GRAD_ACCUM \
  --gpu_id $GPU_ID

if [ $? -ne 0 ]; then
    echo "✗ Training failed!"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ NIAH Training Complete!"
echo "======================================"
echo "Model saved to: $MODEL_DIR"
echo "Results: $MODEL_DIR/final_results.json"
echo "======================================"

