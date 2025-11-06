#!/bin/bash
#
# Professional training script for QA task
# Usage: ./run_qa.sh [GPU_ID]
#

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sonar-llm

# Set GPU
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Configuration
DATA_DIR="data/qa"
MODEL_DIR="models/qa_model"
NUM_SAMPLES=1000
CONTEXT_LENGTH=512
EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=5e-5
GRAD_ACCUM=4

echo "======================================"
echo "SONAR-LLM QA Training Pipeline"
echo "======================================"
echo "GPU: $GPU_ID"
echo "Samples: $NUM_SAMPLES"
echo "Context: $CONTEXT_LENGTH chars"
echo "======================================"
echo ""

# Step 1: Generate dataset
echo "[1/2] Generating QA dataset..."
cd scripts
python generate_qa_data.py \
  --num_samples $NUM_SAMPLES \
  --context_length $CONTEXT_LENGTH \
  --add_distractors \
  --output_dir ../$DATA_DIR \
  --output_name "qa_${NUM_SAMPLES}_${CONTEXT_LENGTH}.json"

if [ $? -ne 0 ]; then
    echo "✗ Dataset generation failed!"
    exit 1
fi

echo "✓ Dataset generated"
echo ""

# Step 2: Train model
echo "[2/2] Training QA model..."
python train_qa.py \
  --data_path ../$DATA_DIR/qa_${NUM_SAMPLES}_${CONTEXT_LENGTH}.json \
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
echo "✓ QA Training Complete!"
echo "======================================"
echo "Model saved to: $MODEL_DIR"
echo "Results: $MODEL_DIR/final_results.json"
echo "======================================"

