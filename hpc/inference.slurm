#!/bin/bash
#SBATCH --job-name=nova_solidity_infer
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err

echo "===== JOB START ====="
date; hostname
module purge
module load cuda/11.8.0
PYTHON=/home/eh3115/.conda/envs/nova/bin/python
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p /scratch/$USER/hf_cache

PROJECT_ROOT=/scratch/eh3115/Decompliler/Decompliler
CHECKPOINT="$PROJECT_ROOT/checkpoints/nova-solidity-1.3b/checkpoint-24060"
echo "Using checkpoint: $CHECKPOINT"

nvidia-smi
mkdir -p "$PROJECT_ROOT/results"

echo "===== Running inference on held-out TEST set ====="
cd "$PROJECT_ROOT/nova"

$PYTHON inference_solidity.py \
    --model_path "$CHECKPOINT" \
    --input_json "$PROJECT_ROOT/data-tokenized/test_set.json" \
    --output_json "$PROJECT_ROOT/results/inference_test.json" \
    --n_samples 500 \
    --num_return_sequences 1

echo "===== Running evaluation ====="
$PYTHON evaluate_solidity.py \
    --results_json "$PROJECT_ROOT/results/inference_test.json" \
    --output_csv   "$PROJECT_ROOT/results/scores_test.csv"

echo "===== JOB END ====="
date
