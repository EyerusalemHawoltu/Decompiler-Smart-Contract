#!/bin/bash
#SBATCH --job-name=nova_solidity_prep
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=prep_dataset_%j.out
#SBATCH --error=prep_dataset_%j.err

echo "===== JOB START ====="
date
hostname
pwd

module purge
module load cuda/11.8.0

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
echo "Python:"
$PYTHON --version

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache
mkdir -p /scratch/$USER/hf_cache

PROJECT_ROOT=/scratch/eh3115/Decompliler/Decompliler

echo "Creating output directories..."
mkdir -p "$PROJECT_ROOT/data-tokenized/nova-solidity-bcr"
mkdir -p "$PROJECT_ROOT/checkpoints/nova-solidity-1.3b"
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/hpc/logs"

echo "===== Preparing Solidity dataset ====="
cd "$PROJECT_ROOT/nova"
$PYTHON prepare_solidity_dataset.py

echo "===== DONE ====="
date
