#!/bin/bash
#SBATCH --job-name=nova_eval
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/nova_eval_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/nova_eval_%j.err

echo "===== JOB START =====" && date && hostname
module purge
module load cuda/11.8.0
PYTHON=/home/eh3115/.conda/envs/nova/bin/python
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/eh3115/hf_cache
export TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache
export CUDA_VISIBLE_DEVICES=0

cd /scratch/eh3115/Decompliler/Decompliler/nova
$PYTHON run_full_eval.py

echo "===== JOB END =====" && date
