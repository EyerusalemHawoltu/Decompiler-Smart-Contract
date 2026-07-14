#!/bin/bash
#SBATCH --job-name=nova_cfg
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/nova_cfg_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/nova_cfg_%j.err
echo START; date; hostname
module purge; module load cuda/11.8.0
export TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/eh3115/hf_cache TRANSFORMERS_CACHE=/scratch/eh3115/hf_cache CUDA_VISIBLE_DEVICES=0
export MODEL_SAVE=/scratch/eh3115/Decompliler/Decompliler/checkpoints/nova-cfg-1.3b
export DATA_DIR=/scratch/eh3115/Decompliler/Decompliler/datasets/cfg_dataset/nova-cfg
cd /scratch/eh3115/Decompliler/Decompliler/nova
/home/eh3115/.conda/envs/nova/bin/python finetune_full.py
echo END; date
