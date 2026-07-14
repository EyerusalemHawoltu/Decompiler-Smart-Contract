#!/bin/bash
#SBATCH --job-name=cfg_ds
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_ds_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_ds_%j.err
echo START; date
export CFG_OUT=/scratch/eh3115/Decompliler/Decompliler/datasets/cfg_dataset/cfg.jsonl
export CFG_WORKERS=16
cd /scratch/eh3115/Decompliler/Decompliler
/home/eh3115/.conda/envs/nova/bin/python run_cfg_dataset.py
echo END; date
