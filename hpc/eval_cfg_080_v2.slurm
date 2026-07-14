#!/bin/bash
#SBATCH --job-name=cfg_080v2
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=cn271
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_080v2_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_080v2_%j.err

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler
echo "=== JOB START ===" && date && hostname

cd $PROJECT
$PYTHON eval_cfg_csv.py --version 0.8.0 --workers 16

echo "" && echo "=== Output ===" && ls -lh $PROJECT/results/cfg_eval/0.8.0_results.csv 2>/dev/null || echo "0.8.0 MISSING"
echo "Row count: $(wc -l < $PROJECT/results/cfg_eval/0.8.0_results.csv 2>/dev/null || echo 0)"
echo "=== JOB END ===" && date
