#!/bin/bash
#SBATCH --job-name=cfg_080
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_080_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval/cfg_080_%j.err

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler
cd $PROJECT
echo "Running eval_cfg_csv.py for version 0.8.0 only"
$PYTHON eval_cfg_csv.py --version 0.8.0 --workers 16
echo "Done. Checking output:"
ls -lh $PROJECT/results/cfg_eval/0.8.0_results.csv
echo "Row count: $(wc -l < $PROJECT/results/cfg_eval/0.8.0_results.csv)"
