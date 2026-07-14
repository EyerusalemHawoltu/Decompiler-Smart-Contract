#!/bin/bash
#SBATCH --job-name=cfg_base
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval_baseline/cfg_base_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval_baseline/cfg_base_%j.err

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler
mkdir -p $PROJECT/results/cfg_eval_baseline

# Pristine evm_cfg_builder (no BFS, no PUSH0, no robustness fixes) + 55k decoded functions
export ECB_DIR=$PROJECT/evm_cfg_builder_pristine
export CFG_EVAL_OUT=$PROJECT/results/cfg_eval_baseline

echo "=== BASELINE: pristine evm_cfg_builder + 55k decoded functions ==="
echo "ECB_DIR=$ECB_DIR"
cd $PROJECT
$PYTHON eval_cfg_csv.py --workers 16

echo "=== version_summary_no_stubs.csv (baseline) ==="
cat $PROJECT/results/cfg_eval_baseline/version_summary_no_stubs.csv
