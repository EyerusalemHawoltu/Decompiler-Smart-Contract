#!/bin/bash
#SBATCH --job-name=cfg_sel
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval_selector/sel_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_eval_selector/sel_%j.err
PROJECT=/scratch/eh3115/Decompliler/Decompliler
mkdir -p $PROJECT/results/cfg_eval_selector
export CFG_EXTRACTOR=selector
export CFG_EVAL_OUT=$PROJECT/results/cfg_eval_selector
cd $PROJECT
echo "=== SELECTOR-ONLY front-end (dispatcher scan + 453k names) ==="
/home/eh3115/.conda/envs/nova/bin/python eval_cfg_csv.py --workers 16
echo "=== version_summary_no_stubs ==="
cat $PROJECT/results/cfg_eval_selector/version_summary_no_stubs.csv
