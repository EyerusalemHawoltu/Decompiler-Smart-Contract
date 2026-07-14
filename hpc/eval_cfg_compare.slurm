#!/bin/bash
#SBATCH --job-name=cfg_cmp
#SBATCH --partition=cs
#SBATCH --account=condo_cs_ali
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/eh3115/Decompliler/Decompliler/results/cfg_compare_%j.out
#SBATCH --error=/scratch/eh3115/Decompliler/Decompliler/results/cfg_compare_%j.err

PYTHON=/home/eh3115/.conda/envs/nova/bin/python
PROJECT=/scratch/eh3115/Decompliler/Decompliler
cd $PROJECT
SAMPLE=150

echo "########## BASELINE (pristine + 55k decoded, hard timeout 10s) ##########"
ECB_DIR=$PROJECT/evm_cfg_builder_pristine \
CFG_EVAL_OUT=$PROJECT/results/cfg_eval_baseline_sample \
CFG_HARD_TIMEOUT=10 \
$PYTHON eval_cfg_csv.py --workers 16 --sample $SAMPLE

echo ""
echo "########## V4 (enhanced, same sample) ##########"
CFG_EVAL_OUT=$PROJECT/results/cfg_eval_v4_sample \
$PYTHON eval_cfg_csv.py --workers 16 --sample $SAMPLE

echo ""
echo "===== BASELINE no_stubs ====="
cat $PROJECT/results/cfg_eval_baseline_sample/version_summary_no_stubs.csv
echo ""
echo "===== V4 no_stubs ====="
cat $PROJECT/results/cfg_eval_v4_sample/version_summary_no_stubs.csv
