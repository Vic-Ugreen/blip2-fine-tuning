#!/bin/bash
################################################################################
# inference_monitor.sh — GPU-monitored inference on GQA testdev samples.
#
# Loads baseline and fine-tuned BLIP-2 sequentially, runs inference on
# N_SAMPLES testdev images, and records GPU utilisation / VRAM / temperature
# 
# Submit  : sbatch inference_blip2.sh
# Monitor : tail -f inference_monitor_<JOBID>.out
################################################################################
#SBATCH --job-name=infer_monitor
#SBATCH --output=inference_monitor_%j.out
#SBATCH --error=inference_monitor_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00

exec 2>&1


echo "======================================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started."
echo "======================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=~/hf_cache
export HF_HOME=~/hf_cache

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python inference.py

EXIT_CODE=$?

echo ""
echo "======================================================"
echo "Finished."
echo "Exit code : $EXIT_CODE"
echo "Logs      : outputs/inference_monitor/"
echo "======================================================"

exit $EXIT_CODE