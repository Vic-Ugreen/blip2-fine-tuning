#!/bin/bash
################################################################################
# count_params.sh — Count trainable vs total parameters of BLIP-2 OPT-2.7B
#                   under the HPC fine-tuning configuration.
#
# No GPU needed — runs on CPU. Loads the model from the HuggingFace cache.
#
# Submit:  sbatch count_params.sh
# Monitor:  tail -f count_params_<JOBID>.out
#           tail -f count_params_<JOBID>.err
################################################################################
#SBATCH --job-name=count_params
#SBATCH --output=count_params_%j.out
#SBATCH --error=count_params_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

exec 2>&1

echo "======================================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "======================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

export TRANSFORMERS_CACHE=~/hf_cache
export HF_HOME=~/hf_cache

python count_params.py

echo ""
echo "======================================================"
echo "Finished : $(date)"
echo "======================================================"