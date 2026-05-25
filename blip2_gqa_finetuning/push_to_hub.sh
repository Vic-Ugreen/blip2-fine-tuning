#!/bin/bash
################################################################################
# push_to_hub.sh — Push fine-tuned BLIP-2 to HuggingFace Hub.
#
# Runs as a CPU job on the login node equivalent.
# No GPU needed — this is purely file upload.
#
# BEFORE submitting:
#   1. Get a HuggingFace WRITE token from https://huggingface.co/settings/tokens
#   2. On the login node run:  huggingface-cli login
#      Paste your token when prompted. This saves it to ~/.cache/huggingface/
#      The saved token will be available to the compute node via NFS.
#
# Submit: sbatch push_to_hub.sh
################################################################################
#SBATCH --job-name=push_hf
#SBATCH --output=push_to_hub_%j.out
#SBATCH --error=push_to_hub_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

exec 2>&1

# Do NOT activate scratch — model files are on NFS and must stay there

echo "======================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "======================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

# HuggingFace token is read from ~/.cache/huggingface/ automatically
# (saved there when you ran huggingface-cli login on the login node)
export HF_HOME=~/.cache/huggingface

echo ""
echo "Starting upload to HuggingFace Hub..."
echo ""

python push_to_hub.py

EXIT_CODE=$?

echo ""
echo "======================================================"
echo "Finished : $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================================"

exit $EXIT_CODE