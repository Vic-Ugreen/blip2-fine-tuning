#!/bin/bash
################################################################################
# train_blip2.sh — Fine-tune BLIP-2 on GQA with 1× NVIDIA H200 on PERUN.
#
#
# Submit   : sbatch train_blip2.sh
# Monitor  : tail -f train_blip2_<JOBID>.out
#            tail -f train_blip2_<JOBID>.err
# Cancel   : scancel <JOBID>
# Status   : squeue -u $USER
################################################################################
#SBATCH --job-name=blip2_gqa_finetune
#SBATCH --output=train_blip2_%j.out
#SBATCH --error=train_blip2_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00


echo "======================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Scratch   : $SCRATCH_DIR"
echo "Started."
echo "======================================================"

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

# Parallelism — match OMP threads to allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Disable tokeniser parallelism to avoid fork warnings with DataLoader workers
export TOKENIZERS_PARALLELISM=false
# Keep HuggingFace model cache on NFS (models download once, persist)
export TRANSFORMERS_CACHE=~/hf_cache
export HF_HOME=~/hf_cache
mkdir -p "$TRANSFORMERS_CACHE"

echo ""
echo "Python    : $(which python)"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo "OMP       : $OMP_NUM_THREADS threads"
echo ""

# Verify GPU
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader
echo ""

# Training
python finetune.py

EXIT_CODE=$?

echo ""
echo "======================================================"
echo "Finished."
echo "Exit code : $EXIT_CODE"
echo "======================================================"

echo "Results will appear in: ~/results_job_${SLURM_JOB_ID}/"

exit $EXIT_CODE