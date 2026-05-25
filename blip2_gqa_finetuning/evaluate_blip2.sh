#!/bin/bash
################################################################################
# evaluate_blip2.sh — Evaluate baseline vs fine-tuned BLIP-2 on GQA testdev.
#
# Run AFTER train_blip2.sh completes
#
# Submit  : sbatch evaluate_blip2.sh
# Monitor : tail -f eval_blip2_<JOBID>.out
################################################################################
#SBATCH --job-name=eval_blip2
#SBATCH --output=eval_blip2_%j.out
#SBATCH --error=eval_blip2_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Merge stderr into stdout — everything in one .out file
exec 2>&1


echo "======================================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Scratch : $SCRATCH_DIR"
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

# Verify checkpoint exists before wasting GPU time
CHECKPOINT="outputs/blip2_gqa_full"
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Copy it back from your training results first:"
    echo "  cp -r ~/results_job_<TRAINING_JOBID>/outputs/blip2_gqa_full/ outputs/"
    exit 1
fi
echo "Checkpoint found : $CHECKPOINT"
echo "Checkpoint size  : $(du -sh $CHECKPOINT | cut -f1)"
echo ""

python evaluate.py

EXIT_CODE=$?

echo ""
echo "======================================================"
echo "Finished."
echo "Exit code : $EXIT_CODE"
echo "Results   : ~/results_job_${SLURM_JOB_ID}/"
echo "======================================================"

exit $EXIT_CODE