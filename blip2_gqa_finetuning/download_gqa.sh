#!/bin/bash
################################################################################
# download_gqa.sh — Download and preprocess all GQA balanced splits to NFS.
#
# Runs as a CPU job.  Does NOT use the scratch system because the dataset
# must persist
#
# No GPU required.
#
# Submit:   sbatch download_gqa.sh
# Monitor:  tail -f download_gqa_<JOBID>.out
#           tail -f download_gqa_<JOBID>.err
################################################################################
#SBATCH --job-name=download_gqa
#SBATCH --output=download_gqa_%j.out
#SBATCH --error=download_gqa_%j.err
#SBATCH --partition=CPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Do NOT source .activate_scratch
# Dataset must go to persistent NFS storage (~/datasets/gqa/)

echo "======================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Started   : $(date)"
echo "======================================================"

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

# Temporary HuggingFace datasets cache
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"

echo ""
echo "Python    : $(which python)"
echo "HF cache  : $HF_DATASETS_CACHE"
echo "Output    : ~/datasets/gqa/"
echo ""

# Download all three splits
python prepare_gqa.py \
    --splits train val testdev \
    --out_dir ~/datasets/gqa

EXIT_CODE=$?

# Cleanup temp cache
echo ""
echo "Cleaning HuggingFace parquet cache..."
rm -rf "$HF_DATASETS_CACHE"

echo ""
echo "======================================================"
echo "Finished  : $(date)"
echo "Exit code : $EXIT_CODE"
echo "======================================================"

# Print dataset sizes for the record
echo ""
echo "Dataset sizes:"
du -sh ~/datasets/gqa/*/images/ 2>/dev/null || true
du -sh ~/datasets/gqa/*/data.jsonl 2>/dev/null || true

exit $EXIT_CODE