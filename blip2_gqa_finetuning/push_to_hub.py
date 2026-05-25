"""
push_to_hub.py — Push the final BLIP-2 fine-tuned model to HuggingFace Hub.

Pushes only the final model files from the output directory root.
Skips intermediate training checkpoints, TensorBoard logs, and GPU monitor logs
— those are not needed for inference and would waste quota unnecessarily.

What gets pushed
----------------
  config.json, generation_config.json
  model.safetensors / model-*.safetensors shards
  preprocessor_config.json
  tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.json
  merges.txt (OPT tokenizer)

What gets skipped
-----------------
  checkpoint-*/        intermediate training checkpoints (~15 GB each)
  runs/                TensorBoard training logs
  gpu_monitor/         TensorBoard GPU logs
  test_data_path.txt   HPC-specific path reference
  *.out / *.err        SLURM log files

Usage
-----
  Run on HPC login node (no GPU needed — this is just file transfer):
    conda activate env_name
    huggingface-cli login        # paste your HF write token when prompted
    python push_to_hub.py
"""

import os
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_DIR = "/mnt/data/home/viuhke649/gqa_finetuning/outputs/blip2_gqa_full"
REPO_ID   = "Ugr1N/blip2-gqa-finetuned"
PRIVATE   = False    # set True if you want the model repo private

api = HfApi()

print(f"Creating repository: {REPO_ID}")
create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,
    private=PRIVATE,
)
print("Repository ready.")

print(f"\nUploading model files from: {MODEL_DIR}")
print("Skipping checkpoints, TensorBoard logs, and SLURM output files...")

api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    ignore_patterns=[
        # Intermediate training checkpoints — large and not needed for inference
        "checkpoint-*",
        # TensorBoard and GPU monitoring logs — already copied locally
        "runs",
        "runs/**",
        "gpu_monitor",
        "gpu_monitor/**",
        # HPC-specific files
        "test_data_path.txt",
        "*.out",
        "*.err",
        # Python cache
        "__pycache__",
        "**/__pycache__",
        "*.pyc",
    ],
    commit_message="Upload fine-tuned BLIP-2 (GQA balanced, full BF16)",
)

print(f"\nDone. Model available at: https://huggingface.co/{REPO_ID}")