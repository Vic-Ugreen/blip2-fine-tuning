"""
count_params.py — Load BLIP-2 OPT-2.7B, apply the HPC unfreezing strategy
                  from finetune_hpc.py, and report trainable vs total parameters.

No GPU is required — the model is loaded on CPU in BFloat16 solely to count
parameters. The unfreezing logic is an exact copy of finetune_hpc.py so the
reported numbers reflect the actual training configuration.

Run via:  sbatch count_params.sh

"""

import torch
from transformers import Blip2ForConditionalGeneration

MODEL_ID        = "Salesforce/blip2-opt-2.7b"
UNFREEZE_LM_FFN = True   # must match finetune_hpc.py

print(f"Loading {MODEL_ID} on CPU in BFloat16...")
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
print("Model loaded.\n")

# Freeze everything first
for p in model.parameters():
    p.requires_grad = False

# Apply the same unfreezing logic as finetune_hpc.py
groups = {
    "Q-Former + query tokens": [],
    "Language projection":     [],
    "OPT attention":           [],
    "OPT feed-forward":        [],
    "Frozen (not trained)":    [],
}

for name, param in model.named_parameters():
    if "qformer" in name or "query_tokens" in name:
        param.requires_grad = True
        groups["Q-Former + query tokens"].append(param.numel())

    elif "language_projection" in name:
        param.requires_grad = True
        groups["Language projection"].append(param.numel())

    elif "language_model" in name:
        is_attn = any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj"))
        is_ffn  = any(k in name for k in ("fc1", "fc2"))

        if is_attn:
            param.requires_grad = True
            groups["OPT attention"].append(param.numel())
        elif is_ffn and UNFREEZE_LM_FFN:
            param.requires_grad = True
            groups["OPT feed-forward"].append(param.numel())
        else:
            groups["Frozen (not trained)"].append(param.numel())
    else:
        groups["Frozen (not trained)"].append(param.numel())

# Compute totals
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
frozen    = total - trainable

# Print results
sep = "=" * 62
print(sep)
print("  BLIP-2 OPT-2.7B — Parameter Count Report")
print(f"  UNFREEZE_LM_FFN = {UNFREEZE_LM_FFN}")
print(sep)

print(f"\n{'Group':<32} {'Parameters':>14} {'% of total':>12}")
print("-" * 62)
for group_name, param_list in groups.items():
    n = sum(param_list)
    pct = 100.0 * n / total
    marker = "  [trainable]" if group_name != "Frozen (not trained)" else "  [frozen]"
    print(f"  {group_name:<30} {n:>14,} {pct:>11.4f}%{marker}")

print("-" * 62)
print(f"  {'Total trainable':<30} {trainable:>14,} {100*trainable/total:>11.4f}%")
print(f"  {'Total frozen':<30} {frozen:>14,} {100*frozen/total:>11.4f}%")
print(f"  {'TOTAL parameters':<30} {total:>14,} {'100.0000%':>12}")
print(sep)

print("\nComparison with DAQUAR LoRA configuration:")
daquar_trainable = 36_380_000   # from fine_tuning.tex Table hyperparameters
daquar_total     = 2_029_860_000
print(f"  DAQUAR LoRA  — trainable: {daquar_trainable:>14,} "
      f"({100*daquar_trainable/daquar_total:.4f}% of {daquar_total:,})")
print(f"  GQA HPC full — trainable: {trainable:>14,} "
      f"({100*trainable/total:.4f}% of {total:,})")
print(f"  Increase factor: {trainable / daquar_trainable:.2f}x more trainable parameters")
print(sep)