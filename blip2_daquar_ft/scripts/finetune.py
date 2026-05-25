# pip install -U transformers accelerate bitsandbytes peft datasets pillow

"""
finetune.py — Fine-tune BLIP-2 (blip2-opt-2.7b) on a local VQA dataset
              using QLoRA (4-bit base + LoRA adapters).

Architecture note
-----------------
LoRA is applied to BOTH the Q-Former attention layers and the OPT language
model layers, consistent with the thesis discussion in Section 2.4 which
identifies the Q-Former as the primary adaptation target while the OPT
language model's attention layers benefit from joint adaptation.

Reproducibility
---------------
Full reproducibility is achieved by:
  1. set_seed(SEED) — covers torch, numpy, and Python random via HuggingFace.
  2. random.seed(SEED) / numpy seed — belt-and-braces for any non-HF code paths.
  3. Shuffling the JSONL dataset with the same fixed SEED before splitting,
     so train and eval sets are balanced and identical across runs.
  4. TrainingArguments(seed=SEED) — seeds the Trainer's internal DataLoader.
  5. seed_worker + Generator passed to DataLoader for worker-level seeding.
  6. torch.backends.cudnn.deterministic = True — deterministic CUDA kernels.

Usage
-----
    python finetune.py
"""

import os
import json
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# Configuration
# ============================================================
MODEL_ID         = "Salesforce/blip2-opt-2.7b"
DATA_PATH        = "dataset/data.jsonl"
OUTPUT_DIR       = "outputs/lora_adapter"
USE_4BIT         = True

# Sequence length — 128 tokens comfortably covers DAQUAR questions (~9 words)
# plus the short single-word or short-phrase answers.
MAX_SEQ_LEN      = 128

# Epochs-based training.
# 2 epochs over 10 000 samples = 5000 optimiser steps at effective batch 4.
NUM_EPOCHS       = 2

# Effective batch size = PER_DEVICE_BATCH * GRAD_ACCUM = 2 * 2 = 4.
PER_DEVICE_BATCH = 2
GRAD_ACCUM       = 2

# LoRA hyperparameters.
# r=16 gives sufficient capacity for 10 000 training samples.
# alpha = 2 * r is the standard convention (effective LR scale = alpha/r = 2).
# Dropout=0.05 provides light regularisation appropriate for this dataset size.
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# Optimiser.
# 2e-4 is the canonical LoRA learning rate. Cosine decay with 3% warmup
# prevents large gradient updates in the first steps.
LR               = 2e-4
WARMUP_RATIO     = 0.03
WEIGHT_DECAY     = 0.01

# Logging / saving.
LOGGING_STEPS    = 50
SAVE_STEPS       = 250

# Split sizes (80/10/10% train/eval/test) (applied after shuffling).
TRAIN_SIZE       = 10_000
EVAL_SIZE        = 1_234
TEST_SIZE        = 1_234

# DataLoader workers.
# unusable, ignore
NUM_WORKERS      = 0

# Master reproducibility seed — used everywhere.
SEED             = 42
# ============================================================


# ── Full reproducibility setup ────────────────────────────────

def set_full_seed(seed: int) -> None:
    """
    Set all relevant random seeds to guarantee reproducibility.
    Covers: Python random, NumPy, PyTorch (CPU + CUDA), and HuggingFace.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA kernels. This may slightly reduce throughput but
    # ensures identical results across runs on the same hardware.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # HuggingFace covers transformers + datasets internal RNG.
    set_seed(seed)


set_full_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Worker seeding for DataLoader ─────────────────────────────

def seed_worker(worker_id: int) -> None:
    """
    Called by each DataLoader worker process at initialisation.
    Ensures that data augmentation and any stochastic operations
    inside __getitem__ are reproducible across workers and runs.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ── Quantisation config ───────────────────────────────────────

quant_config: Optional[BitsAndBytesConfig] = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # NormalFloat4 — optimal for Gaussian weights
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in fp16, store in 4-bit
        bnb_4bit_use_double_quant=True,        # double quantisation reduces memory
    )

# ── Model loading ─────────────────────────────────────────────

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)

# Freeze all base model parameters before applying LoRA.
for p in model.parameters():
    p.requires_grad = False

if USE_4BIT:
    # prepare_model_for_kbit_training:
    #   - Casts LayerNorm weights to fp32 (required for stable 4-bit training)
    #   - Enables gradient checkpointing to reduce activation memory
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

# ── LoRA configuration ────────────────────────────────────────
#
# Target modules are split into two groups:
#
#   Q-Former (primary adaptation target):
#     "query", "key", "value"  — self-attention and cross-attention projections
#     "dense"                  — output projection in each Q-Former attention block
#
#   OPT language model (secondary adaptation target):
#     "q_proj", "v_proj" — attention projections in OPT layers, required to establish
#                          gradient path to Q-Former through the frozen LM; feed-forward

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # Q-Former attention (self-attention + cross-attention)
        "query", "key", "value", "dense",
        # OPT attention — required to establish gradient path to Q-Former
        # through the frozen LM; feed-forward layers are left untouched
        "q_proj", "v_proj",
    ],
)
model = get_peft_model(model, lora_cfg)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable / 1e6:.2f}M")
print(f"Total params     : {total / 1e6:.2f}M")
print(f"Trainable ratio  : {100 * trainable / total:.3f}%")


# ── Dataset ───────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


class VQADataset(Dataset):
    """
    Dataset for BLIP-2 VQA fine-tuning.

    Each record must contain: image_path, question, answer.

    Prompt format (identical in training, evaluation, and inference):
        "Question: <question> Answer: <answer><eos>"

    The language modelling loss is computed only on the answer tokens.
    Prompt tokens are masked with -100 so the model learns to generate
    the answer given the visual context and the question, not to
    memorise the prompt template itself.
    """

    def __init__(self, records: list, processor, max_len: int = 128):
        self.items     = records
        self.processor = processor
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.items[idx]

        image_path = rec.get("image_path")
        if image_path is None:
            raise ValueError(f"Missing 'image_path' at index {idx}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path} (index {idx})")

        if not rec.get("question") or not rec.get("answer"):
            raise ValueError(f"Missing 'question' or 'answer' at index {idx}: {rec}")

        image = Image.open(image_path).convert("RGB")

        prompt    = f"Question: {rec['question']} Answer:"
        target    = rec["answer"]
        full_text = prompt + " " + target + self.processor.tokenizer.eos_token

        # Tokenise the full sequence (prompt + answer + EOS) with padding.
        enc_full = self.processor.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenise the prompt alone to determine how many tokens to mask.
        # No padding needed — we only need the token count.
        enc_prompt = self.processor.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc_full.input_ids[0]
        attn_mask = enc_full.attention_mask[0]
        labels    = input_ids.clone()

        # Mask all prompt tokens: loss is computed only on answer tokens.
        prompt_len = enc_prompt.input_ids.shape[1]
        labels[:prompt_len] = -100

        # Also mask padding tokens so they do not contribute to the loss.
        labels[attn_mask == 0] = -100

        # Extract pixel values via the image processor.
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attn_mask,
            "labels":         labels,
        }


@dataclass
class Collator:
    """Stack a list of sample dicts into a batch dict of tensors."""

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


# ── Data loading and splitting ────────────────────────────────

all_records = load_jsonl(DATA_PATH)
print(f"Total records loaded: {len(all_records)}")

if len(all_records) < TRAIN_SIZE + EVAL_SIZE + TEST_SIZE:
    raise ValueError(
        f"Dataset has only {len(all_records)} records but "
        f"TRAIN_SIZE + EVAL_SIZE + TEST_SIZE = "
        f"{TRAIN_SIZE + EVAL_SIZE + TEST_SIZE}."
    )

rng = random.Random(SEED)
rng.shuffle(all_records)

train_records = all_records[:TRAIN_SIZE]
eval_records  = all_records[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE]
test_records  = all_records[TRAIN_SIZE + EVAL_SIZE : TRAIN_SIZE + EVAL_SIZE + TEST_SIZE]

# Save test split indices to a file so evaluate.py can load
# exactly the same records without needing to know the split sizes.
test_split_path = os.path.join(os.path.dirname(DATA_PATH), "test_split.jsonl")
with open(test_split_path, "w", encoding="utf-8") as f:
    for rec in test_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Test split saved to  : {test_split_path}")
print(f"Train samples        : {len(train_records)}")
print(f"Eval samples         : {len(eval_records)}")
print(f"Test samples         : {len(test_records)}")

train_ds = VQADataset(train_records, processor, max_len=MAX_SEQ_LEN)
eval_ds  = VQADataset(eval_records,  processor, max_len=MAX_SEQ_LEN)

# load one sample to verify the pipeline before training starts.
_sample = train_ds[0]
print(f"Sample keys          : {list(_sample.keys())}")
print(f"pixel_values shape   : {_sample['pixel_values'].shape}")
print(f"input_ids shape      : {_sample['input_ids'].shape}")
print(f"Non-masked label toks: {(_sample['labels'] != -100).sum().item()}")
del _sample


# ── Training arguments ────────────────────────────────────────

# A reproducible Generator for the Trainer's internal DataLoader.
_dl_generator = torch.Generator()
_dl_generator.manual_seed(SEED)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Reproducibility — must be set here in addition to set_full_seed()
    # because Trainer creates its own DataLoader internally.
    seed=SEED,
    data_seed=SEED,

    # Batch size and gradient accumulation.
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH * 2,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Training schedule.
    num_train_epochs=NUM_EPOCHS,

    # Optimiser.
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",

    # Mixed precision (fp16 raised exception, because of this bf16 is used instead).
    bf16=True,

    # DataLoader settings.
    # pin_memory moves CPU tensors to page-locked memory for faster GPU transfer.
    # group_by_length reduces padding by batching sequences of similar length.
    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=True,
    group_by_length=True,

    # Logging and checkpointing.
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="tensorboard",
)


# ── Trainer ───────────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=Collator(),
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save LoRA adapter weights and processor configuration.
# Only the adapter weights (A and B matrices) are saved — not the full model.
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")