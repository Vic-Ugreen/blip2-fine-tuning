"""
finetune_hpc.py — Full fine-tuning of BLIP-2 (blip2-opt-2.7b) on GQA (balanced)
                  for PERUN HPC with a single NVIDIA H200 GPU.

Differences from local finetune.py
------------------------------------
  No quantization   Model loads in pure BF16; no BitsAndBytes dependency.
  No LoRA           Target layers are selectively unfrozen for full-weight updates.
  Larger batches    H200 has 141 GB VRAM; BLIP-2 opt-2.7b
                    for batch_size=32 + grad_accum=2 → effective batch 64.
  grad clipping     max_grad_norm=1.0 clips the spikes.
  More workers      dataloader_num_workers=4 pipelines image I/O.
  Full val          EVAL_SIZE_CAP=None — all 132 K val pairs used at each checkpoint.
  GPU monitoring    logs memory / utilisation / temperature.
  Full checkpoint   Entire model weights saved (not just an adapter).

Layers unfrozen (vision encoder stays fully frozen)
-----------------------------------------------------
  1. Q-Former — qformer.* and query_tokens   (primary adaptation target)
  2. Language projection — language_projection.*   (Q-Former → LM bridge)
  3. OPT attention — q_proj, k_proj, v_proj, out_proj in every decoder layer
  4. OPT feed-forward — fc1, fc2 (optional; controlled by UNFREEZE_LM_FFN)

Usage
-----
  submit via finetune_blip2.sh

Fill in DATA PATHS below after running prepare_gqa.py --out_dir ~/datasets/gqa.
"""

import os
import json
import random
import shutil
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

# Must be set before any HuggingFace tokenizer is imported — suppresses the
# "tokenizer parallelism" fork warning that fires on every DataLoader worker.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pynvml
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
)

# ============================================================
# Configuration
# ============================================================
MODEL_ID    = "Salesforce/blip2-opt-2.7b"

# Fill these in after running prepare_gqa.py --out_dir ~/datasets/gqa
# The script prints the exact paths at the end of each run.
TRAIN_DATA_PATH = os.path.expanduser("~/datasets/gqa/train/data.jsonl")
EVAL_DATA_PATH  = os.path.expanduser("~/datasets/gqa/val/data.jsonl")
TEST_DATA_PATH  = os.path.expanduser("~/datasets/gqa/testdev/data.jsonl")

OUTPUT_DIR  = "outputs/blip2_gqa_full"

# Whether to also unfreeze the OPT feed-forward layers (fc1, fc2).
# True  → more trainable params, potentially higher accuracy, more VRAM/time.
# False → only attention projections unfrozen in the LM (faster, still strong).
UNFREEZE_LM_FFN = True

# Sequence length — GQA questions can be longer than DAQUAR; 256 is safe.
MAX_SEQ_LEN = 256

# Batch / optimisation
# H200 (141 GB VRAM). Full BF16 BLIP-2 opt-2.7b
# activations + gradients + AdamW states at this batch size.
# Effective batch = PER_DEVICE_BATCH × GRAD_ACCUM = 32 × 2 = 64.
NUM_EPOCHS       = 2
PER_DEVICE_BATCH = 32
GRAD_ACCUM       = 2

# Full fine-tuning uses a much lower LR than LoRA (1e-4 → 1e-5).
LR           = 1e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0        # clips the gradient spikes

# Logging / saving
LOGGING_STEPS = 100
SAVE_STEPS    = 1000       # checkpoint every 1 000 steps

# Set to None to use the full val split at each checkpoint.
EVAL_SIZE_CAP = None

NUM_WORKERS = 4            # pipeline image I/O; matches cpus-per-task / 2
SEED        = 42
# ============================================================


# Reproducibility

def set_full_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


set_full_seed(SEED)


# Pre-flight checks 
# Runs before any model loading. Fails fast and loudly if anything is wrong,
# so you don't waste GPU time discovering a path problem 30 min into training.

def preflight_checks() -> None:
    sep = "=" * 62

    print(f"\n{sep}")
    print("  PRE-FLIGHT CHECKS")
    print(sep)

    # 1. Working directory — confirms scratch was activated
    cwd = os.getcwd()
    print(f"\n[1] Working directory : {cwd}")
    if "lustre" in cwd or "scratch" in cwd:
        print("    ✓  Running on Lustre scratch (fast storage)")
    else:
        print("    ⚠  Not on scratch — running on NFS (slower writes)")

    # 2. Output directory writability
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    probe = os.path.join(OUTPUT_DIR, ".write_test")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        print(f"[2] Output dir writable : {os.path.abspath(OUTPUT_DIR)}  ✓")
    except OSError as e:
        raise RuntimeError(f"OUTPUT_DIR is not writable: {OUTPUT_DIR}") from e

    # 3. JSONL files — existence, record count, and first-row inspection
    print(f"\n[3] JSONL data files:")
    for label, path in [("TRAIN", TRAIN_DATA_PATH),
                         ("EVAL",  EVAL_DATA_PATH),
                         ("TEST",  TEST_DATA_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} data not found: {path}\n"
                f"Run prepare_gqa.py --splits train val testdev first."
            )
        with open(path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        n = len(lines)
        first = json.loads(lines[0])
        print(f"\n    {label}  ({n:,} records)  →  {path}")
        print(f"      image_path : {first['image_path']}")
        print(f"      question   : {first['question']}")
        print(f"      answer     : {first['answer']}")
        print(f"      category   : {first.get('category', '—')}")

    # 4. Image access — open the first training image and print its properties
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        first_train = json.loads(f.readline())
    img_path = first_train["image_path"]
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"First training image not found on disk: {img_path}\n"
            f"Check that prepare_gqa.py completed successfully."
        )
    img = Image.open(img_path).convert("RGB")
    print(f"\n[4] First training image opened successfully:")
    print(f"    path   : {img_path}")
    print(f"    size   : {img.size[0]}×{img.size[1]} px  mode: {img.mode}  ✓")
    del img

    # 5. GPU snapshot before model load
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
    name   = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlShutdown()
    print(f"\n[5] GPU state before model load:")
    print(f"    device : {name}")
    print(f"    VRAM   : {mem.used/1024**3:.1f} GB used / "
          f"{mem.total/1024**3:.1f} GB total  ✓")

    # 6. Key configuration summary
    eff_batch = PER_DEVICE_BATCH * GRAD_ACCUM
    print(f"\n[6] Training configuration summary:")
    print(f"    epochs          : {NUM_EPOCHS}")
    print(f"    per-device batch: {PER_DEVICE_BATCH}  ×  grad_accum {GRAD_ACCUM}"
          f"  →  effective batch {eff_batch}")
    print(f"    learning rate   : {LR}")
    print(f"    max seq len     : {MAX_SEQ_LEN}")
    print(f"    DataLoader workers: {NUM_WORKERS}")
    print(f"    unfreeze LM FFN : {UNFREEZE_LM_FFN}")
    print(f"    seed            : {SEED}")

    print(f"\n{sep}")
    print("  ALL PRE-FLIGHT CHECKS PASSED — starting training")
    print(f"{sep}\n")

# Run all sanity checks before touching the model or GPU
preflight_checks()

# GPU monitoring

class GPUMonitorCallback(TrainerCallback):
    """
    Log GPU memory, utilisation, and temperature to TensorBoard at every
    logging step. Runs only on the main process (rank 0).

    Metrics written (visible in TensorBoard under the 'gpu/' prefix):
      gpu/mem_used_gb      — VRAM currently allocated
      gpu/mem_pct          — VRAM utilisation as a percentage
      gpu/util_pct         — GPU compute utilisation (%)
      gpu/temp_c           — GPU temperature in Celsius
    """

    def __init__(self, tb_log_dir: str, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "gpu_monitor"))
        name = pynvml.nvmlDeviceGetName(self.handle)
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / 1024 ** 3
        print(f"GPUMonitorCallback: {name}  |  {total_mem:.1f} GB VRAM")

    def _read(self) -> dict:
        mem  = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu/mem_used_gb": mem.used  / 1024 ** 3,
            "gpu/mem_pct":     mem.used  / mem.total * 100,
            "gpu/util_pct":    util.gpu,
            "gpu/temp_c":      temp,
        }

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return
        stats = self._read()
        step  = state.global_step
        for k, v in stats.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()
        
        print(
            f"[GPU step {step:>6}] "
            f"mem {stats['gpu/mem_used_gb']:.1f} GB "
            f"({stats['gpu/mem_pct']:.0f}%)  "
            f"util {stats['gpu/util_pct']:.0f}%  "
            f"temp {stats['gpu/temp_c']:.0f}°C"
        )

    def on_train_end(self, args, state, control, **kwargs):
        # Print final peak snapshot
        if state.is_world_process_zero:
            stats = self._read()
            print(
                f"\n[GPU final] "
                f"mem {stats['gpu/mem_used_gb']:.2f} GB  "
                f"util {stats['gpu/util_pct']}%  "
                f"temp {stats['gpu/temp_c']}°C"
            )
        self.writer.close()
        pynvml.nvmlShutdown()


# Sample prediction callback
# Every SAMPLE_EVERY optimizer steps, runs inference on a small fixed set
# of training samples and prints image path, question, ground-truth answer,
# and the model's current prediction. Gives a human-readable quality check
# directly in the SLURM .out file without touching the val split.
#
# Design notes:
#   - Uses a FIXED set of samples (same indices every time) so you can watch
#     the same questions improve across checkpoints in the log.
#   - model.eval() / torch.no_grad() during inference, then model.train() after
#     — the Trainer's own training state is never affected.
#   - Wraps the forward pass in try/except so a single bad sample never kills
#     a 25-hour training job.

SAMPLE_EVERY  = 1000   # print samples every N optimizer steps
N_SAMPLES     = 5      # how many samples to show each time

class SamplePredictionCallback(TrainerCallback):
    """
    Print a few sample predictions to stdout at regular step intervals.
    Useful for qualitative monitoring directly in the SLURM .out file.
    """

    def __init__(
        self,
        records:    list,
        processor,
        n_samples:  int = N_SAMPLES,
        every_steps: int = SAMPLE_EVERY,
        seed:       int = SEED,
    ):
        # Pick a fixed set of indices once — same samples shown every time
        rng = random.Random(seed + 99)   # separate seed so it doesn't affect training
        population = list(range(len(records)))
        self.indices   = sorted(rng.sample(population, min(n_samples, len(population))))
        self.samples   = [records[i] for i in self.indices]
        self.processor = processor
        self.every     = every_steps

    @torch.no_grad()
    def _predict_one(self, model, rec: dict) -> str:
        image  = Image.open(rec["image_path"]).convert("RGB")
        prompt = f"Question: {rec['question']} Answer:"
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(model.device)
        ids = model.generate(
            **inputs,
            max_new_tokens=10,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        # Strip the echoed prompt if the model includes it in output
        if text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip()
        return text

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        # Only on the main process, only at the requested interval
        if not state.is_world_process_zero:
            return
        if state.global_step == 0 or state.global_step % self.every != 0:
            return

        sep = "─" * 62
        print(f"\n{'═'*62}")
        print(f"  SAMPLE PREDICTIONS  —  step {state.global_step}")
        print(f"{'═'*62}")

        model.eval()
        for i, rec in enumerate(self.samples):
            try:
                pred = self._predict_one(model, rec)
            except Exception as e:
                pred = f"<ERROR: {e}>"

            print(f"\n  [{i+1}/{len(self.samples)}]")
            print(f"  image    : {os.path.basename(rec['image_path'])}")
            print(f"  question : {rec['question']}")
            print(f"  gt       : {rec['answer']}")
            print(f"  pred     : {pred}")
            match = "✓" if pred.strip().lower() == rec["answer"].strip().lower() else "✗"
            print(f"  match    : {match}")

        model.train()
        print(f"\n{sep}\n")


# Model loading

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # No quantization_config — pure BF16
)

# Selective parameter unfreezing
# Start with everything frozen, then unfreeze target groups.
for p in model.parameters():
    p.requires_grad = False

unfrozen_groups = {
    "Q-Former + query tokens": 0,
    "Language projection":     0,
    "OPT attention":           0,
    "OPT feed-forward":        0,
}

for name, param in model.named_parameters():
    # Group 1: Full Q-Former and learnable query embeddings
    if "qformer" in name or "query_tokens" in name:
        param.requires_grad = True
        unfrozen_groups["Q-Former + query tokens"] += param.numel()

    # Group 2: Q-Former → LM projection bridge
    elif "language_projection" in name:
        param.requires_grad = True
        unfrozen_groups["Language projection"] += param.numel()

    # Groups 3 & 4: OPT decoder layers
    elif "language_model" in name:
        is_attn = any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj"))
        is_ffn  = any(k in name for k in ("fc1", "fc2"))

        if is_attn:
            param.requires_grad = True
            unfrozen_groups["OPT attention"] += param.numel()
        elif is_ffn and UNFREEZE_LM_FFN:
            param.requires_grad = True
            unfrozen_groups["OPT feed-forward"] += param.numel()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters by group:")
for group, n in unfrozen_groups.items():
    print(f"  {group:<30} {n/1e6:>8.2f} M")
print(f"  {'TOTAL trainable':<30} {trainable/1e6:>8.2f} M")
print(f"  {'TOTAL parameters':<30} {total/1e6:>8.2f} M")
print(f"  {'Trainable ratio':<30} {100*trainable/total:>8.3f} %\n")


# Dataset

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class VQADataset(Dataset):
    """
    Dataset for BLIP-2 VQA fine-tuning.

    Prompt format (identical in training, evaluation, and inference):
        "Question: <question> Answer: <answer><eos>"

    The LM loss is computed only on answer tokens; prompt tokens are masked
    with -100 so the model learns to answer, not to memorise the template.
    """

    def __init__(self, records: list, processor, max_len: int = 256):
        self.items     = records
        self.processor = processor
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.items[idx]

        if not os.path.exists(rec["image_path"]):
            raise FileNotFoundError(f"Image not found: {rec['image_path']} (idx {idx})")

        image     = Image.open(rec["image_path"]).convert("RGB")
        prompt    = f"Question: {rec['question']} Answer:"
        target    = rec["answer"]
        full_text = prompt + " " + target + self.processor.tokenizer.eos_token

        enc_full = self.processor.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_prompt = self.processor.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc_full.input_ids[0]
        attn_mask = enc_full.attention_mask[0]
        labels    = input_ids.clone()
        labels[:enc_prompt.input_ids.shape[1]] = -100   # mask prompt
        labels[attn_mask == 0]                 = -100   # mask padding

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
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


# Data loading

all_train = load_jsonl(TRAIN_DATA_PATH)
all_eval  = load_jsonl(EVAL_DATA_PATH)
print(f"Train records  : {len(all_train):,}  ({TRAIN_DATA_PATH})")
print(f"Eval records   : {len(all_eval):,}  ({EVAL_DATA_PATH})")
print(f"Test data path : {TEST_DATA_PATH}  (used only by evaluate_hpc.py)")

rng = random.Random(SEED)
rng.shuffle(all_train)

if EVAL_SIZE_CAP is not None and len(all_eval) > EVAL_SIZE_CAP:
    rng_eval = random.Random(SEED)
    shuffled = all_eval[:]
    rng_eval.shuffle(shuffled)
    eval_records = shuffled[:EVAL_SIZE_CAP]
    print(f"Eval capped to : {len(eval_records):,}")
else:
    eval_records = all_eval

train_ds = VQADataset(all_train, processor, max_len=MAX_SEQ_LEN)
eval_ds  = VQADataset(eval_records, processor, max_len=MAX_SEQ_LEN)

# Verify the full tensor pipeline (tokenisation + image processing) on one sample
_s = train_ds[0]
print(f"Tensor pipeline check:")
print(f"  pixel_values  : {_s['pixel_values'].shape}")
print(f"  input_ids     : {_s['input_ids'].shape}")
print(f"  answer tokens : {(_s['labels'] != -100).sum().item()} (non-masked label tokens)")
del _s


# Training arguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    data_seed=SEED,

    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    max_grad_norm=MAX_GRAD_NORM,

    bf16=True,

    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=True,

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

# Trainer

gpu_callback = GPUMonitorCallback(
    tb_log_dir=OUTPUT_DIR,
    device_index=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
)

sample_callback = SamplePredictionCallback(
    records=all_train,
    processor=processor,
    n_samples=N_SAMPLES,
    every_steps=SAMPLE_EVERY,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=Collator(),
    callbacks=[gpu_callback, sample_callback],
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save full model checkpoint
# trainer.save_model handles distributed training correctly and saves the
# complete model weights (not just an adapter), which evaluate_hpc.py
# loads directly without any PEFT wrapping.
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Full model checkpoint saved to: {OUTPUT_DIR}")

# Record test data path for evaluate_hpc.py
test_ref = os.path.join(OUTPUT_DIR, "test_data_path.txt")
with open(test_ref, "w") as f:
    f.write(TEST_DATA_PATH + "\n")
print(f"Test data path recorded at    : {test_ref}")