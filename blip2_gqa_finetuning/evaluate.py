"""
evaluate_hpc.py — Benchmark baseline vs fine-tuned BLIP-2 on GQA testdev.
                  HPC version: loads full BF16 checkpoint (no LoRA / quantization).


Outputs
-------
    outputs/eval_results.json      Aggregate metrics for both models.
    outputs/eval_per_sample.json   Full per-sample predictions.
    outputs/eval_summary.txt       Human-readable comparison table.
    outputs/eval_gpu_stats.json    GPU memory / utilisation summary per model.
"""

import json
import os
import re
import random
import time
from collections import defaultdict

import numpy as np
import pynvml
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    set_seed,
)

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "Salesforce/blip2-opt-2.7b"

# Directory where finetune_hpc.py saved the full model checkpoint.
MODEL_DIR = "outputs/blip2_gqa_full"

# Held-out test split produced by prepare_gqa.py.
# If set to None, falls back to reading outputs/blip2_gqa_full/test_data_path.txt
TEST_DATA_PATH = None

MAX_NEW_TOKENS = 10
OUT_DIR        = "outputs"
SEED           = 42
# ============================================================

os.makedirs(OUT_DIR, exist_ok=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)


# GPU monitoring

class GPUSnapshot:
    """Lightweight wrapper for one-shot GPU stat reads during inference."""

    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info  = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        name  = pynvml.nvmlDeviceGetName(self.handle)
        print(f"GPU: {name}  |  total VRAM: {info.total/1024**3:.1f} GB")

    def read(self) -> dict:
        mem  = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "mem_used_gb":  round(mem.used  / 1024 ** 3, 2),
            "mem_total_gb": round(mem.total / 1024 ** 3, 2),
            "mem_pct":      round(mem.used  / mem.total * 100, 1),
            "util_pct":     util.gpu,
            "temp_c":       temp,
        }

    def shutdown(self):
        pynvml.nvmlShutdown()


# Answer normalisation

_ARTICLES  = {"a", "an", "the"}
_PUNCT     = set(';/[]"{}()=+\\_-><@`,?!\'')
_PERIOD_RE = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_RE  = re.compile(r"(\d),(\d)")


def normalize(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _PERIOD_RE.sub("", ans)
    ans = _COMMA_RE.sub(r"\1\2", ans)
    ans = "".join(" " if c in _PUNCT else c for c in ans)
    tokens = [t for t in ans.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def exact_match(pred: str, gt: str) -> float:
    return float(normalize(pred) == normalize(gt))


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = normalize(pred).split()
    gt_tokens   = normalize(gt).split()
    if not pred_tokens or not gt_tokens:
        return float(normalize(pred) == normalize(gt))
    common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in set(pred_tokens))
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec  = common / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)


def compute_bleu(predictions, references):
    smooth = SmoothingFunction().method1
    refs   = [[normalize(r).split()] for r in references]
    hyps   = [normalize(p).split()   for p in predictions]
    return {
        "bleu1": round(corpus_bleu(refs, hyps, weights=(1,0,0,0), smoothing_function=smooth), 4),
        "bleu4": round(corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25), smoothing_function=smooth), 4),
    }


def compute_rouge(predictions, references):
    scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1, rL = [], []
    for pred, ref in zip(predictions, references):
        s = scorer_obj.score(normalize(ref), normalize(pred))
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    return {
        "rouge1": round(sum(r1) / len(r1), 4),
        "rougeL": round(sum(rL) / len(rL), 4),
    }


# Data loading

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def resolve_test_path() -> str:
    if TEST_DATA_PATH is not None:
        return os.path.expanduser(TEST_DATA_PATH)
    ref = os.path.join(MODEL_DIR, "test_data_path.txt")
    if not os.path.exists(ref):
        raise FileNotFoundError(
            f"TEST_DATA_PATH is None and no fallback at {ref}. "
            f"Run finetune_hpc.py first or set TEST_DATA_PATH explicitly."
        )
    with open(ref) as f:
        return os.path.expanduser(f.read().strip())


# Model loading

def load_base_model():
    print("\nLoading BASELINE model...")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    mdl  = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    mdl.eval()
    return mdl, proc


def load_finetuned_model():
    print("\nLoading FINE-TUNED model...")
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at {MODEL_DIR}. "
            f"Run finetune_hpc.py first."
        )
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    # Load full model from checkpoint — no PeftModel wrapping needed.
    mdl  = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    mdl.eval()
    return mdl, proc


# Inference

@torch.no_grad()
def predict(image: Image.Image, question: str, model, processor) -> str:
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(model.device)
    ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    match = re.search(r"([a-z0-9_,\s]+)([A-Z].*)", text)
    if match:
        text = match.group(1).strip()
    return text


# Evaluation loop

def evaluate(model, processor, records: list, label: str, gpu: GPUSnapshot) -> dict:
    em_scores, f1_scores, latencies = [], [], []
    all_preds, all_refs = [], []
    per_sample = []
    cat_em     = defaultdict(list)

    # GPU snapshot before inference starts
    gpu_before = gpu.read()

    for rec in tqdm(records, desc=label):
        image    = Image.open(rec["image_path"]).convert("RGB")
        question = rec["question"]
        gt       = rec["answer"]

        t0   = time.perf_counter()
        pred = predict(image, question, model, processor)
        latencies.append(time.perf_counter() - t0)

        em = exact_match(pred, gt)
        f1 = token_f1(pred, gt)
        em_scores.append(em)
        f1_scores.append(f1)
        all_preds.append(pred)
        all_refs.append(gt)

        entry = {
            "image_path":  rec["image_path"],
            "question":    question,
            "gt_answer":   gt,
            "prediction":  pred,
            "exact_match": em,
            "token_f1":    f1,
        }
        if "category" in rec:
            entry["category"] = rec["category"]
            cat_em[rec["category"]].append(em)
        per_sample.append(entry)

    gpu_after = gpu.read()

    n       = len(records)
    avg_lat = sum(latencies) / n
    bleu    = compute_bleu(all_preds, all_refs)
    rouge   = compute_rouge(all_preds, all_refs)

    print(
        f"\n[GPU {label}] "
        f"before: {gpu_before['mem_used_gb']:.1f} GB  "
        f"after: {gpu_after['mem_used_gb']:.1f} GB  "
        f"util: {gpu_after['util_pct']}%  "
        f"temp: {gpu_after['temp_c']}°C"
    )

    return {
        "label":          label,
        "n_samples":      n,
        "exact_match":    round(sum(em_scores) / n, 4),
        "token_f1":       round(sum(f1_scores) / n, 4),
        "bleu1":          bleu["bleu1"],
        "bleu4":          bleu["bleu4"],
        "rouge1":         rouge["rouge1"],
        "rougeL":         rouge["rougeL"],
        "avg_latency_s":  round(avg_lat, 4),
        "throughput_sps": round(1.0 / avg_lat, 2),
        "per_category":   {cat: round(sum(v)/len(v), 4) for cat, v in cat_em.items()},
        "per_sample":     per_sample,
        "gpu_before":     gpu_before,
        "gpu_after":      gpu_after,
    }


# Result display

def print_summary(base: dict, ft: dict) -> list:
    header  = f"\n{'Metric':<26} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}"
    divider = "─" * (len(header) - 1)
    print(header)
    print(divider)

    metrics = [
        ("exact_match",    "Exact Match Acc"),
        ("token_f1",       "Token F1"),
        ("bleu1",          "BLEU-1"),
        ("bleu4",          "BLEU-4"),
        ("rouge1",         "ROUGE-1"),
        ("rougeL",         "ROUGE-L"),
        ("throughput_sps", "Throughput (samp/s)"),
        ("avg_latency_s",  "Avg Latency (s)"),
    ]
    lines = [header, divider]

    for key, label in metrics:
        b, f = base[key], ft[key]
        try:
            delta = f"{f - b:+.4f}"
        except TypeError:
            delta = "—"
        line = f"{label:<26} {str(b):>10} {str(f):>12} {delta:>10}"
        print(line)
        lines.append(line)

    if base["per_category"]:
        cat_header = "\nPer-category Exact Match:"
        print(cat_header)
        lines.append(cat_header)
        for cat in sorted(set(base["per_category"]) | set(ft["per_category"])):
            b_c = base["per_category"].get(cat, "—")
            f_c = ft["per_category"].get(cat, "—")
            try:
                d = f"{f_c - b_c:+.4f}"
            except TypeError:
                d = "—"
            cat_line = f"  {cat:<24} {str(b_c):>10} {str(f_c):>12} {d:>10}"
            print(cat_line)
            lines.append(cat_line)

    return lines


# Main

def main():
    test_path = resolve_test_path()
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found at {test_path}. Run prepare_gqa.py first."
        )
    eval_records = load_jsonl(test_path)
    print(f"Evaluating on {len(eval_records):,} testdev samples  ({test_path})")

    device_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    gpu = GPUSnapshot(device_index=device_idx)

    results = {}

    base_model, base_proc = load_base_model()
    results["baseline"] = evaluate(base_model, base_proc, eval_records, "Baseline", gpu)
    del base_model
    torch.cuda.empty_cache()

    ft_model, ft_proc = load_finetuned_model()
    results["finetuned"] = evaluate(ft_model, ft_proc, eval_records, "Fine-tuned", gpu)
    del ft_model
    torch.cuda.empty_cache()

    gpu.shutdown()

    lines = print_summary(results["baseline"], results["finetuned"])

    # Save per-sample predictions
    per_sample_path = os.path.join(OUT_DIR, "eval_per_sample.json")
    with open(per_sample_path, "w") as f:
        json.dump(
            {"baseline": results["baseline"]["per_sample"],
             "finetuned": results["finetuned"]["per_sample"]},
            f, indent=2,
        )

    # Save aggregate metrics (strip per_sample for readability)
    summary = {
        k: {m: v for m, v in r.items() if m != "per_sample"}
        for k, r in results.items()
    }
    out_path = os.path.join(OUT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save GPU stats separately
    gpu_stats = {
        k: {"gpu_before": r["gpu_before"], "gpu_after": r["gpu_after"]}
        for k, r in results.items()
    }
    gpu_path = os.path.join(OUT_DIR, "eval_gpu_stats.json")
    with open(gpu_path, "w") as f:
        json.dump(gpu_stats, f, indent=2)

    # Save text summary
    txt_path = os.path.join(OUT_DIR, "eval_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAggregate results  → {out_path}")
    print(f"Per-sample results → {per_sample_path}")
    print(f"GPU stats          → {gpu_path}")
    print(f"Text summary       → {txt_path}")


if __name__ == "__main__":
    main()