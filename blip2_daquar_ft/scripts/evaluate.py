# pip install -U transformers accelerate bitsandbytes peft pillow tqdm
# pip install nltk rouge-score

"""
evaluate.py — Benchmark baseline vs fine-tuned BLIP-2 on the local eval split.

Metrics
-------
  Exact-match accuracy    Normalised string equality (lowercase, strip punct/articles).
                          Primary metric for VQA-RAD / PathVQA-style benchmarks.
  Token F1                Unigram F1 between predicted and ground-truth answer tokens.
                          Gives partial credit for partially correct answers.
  BLEU-1 / BLEU-4         N-gram precision with brevity penalty (Papineni et al., 2002).
                          Included for completeness; less reliable for single-word answers.
  ROUGE-1 / ROUGE-L       N-gram recall / longest-common-subsequence F1 (Lin, 2004).
                          More appropriate than BLEU for open-ended answer evaluation.
  Per-category accuracy   Exact match broken down by question category (if available).
  Throughput / latency    Samples per second and average inference time per sample.

Reproducibility
---------------
  The eval split is derived by shuffling the full JSONL with SEED before
  taking records[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE], identical to finetune.py.

Usage
-----
    python evaluate.py

Outputs
-------
    outputs/eval_results.json      Aggregate metrics for both models.
    outputs/eval_per_sample.json   Full per-sample predictions.
    outputs/eval_summary.txt       Human-readable comparison table.
"""

import json
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel

# nltk is used for BLEU computation (corpus_bleu).
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# rouge_score is used for ROUGE-1 and ROUGE-L computation.
from rouge_score import rouge_scorer

# Configuration - must be same through all the files
MODEL_ID        = "Salesforce/blip2-opt-2.7b"
LORA_DIR        = "outputs/lora_adapter"
TEST_SPLIT_PATH = "dataset/test_split.jsonl"
TRAIN_SIZE      = 10_000
EVAL_SIZE       = 1_234
TEST_SIZE       = 1_234
MAX_NEW_TOKENS  = 10
USE_4BIT        = True
OUT_DIR         = "outputs"
SEED            = 42


os.makedirs(OUT_DIR, exist_ok=True)

# Ensure NLTK tokeniser data is present (needed for BLEU tokenisation).
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Seed everything for reproducibility.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)


# ── Answer normalisation ──────────────────────────────────────

_ARTICLES  = {"a", "an", "the"}
_PUNCT     = set(';/[]"{}()=+\\_-><@`,?!\'')
_PERIOD_RE = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_RE  = re.compile(r"(\d),(\d)")


def normalize(ans: str) -> str:
    """
    Normalise an answer string for evaluation:
      - Lowercase
      - Remove periods not between digits
      - Normalise commas in numbers
      - Remove punctuation
      - Remove articles (a, an, the)
    """
    ans = ans.lower().strip()
    ans = _PERIOD_RE.sub("", ans)
    ans = _COMMA_RE.sub(r"\1\2", ans)
    ans = "".join(" " if c in _PUNCT else c for c in ans)
    tokens = [t for t in ans.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


# ── Individual metrics ────────────────────────────────────────

def exact_match(pred: str, gt: str) -> float:
    """1.0 if normalised prediction equals normalised ground truth, else 0.0."""
    return float(normalize(pred) == normalize(gt))


def token_f1(pred: str, gt: str) -> float:
    """
    Unigram F1 between predicted and ground-truth answer tokens.
    Gives partial credit when the prediction is partially correct.
    """
    pred_tokens = normalize(pred).split()
    gt_tokens   = normalize(gt).split()
    if not pred_tokens or not gt_tokens:
        return float(normalize(pred) == normalize(gt))
    common = sum(
        min(pred_tokens.count(t), gt_tokens.count(t))
        for t in set(pred_tokens)
    )
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec  = common / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)


# ── Corpus-level metric computation ──────────────────────────

def compute_bleu(predictions: list, references: list) -> dict:
    """
    Compute corpus-level BLEU-1 and BLEU-4 using NLTK.

    BLEU measures n-gram precision with a brevity penalty.
    For single-word VQA answers, BLEU-4 will frequently be 0 because
    there are no 4-grams to match; BLEU-1 is the more informative figure.

    Smoothing (method1) is applied to avoid zero scores for n>1 on short
    answers, following standard practice in VQA evaluation literature.

    Parameters
    ----------
    predictions : list of str   — model-generated answer strings
    references  : list of str   — ground-truth answer strings
    """
    smooth = SmoothingFunction().method1

    # NLTK corpus_bleu expects:
    #   references: list of list of list of tokens  [[ref_tokens], ...]
    #   hypotheses: list of list of tokens           [hyp_tokens, ...]
    refs  = [[normalize(r).split()] for r in references]
    hyps  = [normalize(p).split()   for p in predictions]

    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
    }


def compute_rouge(predictions: list, references: list) -> dict:
    """
    Compute corpus-level ROUGE-1 F1 and ROUGE-L F1 using the rouge_score library.

    ROUGE measures recall-oriented n-gram overlap.
    ROUGE-1 counts unigram overlap; ROUGE-L uses the longest common subsequence.
    Both are reported as F1 scores (harmonic mean of precision and recall).

    For single-word answers, ROUGE-1 and exact match are equivalent.
    ROUGE-L becomes more informative for multi-word open-ended answers.

    Parameters
    ----------
    predictions : list of str   — model-generated answer strings
    references  : list of str   — ground-truth answer strings
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_scores, rL_scores = [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(normalize(ref), normalize(pred))
        r1_scores.append(scores["rouge1"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(r1_scores) / len(r1_scores), 4),
        "rougeL": round(sum(rL_scores) / len(rL_scores), 4),
    }


# ── Data loading ──────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Model loading ─────────────────────────────────────────────

def get_quant_config() -> "Optional[BitsAndBytesConfig]":
    if USE_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_base_model():
    print("Loading BASELINE model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=get_quant_config(),
    )
    model.eval()
    return model, processor


def load_finetuned_model():
    print("Loading FINE-TUNED model...")
    # The processor is loaded from LORA_DIR because it may have been
    # saved with updated padding/truncation settings.
    processor = AutoProcessor.from_pretrained(LORA_DIR)
    base = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=get_quant_config(),
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return model, processor


# ── Inference ────────────────────────────────────────────────

@torch.no_grad()
def predict(
    image: Image.Image,
    question: str,
    model: "Blip2ForConditionalGeneration",
    processor: "AutoProcessor",
) -> str:
    """
    Generate an answer for a single image-question pair.
    The prompt format MUST match the format used during training exactly.
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(model.device)
    ids = model.generate(
        **inputs,
        max_new_tokens=10,                              
        min_new_tokens=1,                               # force at least one token
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id, # explicit stopping criterion
        pad_token_id=processor.tokenizer.pad_token_id, # suppress pad warning
    )
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()

    match = re.search(r"([a-z0-9_,\s]+)([A-Z].*)", text)
    if match:
        text = match.group(1).strip()

    return text


# ── Evaluation loop ───────────────────────────────────────────

def evaluate(
    model,
    processor,
    records: list,
    label: str,
) -> dict:
    """
    Run inference on all records and compute all metrics.

    Returns a dict containing aggregate metrics and per-sample predictions.
    """
    em_scores, f1_scores, latencies = [], [], []
    all_predictions, all_references = [], []
    per_sample = []
    cat_em = defaultdict(list)
    # Print first 5 predictions to diagnose output format before full run (unused check)
    # print(f"\n--- {label}: sample predictions ---")
    # for rec in records[:5]:
    #     image = Image.open(rec["image_path"]).convert("RGB")
    #     pred  = predict(image, rec["question"], model, processor)
    #     print(f"  Q: {rec['question']}")
    #     print(f"  GT: {rec['answer']}  |  Pred: {repr(pred)}")
    # print("---\n")

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
        all_predictions.append(pred)
        all_references.append(gt)

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

    n       = len(records)
    avg_lat = sum(latencies) / n

    # Compute corpus-level BLEU and ROUGE over all predictions at once.
    bleu_scores  = compute_bleu(all_predictions, all_references)
    rouge_scores = compute_rouge(all_predictions, all_references)

    return {
        "label":          label,
        "n_samples":      n,
        "exact_match":    round(sum(em_scores) / n, 4),
        "token_f1":       round(sum(f1_scores) / n, 4),
        "bleu1":          bleu_scores["bleu1"],
        "bleu4":          bleu_scores["bleu4"],
        "rouge1":         rouge_scores["rouge1"],
        "rougeL":         rouge_scores["rougeL"],
        "avg_latency_s":  round(avg_lat, 4),
        "throughput_sps": round(1.0 / avg_lat, 2),
        "per_category":   {
            cat: round(sum(v) / len(v), 4) for cat, v in cat_em.items()
        },
        "per_sample":     per_sample,
    }


# ── Result display ────────────────────────────────────────────

def print_summary(base: dict, ft: dict) -> list:
    """
    Print and return a formatted comparison table of all aggregate metrics.
    """
    header = (
        f"\n{'Metric':<26} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}"
    )
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
        all_cats = sorted(set(base["per_category"]) | set(ft["per_category"]))
        for cat in all_cats:
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


# ── Main ─────────────────────────────────────────────────────

def main():
    # Load the dedicated test split saved by finetune.py.
    # This split was never seen by the model during training in any form
    # not for gradient updates and not for checkpoint selection.
    if not os.path.exists(TEST_SPLIT_PATH):
        raise FileNotFoundError(
            f"Test split not found at {TEST_SPLIT_PATH}. "
            f"Run finetune.py first to generate it."
        )
    eval_records = load_jsonl(TEST_SPLIT_PATH)
    print(f"Evaluating on {len(eval_records)} held-out test samples")

    results = {}

    # ── Baseline ──────────────────────────────────────────────
    base_model, base_proc = load_base_model()
    results["baseline"] = evaluate(base_model, base_proc, eval_records, "Baseline")
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned ────────────────────────────────────────────
    ft_model, ft_proc = load_finetuned_model()
    results["finetuned"] = evaluate(ft_model, ft_proc, eval_records, "Fine-tuned")
    del ft_model
    torch.cuda.empty_cache()

    # ── Print summary ─────────────────────────────────────────
    lines = print_summary(results["baseline"], results["finetuned"])

    # ── Save per-sample predictions ───────────────────────────
    per_sample_path = os.path.join(OUT_DIR, "eval_per_sample.json")
    with open(per_sample_path, "w") as f:
        json.dump(
            {
                "baseline":  results["baseline"]["per_sample"],
                "finetuned": results["finetuned"]["per_sample"],
            },
            f,
            indent=2,
        )

    # ── Save aggregate results ────────────────────────────────
    summary = {
        k: {m: v for m, v in r.items() if m != "per_sample"}
        for k, r in results.items()
    }
    out_path = os.path.join(OUT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Save human-readable summary ───────────────────────────
    txt_path = os.path.join(OUT_DIR, "eval_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nAggregate results → {out_path}")
    print(f"Per-sample results → {per_sample_path}")
    print(f"Text summary       → {txt_path}")


if __name__ == "__main__":
    main()