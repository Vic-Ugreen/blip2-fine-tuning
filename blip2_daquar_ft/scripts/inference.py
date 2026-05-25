# pip install -U transformers accelerate bitsandbytes peft pillow gradio

"""
inference.py — Side-by-side comparison of baseline vs fine-tuned BLIP-2.

Two modes
---------
  Web UI (default)
      Launches a Gradio interface. Upload any image, enter a question,
      and see both models' answers and captions side by side.

  CLI batch mode  (--batch)
      Runs over a random sample of the eval split from data.jsonl
      and prints predictions to the terminal.

Usage
-----
    # Web UI
    python inference.py

    # CLI batch (10 random examples from the eval split)
    python inference.py --batch --n 10
"""

import argparse
import json
import os
import random
import time
import re
from typing import Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel

# Configuration — must be the same through all the files.
MODEL_ID       = "Salesforce/blip2-opt-2.7b"
LORA_DIR       = "outputs/lora_adapter"
TEST_SPLIT_PATH = "dataset/test_split.jsonl"
TRAIN_SIZE     = 10_000
MAX_NEW_TOKENS = 10
USE_4BIT       = True
SEED           = 42

# Seed for reproducible batch-mode random sampling.
random.seed(SEED)
set_seed(SEED)


# ── Quantisation ─────────────────────────────────────────────

def get_quant_config():
    if USE_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return None


# ── Model loading ─────────────────────────────────────────────

def load_base_model():
    print("Loading baseline model (this may take a moment)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=get_quant_config(),
    )
    model.eval()
    print("Baseline ready.")
    return model, processor


def load_finetuned_model():
    print("Loading fine-tuned model...")
    processor = AutoProcessor.from_pretrained(LORA_DIR)
    base = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=get_quant_config(),
    )
    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    print("Fine-tuned model ready.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────

@torch.no_grad()
def predict(
    image: Image.Image,
    question: str,
    model: "Blip2ForConditionalGeneration",
    processor: "AutoProcessor",
) -> Tuple[str, float]:
    """
    Run VQA inference for a single image-question pair.
    Returns (answer_string, latency_in_seconds).
    Prompt format is identical to training to avoid train/eval mismatch.
    """
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(model.device)

    t0  = time.perf_counter()
    ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    latency = time.perf_counter() - t0

    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()

    # OPT appends continuation text directly after the answer without
    # any separator. The answer is always a short lowercase word or number;
    # the continuation always begins with a capital letter. Split there.
    # Example: "monitorI'm not sure" -> "monitor"
    # Example: "2I'm not sure"       -> "2"
    match = re.search(r"([a-z0-9_,\s]+)([A-Z].*)", text)
    if match:
        text = match.group(1).strip()

    return text, round(latency, 3)


@torch.no_grad()
def caption(
    image: Image.Image,
    model: "Blip2ForConditionalGeneration",
    processor: "AutoProcessor",
) -> str:
    """
    Generate a free-form image caption without a text prompt.
    This reflects the model's general visual description capability
    independently of any question, and is used for illustration in the UI.
    """
    inputs = processor(
        images=image.convert("RGB"), return_tensors="pt"
    ).to(model.device)
    ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=1, do_sample=False, eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ── Gradio UI ─────────────────────────────────────────────────

def launch_ui(base_model, base_proc, ft_model, ft_proc) -> None:
    """
    Launch a Gradio web interface for interactive side-by-side comparison.
    Compatible with Gradio >= 4.0.
    """
    import gradio as gr
    import numpy as np

    def infer(img, question):
        if img is None:
            return "—", "—", "—", "—", "—"

        pil = (
            Image.fromarray(img.astype("uint8")).convert("RGB")
            if isinstance(img, np.ndarray)
            else img.convert("RGB")
        )
        q = (question or "").strip() or "What is in the image?"

        base_ans, base_lat = predict(pil, q, base_model, base_proc)
        ft_ans,   ft_lat   = predict(pil, q, ft_model,   ft_proc)
        base_cap           = caption(pil, base_model, base_proc)
        ft_cap             = caption(pil, ft_model,   ft_proc)

        match        = "✅ Match" if base_ans.lower() == ft_ans.lower() else "❌ Differ"
        latency_info = f"Baseline {base_lat}s  |  Fine-tuned {ft_lat}s"

        return base_ans, ft_ans, base_cap, ft_cap, f"{match}  |  {latency_info}"

    with gr.Blocks(title="BLIP-2 Baseline vs Fine-tuned") as demo:
        gr.Markdown(
            """
            # BLIP-2 — Baseline vs Fine-tuned Comparison
            Upload an image, enter a question, and compare both models' answers
            and free-form captions side by side.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input    = gr.Image(label="Image", type="numpy")
                question_input = gr.Textbox(
                    label="Question", value="What is in the image?"
                )
                with gr.Row():
                    run_btn   = gr.Button("Compare", variant="primary")
                    clear_btn = gr.Button("Clear")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Baseline (stock BLIP-2)")
                base_ans_out = gr.Textbox(label="Answer")
                base_cap_out = gr.Textbox(label="Caption")
            with gr.Column():
                gr.Markdown("### Fine-tuned (LoRA)")
                ft_ans_out = gr.Textbox(label="Answer")
                ft_cap_out = gr.Textbox(label="Caption")

        status_out = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=infer,
            inputs=[image_input, question_input],
            outputs=[base_ans_out, ft_ans_out, base_cap_out, ft_cap_out, status_out],
        )
        clear_btn.click(
            fn=lambda: (None, "What is in the image?", "", "", "", "", ""),
            outputs=[
                image_input, question_input,
                base_ans_out, ft_ans_out,
                base_cap_out, ft_cap_out,
                status_out,
            ],
        )

    # Gradio >= 4.0: queue() takes no positional arguments; share=True creates a public URL (did not create one for me, but local is enough)
    demo.queue().launch(share=True)


# ── CLI batch mode ─────────────────────────────────────────────

def run_batch(base_model, base_proc, ft_model, ft_proc, n: int) -> None:
    """
    Evaluate both models on n randomly sampled examples from the eval split
    and print a per-sample comparison to the terminal.
    The eval split is derived with the same shuffle seed as finetune.py.
    """
    
    with open(TEST_SPLIT_PATH, "r") as f:
        test_records = [json.loads(l) for l in f if l.strip()]
    
    samples = random.sample(test_records, k=min(n, len(test_records)))

    base_correct = ft_correct = 0

    for rec in samples:
        img = Image.open(rec["image_path"]).convert("RGB")
        q   = rec["question"]
        gt  = rec["answer"]

        base_ans, base_lat = predict(img, q, base_model, base_proc)
        ft_ans,   ft_lat   = predict(img, q, ft_model,   ft_proc)

        b_ok = base_ans.lower().strip() == gt.lower().strip()
        f_ok = ft_ans.lower().strip()   == gt.lower().strip()
        base_correct += b_ok
        ft_correct   += f_ok

        print("=" * 62)
        print(f"Image    : {os.path.basename(rec['image_path'])}")
        print(f"Question : {q}")
        print(f"GT Answer: {gt}")
        print(f"Baseline : {base_ans}  ({base_lat}s)  {'✅' if b_ok else '❌'}")
        print(f"FT Model : {ft_ans}  ({ft_lat}s)  {'✅' if f_ok else '❌'}")

    print("\n" + "=" * 62)
    print(
        f"Exact Match — "
        f"Baseline: {base_correct}/{n} ({100*base_correct/n:.1f}%)  |  "
        f"Fine-tuned: {ft_correct}/{n} ({100*ft_correct/n:.1f}%)"
    )


# ── Entry point ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline and fine-tuned BLIP-2 models."
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run CLI batch comparison instead of launching the Gradio web UI.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of random examples to evaluate in batch mode (default: 10).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base_model, base_proc = load_base_model()
    ft_model,   ft_proc   = load_finetuned_model()

    if args.batch:
        run_batch(base_model, base_proc, ft_model, ft_proc, n=args.n)
    else:
        launch_ui(base_model, base_proc, ft_model, ft_proc)


if __name__ == "__main__":
    main()