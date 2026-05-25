"""
inference_monitor.py — Run inference on a few GQA testdev samples with both
                       the baseline and fine-tuned BLIP-2 models while
                       continuously recording GPU utilisation, VRAM usage,
                       and temperature to TensorBoard.

No metrics are computed. The purpose is purely:
  1. Qualitative: print question / ground truth / both predictions side by side.
  2. Quantitative: produce a TensorBoard log of GPU behaviour during inference
                   that can be cited alongside the training GPU logs.

TensorBoard output
------------------
  outputs/inference_monitor/gpu_monitor/
    inference/mem_used_gb
    inference/mem_pct
    inference/util_pct
    inference/temp_c

  The x-axis is wall-clock seconds from the start of inference (not steps),
  which makes it natural to read as a time-series.

Usage
-----
    via sbatch
"""

import json
import os
import random
import threading
import time

import pynvml
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Configuration
MODEL_ID       = "Salesforce/blip2-opt-2.7b"
MODEL_DIR      = "outputs/blip2_gqa_full"       # fine-tuned full checkpoint
TEST_DATA_PATH = os.path.expanduser("~/datasets/gqa/testdev/data.jsonl")

N_SAMPLES      = 20      # number of testdev samples to run inference on
MAX_NEW_TOKENS = 10
SEED           = 42

# How often the background thread reads GPU stats (seconds).
# 0.5 s gives a smooth curve without hammering NVML.
POLL_INTERVAL  = 0.5

TB_LOG_DIR     = "outputs/inference_monitor"


# GPU background monitor

class GPUMonitor:
    """
    Spawns a daemon thread that reads GPU stats every POLL_INTERVAL seconds
    and writes them to TensorBoard. The x-axis step is wall-clock time in
    seconds since start() was called, making the plot easy to read as a
    real-time trace.

    """

    def __init__(self, writer: SummaryWriter, prefix: str, device_index: int = 0):
        self.writer  = writer
        self.prefix  = prefix      # e.g. "baseline" or "finetuned"
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _loop(self):
        t0 = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - t0
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            # Use elapsed seconds as the step so the x-axis is wall-clock time
            step = int(elapsed * 10)   # 0.5 s poll → steps of 5 (tenths of a second)
            self.writer.add_scalar(
                f"inference_{self.prefix}/mem_used_gb",
                mem.used / 1024 ** 3, step
            )
            self.writer.add_scalar(
                f"inference_{self.prefix}/mem_pct",
                mem.used / mem.total * 100, step
            )
            self.writer.add_scalar(
                f"inference_{self.prefix}/util_pct",
                util.gpu, step
            )
            self.writer.add_scalar(
                f"inference_{self.prefix}/temp_c",
                temp, step
            )
            self.writer.flush()
            self._stop.wait(POLL_INTERVAL)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


# Helpers

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@torch.no_grad()
def predict(image: Image.Image, question: str, model, processor) -> str:
    prompt = f"Question: {question} Answer:"
    inputs = processor(
        images=image.convert("RGB"),
        text=prompt,
        return_tensors="pt",
        padding=False,      
        ).to("cuda:0")          

    input_len = inputs["input_ids"].shape[1]

    # OPT generates \n after a short answer before repeating context.
    # Adding it alongside EOS stops generation at the right point for both
    # the base model and the
    # fine-tuned model
    nl_ids  = processor.tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = [processor.tokenizer.eos_token_id] + nl_ids

    ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=1,
        do_sample=False,
        eos_token_id=stop_ids,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Decode only the newly generated tokens — no string-based prompt stripping.
    new_tokens = ids[:, input_len:]
    text = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    # Collapse any remaining whitespace into a single clean line
    return " ".join(text.split())


def run_inference(model, processor, samples: list,
                  label: str, writer: SummaryWriter,
                  device_index: int) -> list:
    """
    Run inference on all samples with GPU monitoring active.
    Returns list of dicts: {"image", "image_path", "question", "gt", "pred"}.
    """
    monitor = GPUMonitor(writer, prefix=label.lower().replace(" ", "_"),
                         device_index=device_index)
    results = []

    print(f"\n{'─'*62}")
    print(f"  Running inference : {label}  ({len(samples)} samples)")
    print(f"{'─'*62}")

    monitor.start()

    for i, rec in enumerate(samples):
        image = Image.open(rec["image_path"]).convert("RGB")
        pred  = predict(image, rec["question"], model, processor)

        results.append({
            "image":      os.path.basename(rec["image_path"]),
            "image_path": rec["image_path"],
            "question":   rec["question"],
            "gt":         rec["answer"],
            "pred":       pred,
        })

        print(f"\n  [{i+1:>2}/{len(samples)}]")
        print(f"  image    : {rec['image_path']}")
        print(f"  question : {rec['question']}")
        print(f"  gt       : {rec['answer']}")
        print(f"  pred     : {pred}")

    monitor.stop()
    print(f"\n  {label} inference complete.")
    return results


# Main

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Pick a fixed set of samples
    random.seed(SEED)
    all_records = load_jsonl(TEST_DATA_PATH)
    samples     = random.sample(all_records, min(N_SAMPLES, len(all_records)))
    print(f"Loaded {len(all_records):,} testdev records — using {len(samples)} samples")
    print(f"TensorBoard logs → {os.path.abspath(TB_LOG_DIR)}")

    device_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    # One TensorBoard writer — both models log into the same run directory
    # so their GPU curves are overlaid on the same chart for easy comparison.
    writer = SummaryWriter(log_dir=os.path.join(TB_LOG_DIR, "gpu_monitor"))

    # Baseline
    print("\nLoading BASELINE model...")
    base_proc  = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
    )
    base_model.eval()

    base_results = run_inference(
        base_model, base_proc, samples, "Baseline", writer, device_index
    )

    del base_model
    torch.cuda.empty_cache()

    # Fine-tuned
    print("\nLoading FINE-TUNED model...")
    ft_proc  = AutoProcessor.from_pretrained(MODEL_DIR)
    ft_model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16,
    )
    ft_model.eval()

    ft_results = run_inference(
        ft_model, ft_proc, samples, "Fine-tuned", writer, device_index
    )

    del ft_model
    torch.cuda.empty_cache()

    writer.close()
    pynvml.nvmlShutdown()

    # Side-by-side summary

    # Debug
    print(f"\nft_results length : {len(ft_results)}")
    if ft_results:
        print(f"ft_results[0]     : {ft_results[0]}")

    print(f"\n{'═'*62}")
    print("  SIDE-BY-SIDE SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Question':<32} {'GT':<12} {'Baseline':<16} {'Fine-tuned'}")
    print(f"  {'─'*32} {'─'*12} {'─'*16} {'─'*16}")
    for i, base_rec in enumerate(base_results):
        ft_pred = ft_results[i]["pred"] if i < len(ft_results) else "<missing>"
        q_short = (base_rec["question"][:30] + "..") if len(base_rec["question"]) > 32 \
                  else base_rec["question"]
        # Truncate predictions so the table stays readable
        bp_short = base_rec["pred"][:15] if len(base_rec["pred"]) > 15 else base_rec["pred"]
        fp_short = ft_pred[:15]          if len(ft_pred) > 15          else ft_pred
        print(f"  {q_short:<32} {base_rec['gt']:<12} {bp_short:<16} {fp_short}")

    print(f"\nTensorBoard : tensorboard --logdir {os.path.abspath(TB_LOG_DIR)}")


if __name__ == "__main__":
    main()