"""
prepare_gqa.py — Download and convert GQA (balanced splits) to the JSONL format
                 expected by finetune_hpc.py and evaluate_hpc.py.

IMPORTANT — HPC path handling
------------------------------
All image_path values written to JSONL are ABSOLUTE paths (resolved at download
time).
But still, control the paths to the files as some parts may differ from user to user

Recommended layout
--------------------------
  ~/datasets/gqa/split/images+data.jsonl

Usage
-----
    via download_gqa.sh, edit the --splits and --out_dir arguments as needed.
  # Testdev only (smoke test):
  --splits testdev --out_dir ~/datasets/gqa

  # All three splits (full HPC run):
  --splits train val testdev --out_dir ~/datasets/gqa

  # Capped smoke test:
  --splits testdev --limit 500 --out_dir ~/datasets/gqa
"""

import argparse
import json
import os
import sys
from pathlib import Path

SPLIT_REGISTRY = {
    "train":   ("train_balanced_images",   "train_balanced_instructions",   "train"),
    "val":     ("val_balanced_images",     "val_balanced_instructions",     "val"),
    "testdev": ("testdev_balanced_images", "testdev_balanced_instructions", "testdev"),
}
DATASET_ID = "lmms-lab/GQA"


def parse_args():
    p = argparse.ArgumentParser(
        description="Download and preprocess GQA splits for BLIP-2 fine-tuning."
    )
    p.add_argument(
        "--splits", nargs="+", default=["testdev"],
        choices=list(SPLIT_REGISTRY.keys()),
        help="Which splits to download. Example: --splits train val testdev"
    )
    p.add_argument(
        "--out_dir", default="~/datasets/gqa",
        help="Root output directory. Tilde is expanded. Default: ~/datasets/gqa"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap on QA pairs per split (for smoke tests)."
    )
    return p.parse_args()


def process_split(split_name, out_dir_resolved, limit=None):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: run:  pip install datasets pillow tqdm")

    from tqdm import tqdm

    img_config, qa_config, hf_split = SPLIT_REGISTRY[split_name]

    # Use resolved absolute path
    split_dir = out_dir_resolved / split_name
    img_dir   = split_dir / "images"
    jsonl_out = split_dir / "data.jsonl"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Split            : {split_name}")
    print(f"  Absolute out dir : {split_dir}")
    print(f"{'='*62}")

    # QA instructions
    print(f"[{split_name}] 1/3 Loading QA instructions ({qa_config})...")
    qa_ds = load_dataset(DATASET_ID, qa_config, split=hf_split, trust_remote_code=True)
    print(f"      {len(qa_ds):,} QA pairs loaded.")

    if limit is not None and limit < len(qa_ds):
        print(f"      Limiting to {limit:,} (--limit).")
        qa_ds = qa_ds.select(range(limit))

    needed_ids = set(qa_ds["imageId"])
    print(f"      Unique images needed: {len(needed_ids):,}")

    # Images
    print(f"[{split_name}] 2/3 Loading and saving images ({img_config})...")
    img_ds = load_dataset(DATASET_ID, img_config, split=hf_split, trust_remote_code=True)
    print(f"      Total images in split: {len(img_ds):,}")

    image_id_to_path = {}

    for row in tqdm(img_ds, desc=f"  Saving {split_name} images", unit="img"):
        img_id = row["id"]
        if img_id not in needed_ids:
            continue
        # resolve() gives the full absolute path — no relative path stored
        png_path = (img_dir / f"{img_id}.png").resolve()
        if not png_path.exists():
            row["image"].convert("RGB").save(png_path, format="PNG")
        image_id_to_path[img_id] = str(png_path)

    missing = needed_ids - set(image_id_to_path.keys())
    if missing:
        print(f"  WARNING: {len(missing)} imageId(s) missing from images config — "
              f"skipping those QA pairs.")

    # JSONL
    print(f"[{split_name}] 3/3 Writing {jsonl_out}...")
    written = skipped = 0

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for row in tqdm(qa_ds, desc=f"  Writing {split_name} JSONL", unit="pair"):
            img_id = row["imageId"]
            answer = row["answer"].strip()
            if img_id not in image_id_to_path or not answer:
                skipped += 1
                continue
            record = {
                # Absolute path — should be safe
                "image_path": image_id_to_path[img_id],
                "question":   row["question"].strip(),
                "answer":     answer,
                "category":   row["types"]["structural"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Written : {written:,}  |  Skipped: {skipped:,}  |  "
          f"Images saved: {len(image_id_to_path):,}")
    print(f"  JSONL   : {jsonl_out}")

    # Sample check
    print(f"\n  First 3 entries:")
    with open(jsonl_out, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rec = json.loads(line)
            print(f"    [{i}] {rec['image_path']}")
            print(f"         Q: {rec['question']}")
            print(f"         A: {rec['answer']}  | cat: {rec.get('category','—')}")

    return written


def main():
    args = parse_args()

    # Expand ~ and resolve to absolute path at download time
    out_dir_resolved = Path(os.path.expanduser(args.out_dir)).resolve()
    print(f"Output root (absolute) : {out_dir_resolved}")
    print(f"Splits                 : {args.splits}")
    if args.limit:
        print(f"QA cap per split       : {args.limit:,}")

    totals = {}
    for split_name in args.splits:
        totals[split_name] = process_split(split_name, out_dir_resolved, args.limit)

    print(f"\n{'='*62}")
    print("  COMPLETE")
    print(f"{'='*62}")
    for split_name, n in totals.items():
        path = out_dir_resolved / split_name / "data.jsonl"
        print(f"  {split_name:<10}  {n:>9,} pairs  →  {path}")
    print()
    print("Set these paths in finetune_hpc.py:")
    for split_name in totals:
        print(f"  {split_name.upper()}_DATA_PATH = "
              f'"{out_dir_resolved / split_name / "data.jsonl"}"')


if __name__ == "__main__":
    main()