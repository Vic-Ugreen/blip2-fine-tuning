"""
csv_to_jsonl.py — Convert the DAQUAR CSV annotation file to JSONL format.

The DAQUAR dataset from Kaggle (bhavikardeshna/visual-question-answering-
computer-vision-nlp) contains a CSV with columns: image_id, question, answer.
Images are stored as dataset/images/<image_id>.jpg.

The output JSONL is shuffled with a fixed seed so that any downstream
train/eval split made by index ([:N] / [N:]) is statistically balanced
and reproducible.

Usage
-----
    python csv_to_jsonl.py
"""

import csv
import json
import os
import random

# ============================================================
# Configuration
# ============================================================
INPUT_CSV    = os.path.join("dataset", "data.csv")
OUTPUT_JSONL = os.path.join("dataset", "data.jsonl")
IMAGE_DIR    = os.path.join("dataset", "images")

# The DAQUAR Kaggle dataset stores images as .jpg.
# If your code uses .png, change this to ".png".
IMAGE_EXT    = ".png"

# Fixed seed ensures the shuffle — and therefore every downstream
# train/eval split — is identical across machines and runs.
SHUFFLE_SEED = 42
# ============================================================


def find_image_path(image_id: str) -> str:
    """
    Return the image path for a given image_id.
    Tries the configured extension first, then falls back to the other.
    Raises FileNotFoundError if neither exists.
    """
    primary = os.path.join(IMAGE_DIR, image_id + IMAGE_EXT)
    if os.path.exists(primary):
        return primary

    # Fallback: try the other common extension
    fallback_ext = ".png" if IMAGE_EXT == ".jpg" else ".jpg"
    fallback = os.path.join(IMAGE_DIR, image_id + fallback_ext)
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Image not found for id '{image_id}'. "
        f"Tried: {primary} and {fallback}"
    )


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    records = []
    missing_images = []

    with open(INPUT_CSV, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_num, row in enumerate(reader, start=2):   # row 1 is header
            image_id = row["image_id"].strip()
            question = row["question"].strip()
            answer   = row["answer"].strip()

            if not image_id or not question or not answer:
                print(f"  [WARN] Row {row_num}: empty field — skipped.")
                continue

            try:
                image_path = find_image_path(image_id)
            except FileNotFoundError as e:
                missing_images.append(str(e))
                continue

            records.append({
                "image_path": image_path,
                "question":   question,
                "answer":     answer,
            })

    if missing_images:
        print(f"\n[WARN] {len(missing_images)} image(s) not found and skipped:")
        for msg in missing_images[:10]:
            print(f"  {msg}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more.")

    # ── Shuffle with fixed seed ───────────────────────────────
    # This is the single most important step for reproducibility:
    # the CSV is ordered by image ID, so without shuffling the first
    # TRAIN_SIZE records and the remaining eval records come from
    # systematically different image IDs, biasing the split.
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(records)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nConversion complete.")
    print(f"  Records written : {len(records)}")
    print(f"  Output path     : {OUTPUT_JSONL}")
    print(f"  Shuffle seed    : {SHUFFLE_SEED}")


if __name__ == "__main__":
    main()