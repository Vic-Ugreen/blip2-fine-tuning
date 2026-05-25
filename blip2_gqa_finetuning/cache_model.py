from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
print("Downloading processor...")
AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
print("Downloading model weights (~15 GB, will take a few minutes)...")
Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.bfloat16,
)
print("Done — model cached in ~/hf_cache/")