from pathlib import Path
import torch
from transformers import MusicgenForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "facebook/musicgen-small"
OUTPUT_DIR = Path("outputs/checkpoints/lora_smoke_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading MusicGen...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

print("Adding LoRA adapters...")

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)

print("\nTrainable parameters after LoRA:")
model.print_trainable_parameters()

print("\nSaving LoRA adapter smoke-test checkpoint...")
model.save_pretrained(OUTPUT_DIR)

print(f"Saved LoRA adapter to: {OUTPUT_DIR}")
print("Smoke test complete.")