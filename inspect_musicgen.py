import torch
from transformers import MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading model...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

print("\nTrainable parameters BEFORE LoRA:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("\nPossible LoRA target modules:")
for name, module in model.named_modules():
    lowered = name.lower()
    if any(key in lowered for key in ["q_proj", "k_proj", "v_proj", "out_proj", "linear"]):
        print(name)

print("\nDone.")