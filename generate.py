from pathlib import Path
from datetime import datetime
import csv
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# -----------------------------
# Output setup
# -----------------------------
OUTPUT_DIR = Path("outputs/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_DIR / f"run_{RUN_ID}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Model settings
# -----------------------------
MODEL_NAME = "facebook/musicgen-small"
MAX_NEW_TOKENS = 1500

PROMPTS = {
    "style_a_baseline": (
        "emotional fantasy folklore soundtrack, warm strings, woodwinds, "
        "gentle cinematic melody, nostalgic medieval world atmosphere"
    ),

    "style_b_baseline": (
        "minimalist reflective piano composition, elegant modern classical harmony, "
        "spacious atmosphere, delicate emotional expression"
    ),

    "hybrid_intimate_baseline": (
        "intimate fantasy soundtrack with minimalist reflective piano, warm strings, "
        "soft woodwinds, elegant spacious harmony, delicate emotional expression, "
        "refined contemplative atmosphere"
    ),

    "hybrid_melancholic_baseline": (
        "melancholic fantasy soundtrack with minimalist piano, warm strings, "
        "nostalgic woodwinds, spacious modern classical harmony, quiet cinematic emotion"
    ),

    "hybrid_hopeful_baseline": (
        "hopeful fantasy soundtrack with minimalist piano, warm strings, "
        "gentle woodwinds, elegant spacious harmony, uplifting contemplative atmosphere"
    ),
}

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load model
# -----------------------------
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

metadata_rows = []

# -----------------------------
# Generate audio
# -----------------------------
for name, prompt in PROMPTS.items():
    print(f"\nGenerating: {name}")
    print(f"Prompt: {prompt}")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    audio_values = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    audio = audio_values[0, 0].detach().cpu().numpy()
    sampling_rate = model.config.audio_encoder.sampling_rate

    output_path = RUN_OUTPUT_DIR / f"{name}.wav"
    wavfile.write(output_path, rate=sampling_rate, data=audio)

    metadata_rows.append({
        "file": str(output_path).replace("\\", "/"),
        "prompt_type": name,
        "prompt": prompt,
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": device,
        "sampling_rate": sampling_rate,
    })

    print(f"Saved: {output_path}")

# -----------------------------
# Save metadata
# -----------------------------
metadata_path = RUN_OUTPUT_DIR / "generation_metadata.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "prompt_type",
            "prompt",
            "model",
            "max_new_tokens",
            "device",
            "sampling_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print("\nDone.")
print(f"Run folder: {RUN_OUTPUT_DIR}")
print(f"Metadata saved to: {metadata_path}")