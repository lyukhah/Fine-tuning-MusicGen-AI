from pathlib import Path
from datetime import datetime
import csv
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"
OUTPUT_DIR = Path("outputs/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_DIR / f"longform_baseline_test_{RUN_ID}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "intimate fantasy soundtrack with minimalist reflective piano, warm strings, "
    "soft woodwinds, elegant spacious harmony, delicate emotional expression, "
    "refined contemplative atmosphere"
)

TOKEN_TESTS = {
    "short_750_tokens": 750,
    "medium_1500_tokens": 1500,
    "long_2500_tokens": 2500,
    "extended_3500_tokens": 3500,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

rows = []

for name, tokens in TOKEN_TESTS.items():
    print(f"\nGenerating {name} with {tokens} tokens")

    inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=tokens)

    audio = audio_values[0, 0].detach().cpu().numpy()
    sr = model.config.audio_encoder.sampling_rate

    output_path = RUN_OUTPUT_DIR / f"{name}.wav"
    wavfile.write(output_path, rate=sr, data=audio)

    rows.append({
        "file": str(output_path).replace("\\", "/"),
        "prompt": PROMPT,
        "max_new_tokens": tokens,
        "model": MODEL_NAME,
        "device": device,
        "sampling_rate": sr,
    })

    print(f"Saved: {output_path}")

metadata_path = RUN_OUTPUT_DIR / "longform_metadata.csv"
with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "prompt", "max_new_tokens", "model", "device", "sampling_rate"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print(f"Run folder: {RUN_OUTPUT_DIR}")