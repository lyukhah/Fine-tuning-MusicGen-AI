from pathlib import Path
from datetime import datetime
import csv
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

MODEL_NAME = "facebook/musicgen-small"
LORA_ADAPTER_PATH = "outputs/checkpoints/lora_80clip_5epoch_r4"

OUTPUT_DIR = Path("outputs/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = OUTPUT_DIR / f"no_prompt_test_{RUN_ID}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 1500

TESTS = {
    "baseline_empty_prompt": {
        "mode": "baseline",
        "prompt": "",
    },
    "baseline_neutral_prompt": {
        "mode": "baseline",
        "prompt": "instrumental music",
    },
    "baseline_hybrid_intimate_prompt": {
        "mode": "baseline",
        "prompt": (
            "intimate fantasy soundtrack with minimalist reflective piano, warm strings, "
            "soft woodwinds, elegant spacious harmony, delicate emotional expression, "
            "refined contemplative atmosphere"
        ),
    },
    "lora_empty_prompt": {
        "mode": "lora",
        "prompt": "",
    },
    "lora_neutral_prompt": {
        "mode": "lora",
        "prompt": "instrumental music",
    },
    "lora_hybrid_intimate_prompt": {
        "mode": "lora",
        "prompt": (
            "intimate fantasy soundtrack with minimalist reflective piano, warm strings, "
            "soft woodwinds, elegant spacious harmony, delicate emotional expression, "
            "refined contemplative atmosphere"
        ),
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading baseline model...")
baseline_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
baseline_model.to(device)
baseline_model.eval()

print("Loading LoRA model...")
base_for_lora = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
lora_model = PeftModel.from_pretrained(base_for_lora, LORA_ADAPTER_PATH)
lora_model.to(device)
lora_model.eval()

metadata_rows = []

for name, test in TESTS.items():
    mode = test["mode"]
    prompt = test["prompt"]

    print(f"\nGenerating: {name}")
    print(f"Mode: {mode}")
    print(f"Prompt: {repr(prompt)}")

    model = baseline_model if mode == "baseline" else lora_model

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    audio = audio_values[0, 0].detach().cpu().numpy()

    if mode == "baseline":
        sampling_rate = model.config.audio_encoder.sampling_rate
        adapter = "none"
    else:
        sampling_rate = model.base_model.model.config.audio_encoder.sampling_rate
        adapter = LORA_ADAPTER_PATH

    output_path = RUN_OUTPUT_DIR / f"{name}.wav"
    wavfile.write(output_path, rate=sampling_rate, data=audio)

    metadata_rows.append({
        "file": str(output_path).replace("\\", "/"),
        "test_name": name,
        "mode": mode,
        "prompt": prompt,
        "model": MODEL_NAME,
        "adapter": adapter,
        "max_new_tokens": MAX_NEW_TOKENS,
        "device": device,
        "sampling_rate": sampling_rate,
    })

    print(f"Saved: {output_path}")

metadata_path = RUN_OUTPUT_DIR / "no_prompt_test_metadata.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "test_name",
            "mode",
            "prompt",
            "model",
            "adapter",
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