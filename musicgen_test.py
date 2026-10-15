import torch
from pathlib import Path
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

OUTPUT_DIR = Path("outputs/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/musicgen-small"
PROMPT = ["gentle emotional orchestral soundtrack with soft piano and strings"]

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

print("Preparing inputs...")
inputs = processor(
    text=PROMPT,
    padding=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}
print("Generating 20-second audio...")
audio_values = model.generate(
    **inputs,
    max_new_tokens=1000
)

audio = audio_values[0, 0].cpu().numpy()
sampling_rate = model.config.audio_encoder.sampling_rate

output_path = OUTPUT_DIR / "musicgen_test_20sec.wav"
wavfile.write(output_path, rate=sampling_rate, data=audio)

print(f"Saved test audio to: {output_path}")