from pathlib import Path
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"
AUDIO_FILE = list(Path("data/processed/style_a").glob("*.wav"))[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

audio, sr = sf.read(AUDIO_FILE)
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

print("Audio shape:", audio.shape)
print("Sample rate:", sr)

inputs = processor(
    audio=audio,
    sampling_rate=sr,
    return_tensors="pt"
)

print("Processor keys:", inputs.keys())
for k, v in inputs.items():
    print(k, v.shape, v.dtype)

print("\nTrying audio encoder...")
with torch.no_grad():
    audio_inputs = {k: v.to(device) for k, v in inputs.items()}
    encoded = model.audio_encoder.encode(**audio_inputs)

print("Encoded type:", type(encoded))
print(encoded)

print("Done.")