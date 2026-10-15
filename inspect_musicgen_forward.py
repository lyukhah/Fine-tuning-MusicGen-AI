from pathlib import Path
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_NAME = "facebook/musicgen-small"
AUDIO_FILE = list(Path("data/processed/style_a").glob("*.wav"))[0]

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

print("decoder_start_token_id:", getattr(model.config.decoder, "decoder_start_token_id", None))
print("num_codebooks:", model.config.decoder.num_codebooks)
print("vocab_size:", model.config.decoder.vocab_size)

audio, sr = sf.read(AUDIO_FILE)
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

audio = audio[:80000]

audio_inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

with torch.no_grad():
    encoded = model.audio_encoder.encode(**audio_inputs)
    codes = encoded.audio_codes

print("codes shape:", codes.shape)
print("codes min/max:", codes.min().item(), codes.max().item())

labels1 = codes.squeeze(1).long()
labels2 = codes.squeeze(0).squeeze(0).long()
labels3 = codes.squeeze(1).permute(0, 2, 1).reshape(1, -1).long()

print("labels1:", labels1.shape)
print("labels2:", labels2.shape)
print("labels3:", labels3.shape)