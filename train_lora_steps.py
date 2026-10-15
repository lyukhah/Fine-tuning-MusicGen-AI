from pathlib import Path
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import LoraConfig, get_peft_model

# -----------------------------
# Settings
# -----------------------------
MODEL_NAME = "facebook/musicgen-small"

# Use hybrid dataset: 5 clips from A + 5 clips from B
STYLE_A_DIR = Path("data/processed/style_a")
STYLE_B_DIR = Path("data/processed/style_b")

OUTPUT_DIR = Path("outputs/checkpoints/lora_10clip_3epoch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_PROMPT = (
    "beautiful emotional fantasy soundtrack with minimalist reflective piano, "
    "warm strings, elegant spacious harmony and contemplative atmosphere"
)

# Safer training settings for RTX 3060 6GB
NUM_A_FILES = 5
NUM_B_FILES = 5
NUM_EPOCHS = 3
MAX_AUDIO_SAMPLES = 160000  # 5 seconds at 32kHz
LEARNING_RATE = 5e-5

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load processor and model
# -----------------------------
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading MusicGen model...")
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)

# -----------------------------
# Add LoRA adapters
# -----------------------------
print("Adding LoRA adapters...")

lora_config = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)

# Required for label-based training
base_model = model.base_model.model
if getattr(base_model.config.decoder, "decoder_start_token_id", None) is None:
    base_model.config.decoder.decoder_start_token_id = 2048

model.to(device)
model.train()

print("\nTrainable parameters:")
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Dataset subset
# -----------------------------
style_a_files = sorted(list(STYLE_A_DIR.glob("*.wav")))[:NUM_A_FILES]
style_b_files = sorted(list(STYLE_B_DIR.glob("*.wav")))[:NUM_B_FILES]
files = style_a_files + style_b_files

print(f"\nTraining on {len(files)} clips")
print(f"Style A clips: {len(style_a_files)}")
print(f"Style B clips: {len(style_b_files)}")

if len(files) == 0:
    raise FileNotFoundError("No wav files found in processed dataset folders.")

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

    for idx, file in enumerate(files, start=1):
        print(f"\nClip {idx}/{len(files)}: {file.name}")

        audio, sr = sf.read(file)

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Shorten audio to reduce VRAM load
        audio = audio[:MAX_AUDIO_SAMPLES]

        print(f"Audio samples used: {len(audio)}")
        print(f"Sample rate: {sr}")

        # Encode audio into discrete EnCodec tokens
        audio_inputs = processor(
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            encoded = base_model.audio_encoder.encode(**audio_inputs)
            labels = encoded.audio_codes

        # Shape: [batch, frame, codebook, tokens] -> [batch, tokens, codebook]
        labels = labels.squeeze(1).permute(0, 2, 1).long()

        print(f"Training label shape: {labels.shape}")

        # Text conditioning
        text_inputs = processor(
            text=[TEXT_PROMPT],
            padding=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        outputs = model(
            **text_inputs,
            labels=labels,
        )

        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(files)
    print(f"\nEpoch {epoch + 1} average loss: {avg_loss:.4f}")

# -----------------------------
# Save trained adapter
# -----------------------------
print("\nSaving trained LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)

print(f"Saved trained LoRA adapter to: {OUTPUT_DIR}")
print("10-clip LoRA fine-tuning run complete.")