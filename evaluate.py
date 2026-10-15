from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
# -----------------------------
# Paths
# -----------------------------
METADATA_PATH = Path("data/metadata/metadata.csv")
FIGURE_DIR = Path("outputs/figures")
SAMPLE_DIR = Path("outputs/samples")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load metadata
# -----------------------------
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

metadata = pd.read_csv(METADATA_PATH)

print("Loaded metadata:")
print(metadata.head())
print("\nDataset summary:")
print(metadata["style_label"].value_counts())

# -----------------------------
# 1. Dataset count chart
# -----------------------------
counts = metadata["style_label"].value_counts()

plt.figure(figsize=(6, 4))
counts.plot(kind="bar")
plt.title("Number of Processed Clips per Style")
plt.xlabel("Style")
plt.ylabel("Number of Clips")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "dataset_style_counts.png")
plt.close()

print(f"Saved: {FIGURE_DIR / 'dataset_style_counts.png'}")

# -----------------------------
# Helper: save spectrogram
# -----------------------------
def save_spectrogram(audio_path, output_name, title):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"Skipping missing file: {audio_path}")
        return

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    plt.figure(figsize=(10, 4))
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    librosa.display.specshow(
        spec_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name)
    plt.close()

    print(f"Saved: {FIGURE_DIR / output_name}")

# -----------------------------
# 2. Source spectrograms
# -----------------------------
style_a_file = metadata[metadata["style_label"] == "style_a"]["filepath"].iloc[0]
style_b_file = metadata[metadata["style_label"] == "style_b"]["filepath"].iloc[0]

save_spectrogram(
    style_a_file,
    "spectrogram_source_style_a.png",
    "Spectrogram of Source Style A Clip"
)

save_spectrogram(
    style_b_file,
    "spectrogram_source_style_b.png",
    "Spectrogram of Source Style B Clip"
)

# -----------------------------
# 3. Find latest generated hybrid output
# -----------------------------
run_folders = sorted(
    [p for p in SAMPLE_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

latest_hybrid = None

for folder in run_folders:
    possible_files = list(folder.glob("*hybrid*.wav"))
    if possible_files:
        latest_hybrid = possible_files[0]
        break

if latest_hybrid:
    save_spectrogram(
        latest_hybrid,
        "spectrogram_generated_hybrid.png",
        "Spectrogram of Generated Hybrid Output"
    )
else:
    print("No generated hybrid wav found. Skipping hybrid spectrogram.")

# -----------------------------
# 4. Create evaluation template
# -----------------------------
evaluation_rows = [
    {
        "sample": "style_a",
        "coherence_1_5": "",
        "style_a_similarity_1_5": "",
        "style_b_similarity_1_5": "",
        "hybrid_success_1_5": "",
        "audio_quality_1_5": "",
        "notes": "",
    },
    {
        "sample": "style_b",
        "coherence_1_5": "",
        "style_a_similarity_1_5": "",
        "style_b_similarity_1_5": "",
        "hybrid_success_1_5": "",
        "audio_quality_1_5": "",
        "notes": "",
    },
    {
        "sample": "hybrid",
        "coherence_1_5": "",
        "style_a_similarity_1_5": "",
        "style_b_similarity_1_5": "",
        "hybrid_success_1_5": "",
        "audio_quality_1_5": "",
        "notes": "",
    },
    {
        "sample": "hybrid_lora",
        "coherence_1_5": "",
        "style_a_similarity_1_5": "",
        "style_b_similarity_1_5": "",
        "hybrid_success_1_5": "",
        "audio_quality_1_5": "",
        "notes": "",
    },
]

evaluation_df = pd.DataFrame(evaluation_rows)
evaluation_path = FIGURE_DIR / "evaluation_template.csv"
evaluation_df.to_csv(evaluation_path, index=False)

print(f"Saved: {evaluation_path}")

print("\nEvaluation complete.")