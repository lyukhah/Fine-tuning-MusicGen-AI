print("prepare_dataset.py is ready")
from pathlib import Path
import csv
import librosa
import soundfile as sf

RAW_STYLE_A_DIR = Path("data/raw/style_a")
RAW_STYLE_B_DIR = Path("data/raw/style_b")

PROCESSED_DIR = Path("data/processed")
METADATA_DIR = Path("data/metadata")
METADATA_CSV = METADATA_DIR / "metadata.csv"

TARGET_SAMPLE_RATE = 32000
CLIP_DURATION = 20

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_files(folder):
    files = []
    if folder.exists():
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file)
    return sorted(files)


def process_file(file_path, style_label):
    print(f"Processing {file_path.name}")

    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Failed: {e}")
        return []

    samples_per_clip = TARGET_SAMPLE_RATE * CLIP_DURATION
    total_samples = len(audio)

    num_clips = total_samples // samples_per_clip

    if num_clips == 0:
        print("Skipped (too short)")
        return []

    output_rows = []

    style_folder = PROCESSED_DIR / style_label
    style_folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_clips):
        start = i * samples_per_clip
        end = start + samples_per_clip
        clip = audio[start:end]

        filename = f"{file_path.stem}_clip_{i+1:03d}.wav"
        save_path = style_folder / filename

        sf.write(save_path, clip, TARGET_SAMPLE_RATE)

        output_rows.append({
            "filepath": str(save_path).replace("\\", "/"),
            "style_label": style_label,
            "source_file": file_path.name,
            "clip_name": filename
        })

    print(f"Created {len(output_rows)} clips")
    return output_rows


def main():
    rows = []

    style_a_files = get_audio_files(RAW_STYLE_A_DIR)
    style_b_files = get_audio_files(RAW_STYLE_B_DIR)

    print(f"Found {len(style_a_files)} style_a files")
    print(f"Found {len(style_b_files)} style_b files")

    for file in style_a_files:
        rows.extend(process_file(file, "style_a"))

    for file in style_b_files:
        rows.extend(process_file(file, "style_b"))

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "style_label", "source_file", "clip_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone")
    print(f"Total clips: {len(rows)}")
    print(f"Metadata saved to {METADATA_CSV}")


if __name__ == "__main__":
    main()