# Fine-tuning MusicGen AI on Hybrid and Individual Music Styles for Soundtrack Composition

Final-year Computer Science dissertation project by Lim Xiao Yu, University of the Arts London.

This project investigates prompted generation and parameter-efficient adaptation of MusicGen-small for fantasy-orchestral, minimalist reflective piano, and hybrid soundtrack ideas. It contains Python workflows for audio preprocessing, LoRA training, generation experiments, and spectrogram analysis.

## Experiments

| Approach | Clips selected | Epochs | LoRA rank | Audio used per clip |
| --- | --- | --- | --- | --- |
| Prompted baseline | No local training data | Not applicable | Not applicable | Not applicable |
| Small-scale LoRA | Up to 5 per style, 10 total | 3 | 2 | Up to 5 seconds |
| Full-dataset LoRA | Up to 40 per style, 80 total | 5 | 4 | Up to 10 seconds |

Both training scripts use a learning rate of `5e-5` and target `q_proj`, `k_proj`, `v_proj`, and `out_proj`. The supplied metadata contains 80 entries, split evenly between `style_a` and `style_b`. The full-dataset experiment still trains LoRA adapters; it is not full-parameter model fine-tuning.

## What is included

- Dataset metadata and preprocessing code.
- Baseline and adapter-based generation scripts with CSV run metadata.
- Prompt-variation, repeated-generation, generation-length, and empty/neutral-prompt experiments.
- Training notes and saved spectrogram figures.
- An unfilled listening-evaluation template.

Training audio, generated WAV files, trained adapters, completed evaluation scores, and full run logs are not included. The scripts show the implementation, but their presence alone does not establish that every experiment completed successfully.

## Technologies

Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, Librosa, NumPy, pandas, SciPy, SoundFile, and Matplotlib.

## Setup

Run commands from the repository root. Create a virtual environment using a Python version compatible with your chosen PyTorch and Transformers releases:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The original dependency versions were not recorded. This is a dependency list, not a verified lockfile; compatibility must be checked before reproducing the work. Model downloads require network access. The project notes describe an NVIDIA RTX 3060 Laptop GPU; scripts select CUDA when available and otherwise CPU. They do not explicitly select Apple MPS acceleration.

## Baseline generation

```bash
python src/generate.py
```

This downloads `facebook/musicgen-small` and generates the configured individual-style and hybrid prompts under timestamped `outputs/samples/` folders. Prompt text, token limit, model, device and sample rate are recorded in CSV files. Runtime and memory depend on hardware and dependency versions.

## Dataset preparation and training

Supply audio you are permitted to use under `data/raw/style_a/` and `data/raw/style_b/`, then run:

```bash
python src/prepare_dataset.py
python src/train_lora_steps.py
# Larger experiment, when ready:
python src/train_lora_full80.py
python src/generate_lora.py
```

Preprocessing converts audio to mono, resamples to 32 kHz, and splits it into complete 20-second clips, discarding shorter remainders. It overwrites `data/metadata/metadata.csv` for the supplied data. Training selects the first sorted files in each style folder and truncates each clip to the lengths in the table above.

`generate_lora.py` defaults to the 80-clip adapter directory. To inspect the small-scale adapter, change `LORA_ADAPTER_PATH` and the output labels in that script. Checkpoints must exist before adapter-based generation.

## Evaluation and figures

```bash
python src/generate_hybrid_variants.py
python src/generate_repeatability_test.py
python src/generate_longform_test.py
python src/evaluate.py
```

These are research experiment scripts, not an automated assertion-based test suite. Repeated generation makes five calls with the same prompt; it does not set or record random seeds. The length experiment requests 750, 1500, 2500 and 3500 tokens; longer requests are exploratory and may exceed model or library generation limits.

`evaluate.py` reads the metadata, plots dataset counts and mel spectrograms, and writes an unfilled listening-evaluation CSV. It requires the referenced audio to reproduce source plots and expects `outputs/samples/` to exist. It overwrites `evaluation_template.csv`, so save completed ratings separately.

### Existing figures

![Dataset counts](outputs/figures/dataset_style_counts.png)
![Generated hybrid spectrogram](outputs/figures/spectrogram_generated_hybrid.png)

Spectrograms visualise audio features; they do not independently establish perceived quality or stylistic success. See [experiment notes](docs/notes.md) for recorded observations and [reproducibility notes](docs/reproducibility.md) for the limits of this snapshot.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/prepare_dataset.py` | Audio segmentation, resampling and metadata |
| `src/train_lora_steps.py` | Small-scale training |
| `src/train_lora_full80.py` | Larger training experiment |
| `src/generate*.py` | Generation experiments |
| `src/evaluate.py` | Plots and evaluation template |
| `src/inspect*.py` | Model and token-shape diagnostics |
| `src/train_lora.py` | Adapter attachment/save smoke test; no training loop |
| `src/train.py` | Empty historical placeholder |
| `config/config.yaml` | Historical configuration; main scripts use their own constants |
| `docs/notes.md` | Original experiment notes |

## Sharing and attribution

Source recordings and model checkpoints are excluded from this repository. Check permissions before adding audio examples or third-party data. Model and dependency licences apply separately. No blanket open-source licence has been assigned to this portfolio snapshot.
