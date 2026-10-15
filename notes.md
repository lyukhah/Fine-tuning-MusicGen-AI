## LoRA Fine-Tuning Attempt

A lightweight fine-tuning experiment was carried out using LoRA adapters on MusicGen-small.

The model successfully attached LoRA adapters to the attention projection layers:

- q_proj
- k_proj
- v_proj
- out_proj

The tiny training test used:
- 1 processed audio clip
- 1 epoch
- approximately 2.5 seconds of audio
- CUDA GPU acceleration
- RTX 3060 Laptop GPU

Training completed successfully.

Result:
- Trainable parameters: 786,432
- Total parameters: 587,671,106
- Trainable percentage: 0.1338%
- Loss: 9.7302
- Adapter saved to: outputs/checkpoints/lora_tiny_success_test

This confirms that parameter-efficient lightweight fine-tuning was successfully implemented as a proof-of-concept under hardware constraints.

## 10-Clip LoRA Fine-Tuning Run

A second LoRA fine-tuning experiment was completed using a 10-clip subset of the processed dataset:

- 5 clips from Style A
- 5 clips from Style B
- 3 epochs
- 5 seconds per clip
- CUDA acceleration on RTX 3060 Laptop GPU
- LoRA target modules: q_proj, k_proj, v_proj, out_proj
- LoRA rank: 2
- Learning rate: 5e-5

The adapter was saved to:

`outputs/checkpoints/lora_10clip_3epoch`

Initial listening suggested that the hybrid output was the strongest result, while Style B became less polished than the baseline. This suggests the adapter was more effective for hybrid stylistic direction than for preserving the clean minimalist quality of Style B alone.