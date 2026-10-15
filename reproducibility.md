# Reproducibility and evidence

This repository is a dissertation snapshot, not a validated production pipeline.

## Evidence included

- Metadata: 80 rows, 40 for each style.
- Training scripts: 10-clip and 80-clip LoRA configurations.
- Original notes: documented tiny proof-of-concept and 10-clip training run; the notes do not document the complete 80-clip result.
- Evaluation assets: dataset chart, source/generated spectrograms, and a blank rating template.

The archive does not include logs or adapters that independently verify the full training run, nor generated audio or completed listener ratings. Any dissertation conclusions should be linked to the report and supporting outputs when available. Initial listening observations in the original notes are subjective, not measured improvement claims.

## Experimental limitations

The small and larger runs differ in dataset size, epochs, rank and audio duration. Their comparison cannot isolate a dataset-size effect. Both use one shared hybrid text prompt for all training clips and a fixed sorted order, with Style A followed by Style B. There is no explicit held-out split or automated quality metric in the supplied training scripts.

The repeated-generation script has no explicit seed control; it explores repeated outputs rather than proving deterministic reproducibility. The empty/neutral-prompt script loads both models simultaneously, which can be memory-intensive. Long generation requests may fail depending on the installed model/library limits.

The YAML contains a 15-second clip duration, while preprocessing actually uses 20 seconds; it is not loaded by the main scripts. Actual executable settings are the Python constants, including the 5- and 10-second training truncation limits.

## Portfolio preparation changes

- Expanded README with the workflow, experiment settings, existing figures and limitations.
- Added missing `transformers` and `peft` dependencies without inventing version pins.
- Extended ignore rules for environment secrets and common local outputs.
- Preserved all experimental Python scripts and original notes.

Validation during preparation: all 16 Python files parsed successfully; CSV counts and local README asset paths were checked. No models were downloaded and no training or generation was rerun. Runtime compatibility and audio quality have not been verified in this review.
