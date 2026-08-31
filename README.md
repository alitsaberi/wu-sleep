# WU-Sleep

Inference API for WU-Sleep, a domain-adapted sleep staging model for single-channel forehead wearable EEG.

## Install

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

## Usage

```python
import numpy as np
from wu_sleep import run_sleep_scoring

labels = run_sleep_scoring(
    eeg,  # shape (n_samples, 1)
    sample_rate_hz=256.0,
    model_path="model/model.onnx",
    output="labels",
)
```

Download `model.onnx` from Hugging Face and place it in `model/` alongside `model.yaml`.

Recordings are scored in non-overlapping 30 s epochs at 128 Hz. If the signal length is not an integer number of 30 s epochs after preprocessing, the final partial epoch is edge-padded to 30 s and still scored.

## Example

```bash
uv run python examples/score_recording.py
```
