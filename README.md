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

## Example

```bash
uv run python examples/score_recording.py
```
