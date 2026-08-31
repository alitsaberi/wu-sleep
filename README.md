# WU-Sleep

Inference API for WU-Sleep, a domain-adapted sleep staging model for forehead wearable EEG.

## Requirements

- **Python:** `>=3.10`

## Install

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

## Input

Pass EEG as `(n_samples, n_channels)`.

- **Single channel** — one forehead derivation, shape `(n_samples, 1)`.
- **Multiple channels** — each derivation as a column. Each column is scored independently; posteriors are summed and renormalized per epoch. **Column order does not matter.**

WU-Sleep is intended for **forehead wearable EEG** with bipolar derivations similar to those used in training (e.g. left/right frontal sites referenced to Fpz). The reference montage from the preprint (Hypnodyne ZMax) is F7–Fpz and F8–Fpz. Similar montages on other devices may work, but performance outside the validated setting has not been established.

**Units.** Preprocessing applies per-channel robust scaling (median 0, IQR 1) after resampling and band-pass filtering. The API does not require a specific voltage unit (µV, mV, V, etc.) as long as **each channel uses a consistent unit** throughout the recording.

**Preprocessing** (applied once to all channels before scoring): resample to 128 Hz, 0.3–35 Hz band-pass filter, robust scaling, IQR clipping (see the preprint, Section 2.2).

## Output

Each recording is scored in non-overlapping **30 s epochs**. If the signal length is not an integer number of 30 s epochs after preprocessing, the final partial epoch is edge-padded to 30 s and still scored.

`score_sleep_stages(..., output="probs")` returns a float64 array of shape `(n_epochs, n_classes)`. Rows sum to 1.

`score_sleep_stages(..., output="labels")` returns an object array of shape `(n_epochs,)` with one label per epoch.

**Class order** (columns of `probs`, index order for argmax):

| Index | Label |
|-------|-------|
| 0 | W |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

This matches `class_labels` in `model/wu-sleep.yaml`.

## Model

The fine-tuned ONNX weights ship in this repository as **`model/wu-sleep.onnx`**, with sidecar metadata in **`model/wu-sleep.yaml`**. Clone the repo or download a release tag to obtain both files.

When calling `score_sleep_stages`, the default `model_path` is `model/wu-sleep.onnx` (relative to your working directory). Pass an absolute path if you run from elsewhere.

## Usage

```python
from wu_sleep import score_sleep_stages

# Single channel
labels = score_sleep_stages(
    eeg[:, :1],
    sample_rate_hz=256.0,
    model_path="model/wu-sleep.onnx",
    channel_names=["EEG_L"],
    output="labels",
)

# Multiple channels (order arbitrary)
labels = score_sleep_stages(
    eeg,
    sample_rate_hz=256.0,
    model_path="model/wu-sleep.onnx",
    channel_names=["EEG_L", "EEG_R"],
    output="labels",
)
```

## Example

```bash
uv run python examples/score_recording.py
```

## Citation

If you use WU-Sleep in your research, please cite:

> **WU-Sleep citation will be added when the preprint becomes available.**

## Model lineage

WU-Sleep builds on the U-Sleep architecture and was fine-tuned from the SLEEPYLAND `u-sleep-nsrr-2024_eeg` checkpoint.

When describing the model architecture or pretrained checkpoint, please also cite the corresponding upstream work:

* **U-Sleep:** Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*, 4, 72. https://doi.org/10.1038/s41746-021-00440-5
* **SLEEPYLAND:** Rossi, A. D., Metaldi, M., Bechny, M., et al. (2026). SLEEPYLAND: trust begins with fair evaluation of automatic sleep staging models. *npj Digital Medicine*, 9, 55. https://doi.org/10.1038/s41746-025-02237-2

## License

This repository is released under the [MIT License](LICENSE).

## TODO

- [ ] Release data preparation, training, and evaluation code
- [ ] Integrate artifact detection at inference

