# WU-Sleep

WU-Sleep is a domain-adapted sleep staging model for single-channel forehead wearable EEG. It adapts U-Sleep, originally developed using conventional polysomnography, to the different electrode placement and signal characteristics in wearable forehead recordings.

This repository provides the fine-tuned model together with a lightweight Python interface for preprocessing EEG, running inference, and combining predictions from multiple forehead channels.

## Model description

The model itself is single-channel. It processes each EEG derivation independently and predicts one of five sleep stages for every 30-second epoch. When multiple channels are provided, their posterior probabilities are summed and renormalized per epoch before the final label is assigned.

| Property              | Value                               |
| --------------------- | ----------------------------------- |
| Input                 | Single-channel bipolar forehead EEG |
| Context window        | 35 consecutive 30-second epochs     |
| Prediction resolution | 30 seconds                          |
| Output classes        | `W`, `N1`, `N2`, `N3`, `REM`        |

## Requirements

- **Python:** `>=3.10`

## Installation

Using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

With example dependencies (EDF I/O and plotting):

```bash
uv sync --extra examples
```

Or with pip:

```bash
pip install .
pip install ".[examples]"
```

## EEG input

Pass EEG as a NumPy array with shape:

```text
(n_samples, n_channels)
```

The typical configurations are:

- **Single channel:** one bipolar forehead derivation, such as `F7-Fpz` or `F8-Fpz`, with shape `(n_samples, 1)`.
- **Multiple channels:** each derivation supplied as a separate column. Each channel is scored independently, and the posteriors are combined per epoch. Column order does not affect the result.

The reference montage used to develop and evaluate WU-Sleep consists of the following Hypnodyne ZMax derivations:

```text
F7 - Fpz
F8 - Fpz
```

The API does not construct or re-reference EEG derivations. Input signals must already represent the intended bipolar channels. The `channel_names` argument is descriptive metadata only.

### Other EEG systems

WU-Sleep is not tied to the ZMax file format or its channel names. It can process recordings from other EEG systems when they provide comparable bipolar forehead signals, particularly derivations between a lateral forehead electrode and a midline forehead reference.

Differences in electrode placement, reference location, polarity, electrode material, amplifier characteristics, or recording environment may affect performance. Comparable signals should use the same polarity as the reference montage:

```text
lateral forehead electrode - midline forehead reference
```

Performance on other devices and montages has not yet been independently established.

### Units

Preprocessing applies robust scaling independently to each channel after resampling and filtering. The API therefore does not require a specific voltage unit, such as µV, mV, or V, provided that each channel uses a consistent unit throughout the recording.

### Preprocessing

The inference pipeline applies:

1. Resampling to 128 Hz.
2. Band-pass filtering between 0.3 and 35 Hz.
3. Per-channel robust scaling to a median of 0 and an IQR of 1.
4. IQR-based clipping.

An original sampling rate of at least 128 Hz is recommended.

## Output

Recordings are scored in non-overlapping 30-second epochs. If the preprocessed signal length is not an integer number of epochs, the final partial epoch is edge-padded to 30 seconds and still scored.

```python
score_sleep_stages(..., output="probs")
```

returns a `float64` array with shape `(n_epochs, n_classes)`. Each row contains normalized class probabilities and sums to 1.

```python
score_sleep_stages(..., output="labels")
```

returns an object array with shape `(n_epochs,)`, containing one sleep stage label per epoch.

To convert existing probability arrays without re-running inference:

```python
from wu_sleep import labels_from_probabilities

indices = labels_from_probabilities(probs)  # int64 class indices
labels = labels_from_probabilities(probs, ["W", "N1", "N2", "N3", "REM"])
```

The probability columns follow this order:


| Index | Label |
| ----- | ----- |
| 0     | `W`   |
| 1     | `N1`  |
| 2     | `N2`  |
| 3     | `N3`  |
| 4     | `REM` |


This order matches `class_labels` in `model/wu-sleep.yaml`.

## Model files

The fine-tuned model and its metadata are included in this repository:

```text
model/
├── wu-sleep.onnx
└── wu-sleep.yaml
```

Clone the repository or download a versioned release to obtain both files.

The default model path is:

```text
model/wu-sleep.onnx
```

This path is resolved relative to the current working directory. Pass an absolute path when running the API from another location.

## Usage

```python
from wu_sleep import score_sleep_stages

# Single-channel scoring
labels = score_sleep_stages(
    eeg[:, :1],
    sample_rate_hz=256.0,
    model_path="model/wu-sleep.onnx",
    channel_names=["F7-Fpz"],
    output="labels",
)

# Two-channel scoring with posterior fusion
labels = score_sleep_stages(
    eeg,
    sample_rate_hz=256.0,
    model_path="model/wu-sleep.onnx",
    channel_names=["F7-Fpz", "F8-Fpz"],
    output="labels",
)
```

## Examples

Minimal synthetic demo:

```bash
uv run python examples/score_synthetic.py
# or
python examples/score_synthetic.py
```

Full-night EEG-only example with LEFT, RIGHT, and fused scoring, plus expert ground truth:

```bash
uv run --extra examples python examples/score_example.py
# or, after pip install ".[examples]"
python examples/score_example.py
```

Data and outputs:

```text
examples/data/example.edf   # EEG_LEFT + EEG_RIGHT, ~10 h @ 128 Hz
examples/data/example.ids   # expert hypnogram
examples/output/example.png
```

## Citation

If you use WU-Sleep in your research, please cite:

> **WU-Sleep citation will be added when the preprint becomes available.**

## Model lineage

WU-Sleep builds on the U-Sleep architecture and was fine-tuned from the SLEEPYLAND `u-sleep-nsrr-2024_eeg` checkpoint.

When describing the model architecture or pretrained checkpoint, please also cite the corresponding upstream work:

- **U-Sleep:** Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*, 4, 72. [https://doi.org/10.1038/s41746-021-00440-5](https://doi.org/10.1038/s41746-021-00440-5)
- **SLEEPYLAND:** Rossi, A. D., Metaldi, M., Bechny, M., et al. (2026). SLEEPYLAND: trust begins with fair evaluation of automatic sleep staging models. *npj Digital Medicine*, 9, 55. [https://doi.org/10.1038/s41746-025-02237-2](https://doi.org/10.1038/s41746-025-02237-2)

## License

This repository is released under the [MIT License](LICENSE).

## TODO

- [ ] Validate WU-Sleep on other forehead EEG devices and comparable bipolar montages.
- [ ] Release the data-preparation, training, and evaluation code.
- [ ] Integrate artifact detection into the inference pipeline.