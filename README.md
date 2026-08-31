# WU-Sleep

Inference API for WU-Sleep, a domain-adapted sleep staging model for forehead wearable EEG.

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

## Usage

```python
from wu_sleep import score_sleep_stages

# Single channel
labels = score_sleep_stages(
    eeg[:, :1],
    sample_rate_hz=256.0,
    model_path="model/model.onnx",
    channel_names=["EEG_L"],
    output="labels",
)

# Multiple channels (order arbitrary)
labels = score_sleep_stages(
    eeg,
    sample_rate_hz=256.0,
    model_path="model/model.onnx",
    channel_names=["EEG_L", "EEG_R"],
    output="labels",
)
```

Download `model.onnx` from Hugging Face and place it in `model/` alongside `model.yaml`.

Recordings are scored in non-overlapping 30 s epochs at 128 Hz. If the signal length is not an integer number of 30 s epochs after preprocessing, the final partial epoch is edge-padded to 30 s and still scored.

## Example

```bash
uv run python examples/score_recording.py
```

## License

This repository is released under the [MIT License](LICENSE).

## Attribution

WU-Sleep builds on the U-Sleep architecture and was fine-tuned from U-Sleep weights distributed via SLEEPYLAND. If you use this software or model, please cite WU-Sleep and acknowledge the upstream work:

- **U-Sleep** — Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*, 4, 72. [https://doi.org/10.1038/s41746-021-00440-5](https://doi.org/10.1038/s41746-021-00440-5)
- **SLEEPYLAND** — Rossi, A. D., Metaldi, M., Bechny, M., Filchenko, I., van der Meer, J., Schmidt, M. H., Bassetti, C. L. A., Tzovara, A., Faraci, F. D., & Fiorillo, L. (2026). SLEEPYLAND: trust begins with fair evaluation of automatic sleep staging models. *npj Digital Medicine*, 9, 55. [https://doi.org/10.1038/s41746-025-02237-2](https://doi.org/10.1038/s41746-025-02237-2)
