"""Minimal example: score synthetic forehead EEG (single- and dual-channel)."""

from pathlib import Path

import numpy as np

from wu_sleep import score_sleep_stages

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.onnx"
SAMPLE_RATE_HZ = 256.0
DURATION_S = 60.0


def main() -> None:
    n_samples = int(DURATION_S * SAMPLE_RATE_HZ)
    rng = np.random.default_rng(0)

    eeg_one = rng.normal(size=(n_samples, 1))
    labels = score_sleep_stages(
        eeg_one,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_path=MODEL_PATH,
        channel_names=["EEG_L"],
        output="labels",
    )
    print(f"Single channel: {labels.size} epochs")
    print(labels[:10])

    eeg_two = rng.normal(size=(n_samples, 2))
    fused_labels = score_sleep_stages(
        eeg_two,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_path=MODEL_PATH,
        channel_names=["EEG_L", "EEG_R"],
        output="labels",
    )
    print(f"Two channels (fused): {fused_labels.size} epochs")
    print(fused_labels[:10])


if __name__ == "__main__":
    main()
