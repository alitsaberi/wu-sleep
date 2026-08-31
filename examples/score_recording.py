"""Minimal example: score synthetic single-channel EEG."""

from pathlib import Path

import numpy as np

from wu_sleep import run_sleep_scoring

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.onnx"
SAMPLE_RATE_HZ = 256.0
DURATION_S = 60.0


def main() -> None:
    n_samples = int(DURATION_S * SAMPLE_RATE_HZ)
    rng = np.random.default_rng(0)
    eeg = rng.normal(size=(n_samples, 1))

    labels = run_sleep_scoring(
        eeg,
        sample_rate_hz=SAMPLE_RATE_HZ,
        model_path=MODEL_PATH,
        output="labels",
    )
    print(f"Scored {labels.size} epochs")
    print(labels[:10])


if __name__ == "__main__":
    main()
