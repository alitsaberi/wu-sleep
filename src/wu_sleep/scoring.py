from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel
from somnio.tasks.sleep_scoring.score import score_sleep_stages
from somnio.transforms.clip import apply_clip_iqr
from somnio.transforms.filter import apply_fir_filter
from somnio.transforms.resample import apply_resample
from somnio.transforms.scale import apply_scale


OutputMode = Literal["probs", "labels"]


def run_sleep_scoring(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    model_path: str | Path = Path("model/model.onnx"),
    output: OutputMode = "probs",
) -> np.ndarray:
    """Run sleep-stage inference on a raw (n_samples, n_channels) array.

    Recordings are scored in non-overlapping 30 s epochs. If the length is not
    an integer number of 30 s epochs after preprocessing, the final partial
    epoch is edge-padded to 30 s and still scored.

    Args:
        values: Signal array, shape ``(n_samples, n_channels)``.
        sample_rate_hz: Nominal sample rate of ``values`` in Hz.
        model_path: ONNX model path. Sidecar metadata is discovered next to it.
        output: ``"probs"`` or ``"labels"``.

    Returns:
        - If ``output="probs"``: float64 array of shape ``(n_epochs, n_classes)``.
        - If ``output="labels"``: object array of shape ``(n_epochs,)``.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"values must have shape (n_samples, n_channels), got {x.shape}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")

    n_samples = x.shape[0]
    step_ns = int(round(1e9 / float(sample_rate_hz)))
    timestamps = np.arange(n_samples, dtype=np.int64) * step_ns
    channel_names = [f"ch{i}" for i in range(x.shape[1])]

    ts = TimeSeries(
        values=x,
        timestamps=timestamps,
        channel_names=channel_names,
        units=["V"] * x.shape[1],
        sample_rate=float(sample_rate_hz),
    )

    ts = apply_resample(ts, 128.0)
    ts = apply_fir_filter(ts, low_cutoff=0.3, high_cutoff=35.0)
    ts = apply_scale(ts, method="robust")
    ts = apply_clip_iqr(ts, iqr_factor=20.0)

    model = OnnxSleepScoringModel.load(model_path)

    if output == "probs":
        probs_ts = score_sleep_stages(
            ts,
            backend=model,
            metadata=model.metadata,
            output="probs_timeseries",
        )
        assert isinstance(probs_ts, TimeSeries)
        return np.asarray(probs_ts.values, dtype=np.float64)

    epochs = score_sleep_stages(
        ts,
        backend=model,
        metadata=model.metadata,
        output="labels_epochs",
    )
    return np.asarray(epochs.labels, dtype=object)
