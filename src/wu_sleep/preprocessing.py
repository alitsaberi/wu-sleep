from __future__ import annotations

import math

import numpy as np

from somnio.data import TimeSeries
from somnio.transforms.clip import apply_clip_iqr
from somnio.transforms.filter import apply_filter
from somnio.transforms.resample import apply_resample
from somnio.transforms.scale import apply_scale


def _validate_eeg_input(
    values: np.ndarray,
    sample_rate_hz: float,
    channel_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"values must have shape (n_samples, n_channels), got {x.shape}")
    if x.shape[0] < 1:
        raise ValueError(f"values must include at least one sample, got {x.shape}")
    if x.shape[1] < 1:
        raise ValueError(f"values must include at least one channel, got {x.shape}")

    rate = float(sample_rate_hz)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}")

    n_channels = x.shape[1]
    names = channel_names or [f"ch{i}" for i in range(n_channels)]
    if len(names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(names)}) must match n_channels ({n_channels})"
        )
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique")

    return x, names


def preprocess_eeg(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    channel_names: list[str] | None = None,
) -> TimeSeries:
    """Preprocess raw EEG to match the WU-Sleep training pipeline.

    Applies resampling to 128 Hz, 0.3–35 Hz band-pass filtering, robust scaling,
    and IQR clipping.
    """
    x, names = _validate_eeg_input(values, sample_rate_hz, channel_names)
    n_channels = x.shape[1]

    n_samples = x.shape[0]
    step_ns = int(round(1e9 / float(sample_rate_hz)))
    timestamps = np.arange(n_samples, dtype=np.int64) * step_ns

    ts = TimeSeries(
        values=x,
        timestamps=timestamps,
        channel_names=names,
        units=["UNKNOWN"] * n_channels,
        sample_rate=float(sample_rate_hz),
    )
    ts = apply_resample(ts, 128.0)
    ts = apply_filter(ts, low_cutoff=0.3, high_cutoff=35.0)
    ts = apply_scale(ts, method="robust")
    return apply_clip_iqr(ts, iqr_factor=20.0)
