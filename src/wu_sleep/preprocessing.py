from __future__ import annotations

import numpy as np

from somnio.data import TimeSeries
from somnio.transforms.clip import apply_clip_iqr
from somnio.transforms.filter import apply_filter
from somnio.transforms.resample import apply_resample
from somnio.transforms.scale import apply_scale


def preprocess_eeg(
    values: np.ndarray,
    sample_rate_hz: float,
    *,
    channel_names: list[str] | None = None,
) -> TimeSeries:
    """Preprocess raw EEG to match the WU-Sleep training pipeline.

    Applies resampling to 128 Hz, 0.3–35 Hz band-pass filtering, robust scaling,
    and IQR clipping (see the WU-Sleep preprint, Section 2.2).
    """
    x = np.asarray(values, dtype=np.float64)
    n_channels = x.shape[1]
    names = channel_names or [f"ch{i}" for i in range(n_channels)]

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
