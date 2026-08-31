from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel
from somnio.tasks.sleep_scoring.score import score_sleep_stages as somnio_score_sleep_stages

from wu_sleep.preprocessing import preprocess_eeg

OutputMode = Literal["probs", "labels"]


def fuse_probabilities(*probabilities: np.ndarray) -> np.ndarray:
    """Fuse posterior probabilities from independently scored channels.

    Channel posteriors are summed per epoch, then renormalized to sum to 1.
    Fusion is order-invariant and matches the WU-Sleep preprint (Section 2.8).

    Each input array must have shape ``(n_epochs, n_classes)``.
    """
    if len(probabilities) < 1:
        raise ValueError("at least one probability array is required")

    arrays = [np.asarray(p, dtype=np.float64) for p in probabilities]
    reference = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != reference:
            raise ValueError(
                f"probability arrays must have the same shape, got {reference} and "
                f"{arr.shape} at index {i}"
            )

    fused = sum(arrays)
    if len(arrays) == 1:
        return fused

    row_sums = fused.sum(axis=1, keepdims=True)
    return fused / row_sums


def _score_preprocessed(
    ts: TimeSeries,
    *,
    model: OnnxSleepScoringModel,
    output: OutputMode = "probs",
) -> np.ndarray:
    if ts.n_channels != 1:
        raise ValueError(
            f"expected a single-channel TimeSeries, got n_channels={ts.n_channels}"
        )

    if output == "probs":
        probs_ts = somnio_score_sleep_stages(
            ts,
            backend=model,
            metadata=model.metadata,
            output="probs_timeseries",
        )
        assert isinstance(probs_ts, TimeSeries)
        return np.asarray(probs_ts.values, dtype=np.float64)

    epochs = somnio_score_sleep_stages(
        ts,
        backend=model,
        metadata=model.metadata,
        output="labels_epochs",
    )
    return np.asarray(epochs.labels, dtype=object)


def _labels_from_probabilities(
    probabilities: np.ndarray,
    class_labels: list[str],
) -> np.ndarray:
    indices = np.argmax(probabilities, axis=1).astype(np.int64)
    return np.asarray([class_labels[int(i)] for i in indices], dtype=object)


def score_sleep_stages(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    model_path: str | Path = Path("model/model.onnx"),
    output: OutputMode = "probs",
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """Run sleep-stage inference on forehead wearable EEG.

    Pass raw EEG as ``(n_samples, n_channels)``. The underlying model is
    single-channel. Each column is scored independently; channel posteriors are
    summed and renormalized per 30 s epoch (column order does not matter).

    All channels are preprocessed together once (resample, filter, scale, clip),
    then each column is scored separately.

    WU-Sleep targets **forehead wearable EEG** with bipolar derivations similar
    to those used in training (e.g. F7–Fpz, F8–Fpz). See the README for the
    reference montage.

    Recordings are scored in non-overlapping 30 s epochs. If the length is not
    an integer number of 30 s epochs after preprocessing, the final partial
    epoch is edge-padded to 30 s and still scored.

    Args:
        values: EEG array, shape ``(n_samples, n_channels)``.
        sample_rate_hz: Nominal sample rate of ``values`` in Hz.
        model_path: ONNX model path. Sidecar metadata is discovered next to it.
        output: ``"probs"`` or ``"labels"``.
        channel_names: Optional channel names, one per column (e.g. ``["EEG_L", "EEG_R"]``).
            Defaults to ``["ch0", ...]`` when omitted.

    Returns:
        - If ``output="probs"``: float64 array of shape ``(n_epochs, n_classes)``.
          Rows sum to 1. With one channel, these are the model posteriors; with
          multiple channels, fused and renormalized posteriors.
        - If ``output="labels"``: object array of shape ``(n_epochs,)``.
    """
    model = OnnxSleepScoringModel.load(model_path)
    ts = preprocess_eeg(values, sample_rate_hz, channel_names=channel_names)

    channel_probs = [
        _score_preprocessed(ts.select_channels([name]), model=model, output="probs")
        for name in ts.channel_names
    ]
    fused = fuse_probabilities(*channel_probs)

    if output == "probs":
        return fused
    return _labels_from_probabilities(fused, model.metadata.class_labels)
