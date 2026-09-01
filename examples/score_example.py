"""Score the bundled example recording and plot predictions vs ground truth.

Plots expert hypnogram together with LEFT, RIGHT, and fused hypnodensity /
hypnograms for the full-night EEG-only example recording.

Run with::

    uv run --extra examples python examples/score_example.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyedflib
from matplotlib.gridspec import GridSpec

from wu_sleep import labels_from_probabilities, score_sleep_stages

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
EDF_PATH = EXAMPLES_DIR / "data" / "example.edf"
IDS_PATH = EXAMPLES_DIR / "data" / "example.ids"
MODEL_PATH = REPO_ROOT / "model" / "wu-sleep.onnx"
OUTPUT_PATH = EXAMPLES_DIR / "output" / "example.png"

EEG_LEFT = "EEG_LEFT"
EEG_RIGHT = "EEG_RIGHT"
CLASS_LABELS = ["W", "N1", "N2", "N3", "REM"]
EPOCH_S = 30.0

STAGE_COLORS = {
    "W": "#E8B84A",
    "N1": "#8FBF6A",
    "N2": "#2F6F4E",
    "N3": "#1F4E79",
    "REM": "#7A3E9D",
    "UNKNOWN": "#A8AEB5",
    "ARTIFACT": "#1C1C1C",
}

HYPNO_ORDER = ["W", "REM", "N1", "N2", "N3", "UNKNOWN", "ARTIFACT"]


@dataclass
class Block:
    start: float
    duration: float
    label: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def load_eeg(edf_path: Path, channel_names: list[str]) -> tuple[np.ndarray, float]:
    reader = pyedflib.EdfReader(str(edf_path))
    labels = [reader.getLabel(i).strip() for i in range(reader.signals_in_file)]
    indices = [labels.index(name) for name in channel_names]
    sample_rate_hz = reader.getSampleFrequency(indices[0])
    values = np.column_stack([reader.readSignal(i) for i in indices])
    reader.close()
    return values, sample_rate_hz


def parse_ids(path: Path) -> list[Block]:
    blocks: list[Block] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start, duration, label = line.split(",")
            blocks.append(Block(float(start), float(duration), label.strip()))
    return blocks


def labels_to_blocks(labels: np.ndarray, epoch_sec: float) -> list[Block]:
    if labels.size == 0:
        return []

    blocks: list[Block] = []
    start = 0
    current = str(labels[0])
    for i in range(1, len(labels)):
        label = str(labels[i])
        if label != current:
            blocks.append(Block(start * epoch_sec, (i - start) * epoch_sec, current))
            start = i
            current = label
    blocks.append(Block(start * epoch_sec, (len(labels) - start) * epoch_sec, current))
    return blocks


def _style_panel_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)


def _row_label(fig: plt.Figure, cell, text: str) -> None:
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(
        1.0,
        0.5,
        text,
        ha="right",
        va="center",
        rotation=90,
        fontsize=8,
        transform=ax.transAxes,
    )


def draw_hypnodensity(ax: plt.Axes, probabilities: np.ndarray) -> None:
    colors = [STAGE_COLORS[label] for label in CLASS_LABELS]
    hours = np.arange(len(probabilities)) * EPOCH_S / 3600.0
    stack = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-8, None)
    ax.stackplot(hours, stack.T, colors=colors, alpha=0.92, lw=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=7)
    _style_panel_ax(ax)


def draw_hypnogram(ax: plt.Axes, blocks: Sequence[Block]) -> None:
    label_to_y = {lab: i for i, lab in enumerate(HYPNO_ORDER)}
    for block in blocks:
        lab = block.label if block.label in label_to_y else "UNKNOWN"
        y = label_to_y[lab]
        color = STAGE_COLORS.get(lab, STAGE_COLORS["UNKNOWN"])
        ax.broken_barh(
            [(block.start / 3600.0, block.duration / 3600.0)],
            (y - 0.4, 0.8),
            facecolors=color,
            edgecolors="none",
            alpha=0.95,
        )

    if blocks:
        xs: list[float] = []
        ys: list[float] = []
        for block in blocks:
            lab = block.label if block.label in label_to_y else "UNKNOWN"
            y = label_to_y[lab]
            xs.extend([block.start / 3600.0, block.end / 3600.0])
            ys.extend([y, y])
        ax.plot(xs, ys, color="#222222", lw=0.35, solid_capstyle="butt")

    ax.set_yticks(range(len(HYPNO_ORDER)))
    ax.set_yticklabels(HYPNO_ORDER, fontsize=7)
    ax.set_ylim(len(HYPNO_ORDER) - 0.5, -0.5)
    _style_panel_ax(ax)
    ax.tick_params(axis="y", length=0)


def stage_legend_handles() -> list[mpatches.Patch]:
    labels = ["W", "N1", "N2", "N3", "REM", "UNKNOWN"]
    return [
        mpatches.Patch(facecolor=STAGE_COLORS[lab], edgecolor="none", label=lab)
        for lab in labels
    ]


def main() -> None:
    if not EDF_PATH.is_file():
        raise FileNotFoundError(f"Missing recording: {EDF_PATH}")

    if not IDS_PATH.is_file():
        raise FileNotFoundError(f"Missing ground truth: {IDS_PATH}")

    eeg, sample_rate_hz = load_eeg(EDF_PATH, [EEG_LEFT, EEG_RIGHT])
    ground_truth = parse_ids(IDS_PATH)
    print(
        f"Loaded {eeg.shape[1]} channels, "
        f"{eeg.shape[0] / sample_rate_hz / 3600:.2f} h at {sample_rate_hz} Hz"
    )

    left_probs = score_sleep_stages(
        eeg[:, :1],
        sample_rate_hz=sample_rate_hz,
        model_path=MODEL_PATH,
        channel_names=[EEG_LEFT],
        output="probs",
    )
    right_probs = score_sleep_stages(
        eeg[:, 1:2],
        sample_rate_hz=sample_rate_hz,
        model_path=MODEL_PATH,
        channel_names=[EEG_RIGHT],
        output="probs",
    )
    fused_probs = score_sleep_stages(
        eeg,
        sample_rate_hz=sample_rate_hz,
        model_path=MODEL_PATH,
        channel_names=[EEG_LEFT, EEG_RIGHT],
        output="probs",
    )

    pred_panels = [
        ("LEFT", left_probs, labels_from_probabilities(left_probs, CLASS_LABELS)),
        ("RIGHT", right_probs, labels_from_probabilities(right_probs, CLASS_LABELS)),
        ("Fused", fused_probs, labels_from_probabilities(fused_probs, CLASS_LABELS)),
    ]

    duration_h = left_probs.shape[0] * EPOCH_S / 3600.0
    print(f"Scored {left_probs.shape[0]} epochs ({duration_h:.2f} h)")

    # legend, ground truth, then dens+pred for each source
    height_ratios = [0.35, 1.3]
    for _ in pred_panels:
        height_ratios.extend([0.9, 1.3])

    fig = plt.figure(figsize=(11.5, 0.85 * sum(height_ratios) + 1.0), constrained_layout=False)
    gs = GridSpec(
        len(height_ratios),
        2,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[0.28, 10],
        hspace=0.08,
        wspace=0.12,
        left=0.025,
        right=0.98,
        top=0.96,
        bottom=0.04,
    )

    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=stage_legend_handles(),
        loc="center",
        ncol=6,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.2,
    )
    ax_leg.set_title("example", fontsize=11, pad=2)

    axes: list[plt.Axes] = []
    _row_label(fig, gs[1, 0], "Ground truth")
    ax_gt = fig.add_subplot(gs[1, 1])
    draw_hypnogram(ax_gt, ground_truth)
    axes.append(ax_gt)

    for i, (source, probs, labels) in enumerate(pred_panels):
        dens_row = 2 + 2 * i
        hyp_row = 3 + 2 * i
        _row_label(fig, gs[dens_row, 0], f"{source}\nProbabilities")
        _row_label(fig, gs[hyp_row, 0], f"{source}\nLabels")
        dens_ax = fig.add_subplot(gs[dens_row, 1], sharex=axes[0])
        hyp_ax = fig.add_subplot(gs[hyp_row, 1], sharex=axes[0])
        draw_hypnodensity(dens_ax, probs)
        draw_hypnogram(hyp_ax, labels_to_blocks(labels, EPOCH_S))
        axes.extend([dens_ax, hyp_ax])

    for ax in axes:
        ax.set_xlim(0, duration_h)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        if ax is not axes[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (hours)", fontsize=9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
