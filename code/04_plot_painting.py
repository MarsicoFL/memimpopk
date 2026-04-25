#!/usr/bin/env python3
"""Part 3 — visualize the ancestry painting against ground truth.

Reads:
  solutions/ancestry_segments.tsv  (output of code/03_run_ancestry.sh)
  data/ground_truth_tracts.tsv

Writes:
  figures/student_ancestry_painting.png
"""

from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PRED = Path("solutions/ancestry_segments.tsv")
TRUTH = Path("data/ground_truth_tracts.tsv")
OUT = Path("figures/student_ancestry_painting.png")

COLORS = {"AFR": "#cc6600", "EUR": "#3366cc"}


def read_segments(path: Path, sample_col: str, anc_col: str):
    by_sample: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            sample = f[col[sample_col]]
            start = int(f[col["start"]])
            end = int(f[col["end"]])
            ancestry = f[col[anc_col]]
            by_sample[sample].append((start, end, ancestry))
    return by_sample


def main() -> int:
    if not PRED.exists() or not TRUTH.exists():
        print("ERROR: run bash code/03_run_ancestry.sh first.", file=sys.stderr)
        return 1

    pred = read_segments(PRED, "sample", "ancestry")
    truth = read_segments(TRUTH, "chim_id", "ancestry")

    samples = sorted(pred.keys())
    n = len(samples)
    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.7 * n))
    bar_h = 0.32

    all_coords = []
    for s in samples:
        for start, end, _ in truth.get(s, []) + pred.get(s, []):
            all_coords.extend([start, end])
    x_lo = min(all_coords) / 1e6
    x_hi = max(all_coords) / 1e6
    pad = (x_hi - x_lo) * 0.04
    label_x_left = x_lo - pad * 2.2
    label_x_right = x_hi + pad * 0.5

    for i, s in enumerate(samples):
        y_truth = i + 0.18
        y_pred = i - 0.18
        for start, end, anc in truth.get(s, []):
            ax.add_patch(Rectangle((start / 1e6, y_truth - bar_h / 2), (end - start) / 1e6, bar_h,
                                   facecolor=COLORS.get(anc, "#999999"), edgecolor="white", linewidth=0.5))
        for start, end, anc in pred.get(s, []):
            ax.add_patch(Rectangle((start / 1e6, y_pred - bar_h / 2), (end - start) / 1e6, bar_h,
                                   facecolor=COLORS.get(anc, "#999999"), edgecolor="white", linewidth=0.5))
        ax.text(label_x_left, y_truth, "truth", ha="right", va="center", fontsize=8, color="#333333")
        ax.text(label_x_left, y_pred, "impop", ha="right", va="center", fontsize=8, color="#333333")
        ax.text(label_x_right, i, s, ha="left", va="center", fontsize=9)

    ax.set_xlim(x_lo - pad * 3, x_hi + pad * 3)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("chr12 position (Mb)")
    ax.set_title(f"Local ancestry painting vs. ground truth — chr12:{int(x_lo)}-{int(x_hi)} Mb")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    legend = [Rectangle((0, 0), 1, 1, facecolor=COLORS["AFR"], label="AFR"),
              Rectangle((0, 0), 1, 1, facecolor=COLORS["EUR"], label="EUR")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, 1.05), ncol=2, frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
