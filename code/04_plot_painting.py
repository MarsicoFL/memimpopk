#!/usr/bin/env python3
"""Part 3 — visualize the ancestry painting against ground truth.

Reads:
  solutions/ancestry_segments.tsv  (output of code/03_run_ancestry.sh)
  data/ground_truth_tracts.tsv

Writes:
  figures/ancestry_painting.svg

Pure stdlib — no numpy / matplotlib required.
"""

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svgplot as svg

PRED = Path("solutions/ancestry_segments.tsv")
TRUTH = Path("data/ground_truth_tracts.tsv")
OUT = Path("figures/ancestry_painting.svg")

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

    # ---- determine genomic span across all samples ----------------------
    all_coords: list[int] = []
    for s in samples:
        for start, end, _ in truth.get(s, []) + pred.get(s, []):
            all_coords.extend([start, end])
    x_lo_mb = min(all_coords) / 1e6
    x_hi_mb = max(all_coords) / 1e6
    pad = (x_hi_mb - x_lo_mb) * 0.04

    # ---- pixel layout ---------------------------------------------------
    W = 1400
    row_h = 70
    H = 80 + row_h * n + 80  # title + rows + axis
    L_margin = 110
    R_margin = 110
    T_margin = 80
    plot_x = L_margin
    plot_w = W - L_margin - R_margin
    plot_y_top = T_margin
    plot_y_bot = H - 60  # leave 60 px for the bottom axis labels
    rows_h = plot_y_bot - plot_y_top

    # x-axis maps Mb [x_lo_mb - pad*3, x_hi_mb + pad*3] to [plot_x, plot_x + plot_w]
    ax_lo = x_lo_mb - pad * 3
    ax_hi = x_hi_mb + pad * 3

    def x_to_px(mb): return plot_x + (mb - ax_lo) / (ax_hi - ax_lo) * plot_w

    out = [svg.header(W, H)]

    # title
    out.append(svg.text(
        W / 2, T_margin - 35,
        f"Local ancestry painting vs. ground truth — chr12:{int(x_lo_mb)}-{int(x_hi_mb)} Mb",
        size=15, anchor="middle", va="baseline", color="#222"))

    # legend (top-left)
    leg_x, leg_y = 30, T_margin - 50
    sw, sh = 22, 14
    out.append(svg.rect(leg_x,            leg_y, sw, sh, COLORS["AFR"]))
    out.append(svg.text(leg_x + sw + 5,   leg_y + sh / 2, "AFR",
                        size=12, anchor="start", va="middle"))
    out.append(svg.rect(leg_x + 90,       leg_y, sw, sh, COLORS["EUR"]))
    out.append(svg.text(leg_x + 90 + sw + 5, leg_y + sh / 2, "EUR",
                        size=12, anchor="start", va="middle"))

    # rows
    bar_h = 17
    label_left_px = x_to_px(x_lo_mb) - 10  # right-aligned at this x
    label_right_px = x_to_px(x_hi_mb) + 10  # left-aligned at this x

    for i, s in enumerate(samples):
        # match matplotlib's bottom-up ordering: highest sample index at the
        # top of the figure, sample 0 at the bottom.
        row_center = plot_y_top + rows_h * ((n - 1 - i) + 0.5) / n
        y_truth = row_center - 14
        y_pred = row_center + 14

        for start, end, anc in truth.get(s, []):
            x0 = x_to_px(start / 1e6)
            x1 = x_to_px(end / 1e6)
            out.append(svg.rect(x0, y_truth - bar_h / 2, x1 - x0, bar_h,
                                COLORS.get(anc, "#999999"),
                                stroke="white", stroke_width=0.5))
        for start, end, anc in pred.get(s, []):
            x0 = x_to_px(start / 1e6)
            x1 = x_to_px(end / 1e6)
            out.append(svg.rect(x0, y_pred - bar_h / 2, x1 - x0, bar_h,
                                COLORS.get(anc, "#999999"),
                                stroke="white", stroke_width=0.5))

        out.append(svg.text(label_left_px, y_truth, "truth",
                            size=11, anchor="end", va="middle", color="#333"))
        out.append(svg.text(label_left_px, y_pred, "impop",
                            size=11, anchor="end", va="middle", color="#333"))
        out.append(svg.text(label_right_px, row_center, s,
                            size=12, anchor="start", va="middle", color="#222"))

    # x-axis along the bottom
    axis_y = plot_y_bot + 6
    x_ticks = svg.nice_ticks(int(x_lo_mb), int(x_hi_mb) + 1, target=8)
    out.append(svg.x_axis(plot_x, axis_y, plot_w, ax_lo, ax_hi, x_ticks,
                          tick_len=5, label_size=11))
    out.append(svg.text(plot_x + plot_w / 2, axis_y + 35,
                        "chr12 position (Mb)",
                        size=13, anchor="middle", va="baseline"))

    out.append(svg.footer())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(out))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
