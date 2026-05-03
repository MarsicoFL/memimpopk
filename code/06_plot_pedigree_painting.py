#!/usr/bin/env python3
"""Part 4 — visualize the grandparent painting of the five grandchildren.

Reads:
  solutions/pedigree_painting.tsv  (output of code/05_paint_pedigree.sh)

Writes:
  figures/pedigree_painting.svg

For each grandchild (NA12879, NA12881, NA12882, NA12885, NA12886) we paint
both haplotypes (#1 and #2) across chr12:40-130 Mb. Colors:

    GP_PAT (NA12889)  paternal grandfather  orange
    GM_PAT (NA12890)  paternal grandmother  red
    GP_MAT (NA12891)  maternal grandfather  blue
    GM_MAT (NA12892)  maternal grandmother  teal

The biological expectation: one of the two haplotypes of each grandchild
should be a mosaic of GP_PAT + GM_PAT only (paternal homolog) and the
other a mosaic of GP_MAT + GM_MAT only (maternal homolog). The painting
tells us, without any phasing metadata, which of #1 and #2 is paternal.

Pure stdlib — no numpy / matplotlib required.
"""

from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svgplot as svg

PRED = Path("solutions/pedigree_painting.tsv")
OUT = Path("figures/pedigree_painting.svg")

COLORS = {
    "GP_PAT": "#cc6600",
    "GM_PAT": "#b8553a",
    "GP_MAT": "#3366cc",
    "GM_MAT": "#4a9999",
}

GRANDCHILDREN = ["NA12879", "NA12881", "NA12882", "NA12885", "NA12886"]


def read_segments(path: Path):
    by_sample: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            sample = f[col["sample"]]
            start = int(f[col["start"]])
            end = int(f[col["end"]])
            ancestry = f[col["ancestry"]]
            by_sample[sample].append((start, end, ancestry))
    return by_sample


def main() -> int:
    if not PRED.exists():
        print("ERROR: run bash code/05_paint_pedigree.sh first.", file=sys.stderr)
        return 1

    pred = read_segments(PRED)

    rows = [(g, h) for g in GRANDCHILDREN for h in ("#1", "#2")]
    n_rows = len(rows)

    # ---- pixel layout ----
    W = 1450
    row_h = 38
    L_margin, R_margin = 110, 30
    T_margin, B_margin = 100, 75
    H = T_margin + n_rows * row_h + B_margin
    plot_x = L_margin
    plot_w = W - L_margin - R_margin
    plot_y_top = T_margin
    plot_y_bot = T_margin + n_rows * row_h

    # x-axis maps Mb [38, 131] (matches matplotlib's xlim)
    ax_lo, ax_hi = 38.0, 131.0

    def x_to_px(mb): return plot_x + (mb - ax_lo) / (ax_hi - ax_lo) * plot_w

    out = [svg.header(W, H)]

    # legend (centered, above the title — matches matplotlib bbox_to_anchor=(0.5, 1.12))
    leg_items = [(p, COLORS[p]) for p in ("GP_PAT", "GM_PAT", "GP_MAT", "GM_MAT")]
    sw, sh = 22, 14
    item_w = 110
    leg_total = item_w * len(leg_items)
    leg_x0 = (W - leg_total) / 2
    leg_y = 25
    for k, (lab, color) in enumerate(leg_items):
        x = leg_x0 + k * item_w
        out.append(svg.rect(x, leg_y, sw, sh, color))
        out.append(svg.text(x + sw + 6, leg_y + sh / 2, lab,
                            size=12, anchor="start", va="middle"))

    # title
    out.append(svg.text(W / 2, leg_y + sh + 28,
                        "Grandparent painting of CEPH 1463 grandchildren — chr12:40-130 Mb",
                        size=14, anchor="middle", va="baseline"))

    # rows
    bar_h = row_h * 0.55
    for i, (grand, hap) in enumerate(rows):
        sample = f"{grand}{hap}"
        # match matplotlib: row 0 (NA12879#1) at top, row n-1 (NA12886#2) at bottom
        y_center = plot_y_top + (i + 0.5) * row_h

        # background bar from 40 to 130 Mb, light gray
        x_bg0 = x_to_px(40)
        x_bg1 = x_to_px(130)
        out.append(svg.rect(x_bg0, y_center - bar_h / 2, x_bg1 - x_bg0, bar_h,
                            "#f0f0f0", stroke="#cccccc", stroke_width=0.3))

        for start, end, anc in pred.get(sample, []):
            x0 = x_to_px(start / 1e6)
            x1 = x_to_px(end / 1e6)
            out.append(svg.rect(x0, y_center - bar_h / 2, x1 - x0, bar_h,
                                COLORS.get(anc, "#999999"),
                                stroke="white", stroke_width=0.3))

        # left-side sample label, right-aligned at x just left of 40 Mb
        out.append(svg.text(x_to_px(39.5), y_center, sample,
                            size=12, anchor="end", va="middle", color="#222"))

    # x-axis at the bottom of the rows
    axis_y = plot_y_bot + 8
    x_ticks = [40, 60, 80, 100, 120]
    out.append(svg.x_axis(plot_x, axis_y, plot_w, ax_lo, ax_hi, x_ticks,
                          tick_len=5, label_size=11))
    out.append(svg.text(plot_x + plot_w / 2, axis_y + 38,
                        "chr12 position (Mb)",
                        size=13, anchor="middle", va="baseline"))

    out.append(svg.footer())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(out))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
