#!/usr/bin/env python3
"""Part 1 — IBS exploration.

Reads the CEPH pedigree identity matrix and produces:
  - A summary of the columns and the value distribution.
  - A histogram of identity values, separated by pair-of-individual identity:
    intra-individual (hap1 vs hap2 of the same person) vs inter-individual.

Pure stdlib — no numpy / matplotlib required. Output is SVG; opens in any
browser, image viewer or Inkscape.
"""

from collections import defaultdict
from pathlib import Path
from statistics import fmean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svgplot as svg

DATA = Path("data/identity_chr12_pedigree.tsv")
OUT_HIST = Path("figures/ibs_histogram.svg")


def load(path: Path):
    intra: list[float] = []
    inter: list[float] = []
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            a, b = f[col["group.a"]], f[col["group.b"]]
            ind_a, ind_b = a.split("#")[0], b.split("#")[0]
            try:
                ident = float(f[col["estimated.identity"]])
            except ValueError:
                continue
            if ind_a == ind_b:
                intra.append(ident)
            else:
                inter.append(ident)
    return intra, inter


def summarize(name: str, x: list[float]) -> None:
    if not x:
        print(f"  {name}: empty")
        return
    print(
        f"  {name:>20s}  n={len(x):>7d}  "
        f"mean={fmean(x):.4f}  median={svg.quantile(x, 0.5):.4f}  "
        f"p05={svg.quantile(x, 0.05):.4f}  p95={svg.quantile(x, 0.95):.4f}"
    )


def histcounts(xs: list[float], edges: list[float]) -> list[int]:
    """Right-open bins: counts[i] = #{x : edges[i] <= x < edges[i+1]},
    except the last bin is closed on both ends to include x == edges[-1]."""
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    lo = edges[0]
    hi = edges[-1]
    width = (hi - lo) / n_bins
    for x in xs:
        if x < lo or x > hi:
            continue
        idx = int((x - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1
    return counts


def main() -> int:
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Run from the workshop/ directory.", file=sys.stderr)
        return 1

    intra, inter = load(DATA)
    print(f"Loaded {DATA}: {len(intra) + len(inter)} pair-window rows")
    summarize("intra-individual", intra)
    summarize("inter-individual", inter)

    OUT_HIST.parent.mkdir(parents=True, exist_ok=True)

    # ---- bin the data -----------------------------------------------------
    n_bins = 79
    bin_lo, bin_hi = 0.985, 1.001
    edges = [bin_lo + i * (bin_hi - bin_lo) / n_bins for i in range(n_bins + 1)]
    counts_intra = histcounts(intra, edges)
    counts_inter = histcounts(inter, edges)
    y_max = max(max(counts_intra), max(counts_inter), 1) * 1.05

    # ---- layout (pixels) --------------------------------------------------
    W, H = 1050, 600
    L, R, T, B = 90, 30, 60, 70  # margins
    plot_x, plot_y = L, T
    plot_w, plot_h = W - L - R, H - T - B

    def x_to_px(x): return plot_x + (x - bin_lo) / (bin_hi - bin_lo) * plot_w
    def y_to_px(y): return plot_y + (1 - y / y_max) * plot_h

    out = [svg.header(W, H)]

    # title
    out.append(svg.text(W / 2, T - 22,
                        "CEPH 1463, chr12:40-130 Mb (long q arm) — identity distribution",
                        size=14, anchor="middle", va="baseline", color="#222"))

    # axes (bottom + left)
    x_ticks = [0.986, 0.988, 0.990, 0.992, 0.994, 0.996, 0.998, 1.000]
    y_ticks = svg.nice_ticks(0, y_max, target=7)
    out.append(svg.x_axis(plot_x, plot_y + plot_h, plot_w, bin_lo, bin_hi, x_ticks))
    out.append(svg.y_axis(plot_x, plot_y, plot_h, 0, y_max, y_ticks))

    # axis labels
    out.append(svg.text(plot_x + plot_w / 2, H - 18,
                        "estimated identity per 10 kb window",
                        size=12, anchor="middle", va="baseline"))
    out.append(
        f'<text x="{18}" y="{plot_y + plot_h / 2:.2f}" '
        f'font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 18 {plot_y + plot_h / 2:.2f})" '
        f'fill="#222">number of pair-windows</text>'
    )

    # bars — inter first (orange), then intra (gray) on top, both alpha 0.6,
    # so the visual matches the matplotlib output where overlap goes brown.
    def draw_bars(counts, color, label_n):
        for i, c in enumerate(counts):
            if c == 0:
                continue
            x0 = x_to_px(edges[i])
            x1 = x_to_px(edges[i + 1])
            y0 = y_to_px(c)
            y1 = y_to_px(0)
            out.append(svg.rect(x0, y0, x1 - x0, y1 - y0, color, opacity=0.6))

    draw_bars(counts_inter, "#cc6600", len(inter))
    draw_bars(counts_intra, "#444444", len(intra))

    # legend (top-left, inside plot area)
    leg_x, leg_y = plot_x + 14, plot_y + 18
    sw, sh, gap = 18, 12, 8
    out.append(svg.rect(leg_x, leg_y, sw, sh, "#444444", opacity=0.6))
    out.append(svg.text(leg_x + sw + 6, leg_y + sh / 2,
                        f"intra-individual (n={len(intra)})",
                        size=12, anchor="start", va="middle"))
    out.append(svg.rect(leg_x, leg_y + sh + gap, sw, sh, "#cc6600", opacity=0.6))
    out.append(svg.text(leg_x + sw + 6, leg_y + sh + gap + sh / 2,
                        f"inter-individual (n={len(inter)})",
                        size=12, anchor="start", va="middle"))

    out.append(svg.footer())
    OUT_HIST.write_text("".join(out))
    print(f"Wrote {OUT_HIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
