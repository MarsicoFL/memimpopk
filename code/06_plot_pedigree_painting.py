#!/usr/bin/env python3
"""Part 4 — visualize the grandparent painting of the five grandchildren.

Reads:
  solutions/pedigree_painting.tsv  (output of code/05_paint_pedigree.sh)

Writes:
  figures/student_pedigree_painting.png

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
"""

from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PRED = Path("solutions/pedigree_painting.tsv")
OUT = Path("figures/student_pedigree_painting.png")

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
    fig, ax = plt.subplots(figsize=(10, 0.55 * n_rows + 1))
    bar_h = 0.55

    for i, (grand, hap) in enumerate(rows):
        sample = f"{grand}{hap}"
        y = n_rows - 1 - i
        ax.add_patch(Rectangle((40, y - bar_h / 2), 90, bar_h,
                               facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.3))
        for start, end, anc in pred.get(sample, []):
            ax.add_patch(Rectangle((start / 1e6, y - bar_h / 2),
                                   (end - start) / 1e6, bar_h,
                                   facecolor=COLORS.get(anc, "#999999"),
                                   edgecolor="white", linewidth=0.3))
        ax.text(39.5, y, sample, ha="right", va="center", fontsize=9)

    ax.set_xlim(38, 131)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("chr12 position (Mb)")
    ax.set_title("Grandparent painting of CEPH 1463 grandchildren — chr12:40-130 Mb")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    legend = [Rectangle((0, 0), 1, 1, facecolor=COLORS[p], label=p)
              for p in ("GP_PAT", "GM_PAT", "GP_MAT", "GM_MAT")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, frameon=False, fontsize=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
