#!/usr/bin/env python3
"""Part 1 — IBS exploration.

Reads the CEPH pedigree identity matrix and produces:
  - A summary of the columns and the value distribution.
  - A histogram of identity values, separated by pair-of-individual identity:
    intra-individual (hap1 vs hap2 of the same person) vs inter-individual.

No external deps beyond matplotlib + numpy (both shipped on the VM).
"""

from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("data/identity_chr12_pedigree.tsv")
OUT_HIST = Path("figures/ibs_histogram.png")


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
    return np.asarray(intra), np.asarray(inter)


def summarize(name: str, x: np.ndarray) -> None:
    if x.size == 0:
        print(f"  {name}: empty")
        return
    print(
        f"  {name:>20s}  n={x.size:>7d}  "
        f"mean={x.mean():.4f}  median={np.median(x):.4f}  "
        f"p05={np.quantile(x, 0.05):.4f}  p95={np.quantile(x, 0.95):.4f}"
    )


def main() -> int:
    if not DATA.exists():
        print(f"ERROR: {DATA} not found. Run from the workshop/ directory.", file=sys.stderr)
        return 1

    intra, inter = load(DATA)
    print(f"Loaded {DATA}: {intra.size + inter.size} pair-window rows")
    summarize("intra-individual", intra)
    summarize("inter-individual", inter)

    OUT_HIST.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.985, 1.001, 80)
    if intra.size:
        ax.hist(intra, bins=bins, alpha=0.6, label=f"intra-individual (n={intra.size})", color="#444444")
    if inter.size:
        ax.hist(inter, bins=bins, alpha=0.6, label=f"inter-individual (n={inter.size})", color="#cc6600")
    ax.set_xlabel("estimated identity per 10 kb window")
    ax.set_ylabel("number of pair-windows")
    ax.set_title("CEPH 1463, chr12:40-130 Mb (long q arm) — identity distribution")
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_HIST, dpi=150)
    print(f"Wrote {OUT_HIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
