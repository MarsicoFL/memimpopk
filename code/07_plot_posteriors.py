#!/usr/bin/env python3
"""Part 3 deeper dive — posterior trace and panel-similarity heatmap.

Reads:
  solutions/ancestry_posteriors.tsv   (per-window forward-backward posteriors)
  solutions/ancestry_segments.tsv     (Viterbi segments)
  data/identity_chr12_admix.tsv       (raw per-window identity)
  data/populations.tsv                (panel labels)
  data/ground_truth_tracts.tsv        (truth)

Writes:
  figures/student_posterior_trace.png
  figures/student_panel_similarity.png

Plot A — posterior trace for CHIM_03:
  top   : ground-truth ancestry bar
  middle: P(AFR) per window from forward-backward (the "marginal posterior")
  bottom: Viterbi-decoded segments (the most likely state path)

Plot B — panel-similarity for CHIM_03:
  for each window, the mean estimated identity to the AFR panel haps and to
  the EUR panel haps. This is the input to the HMM emissions; the gap
  between the two lines is the per-window contrast Δ that the HMM has to
  amplify into a confident call.
"""

from collections import defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

POST = Path("solutions/ancestry_posteriors.tsv")
SEGS = Path("solutions/ancestry_segments.tsv")
SIM  = Path("data/identity_chr12_admix.tsv")
POPS = Path("data/populations.tsv")
TRUTH = Path("data/ground_truth_tracts.tsv")

OUT_TRACE = Path("figures/posterior_trace.png")
OUT_PANEL = Path("figures/panel_similarity.png")

QUERY = "CHIM_03#1"
COLORS = {"AFR": "#cc6600", "EUR": "#3366cc"}


def load_posteriors(query):
    rows = []
    with POST.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[col["sample"]] != query:
                continue
            rows.append((int(f[col["start"]]), int(f[col["end"]]),
                         float(f[col["P(AFR)"]]), float(f[col["P(EUR)"]])))
    rows.sort()
    return rows


def load_segments(query):
    out = []
    with SEGS.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[col["sample"]] != query:
                continue
            out.append((int(f[col["start"]]), int(f[col["end"]]),
                        f[col["ancestry"]]))
    out.sort()
    return out


def load_truth(query):
    out = []
    with TRUTH.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[col["chim_id"]] != query:
                continue
            out.append((int(f[col["start"]]), int(f[col["end"]]),
                        f[col["ancestry"]]))
    out.sort()
    return out


def load_panel_membership():
    by_pop = {"AFR": set(), "EUR": set()}
    with POPS.open() as fh:
        for line in fh:
            pop, hap = line.rstrip("\n").split("\t")
            by_pop[pop].add(hap)
    return by_pop


def panel_similarity_per_window(query):
    """For each window, return the *max* estimated.identity to any AFR
    panel hap and to any EUR panel hap. Max-aggregation is what the HMM
    actually uses by default for emissions, so this is the real input
    contrast that the model is trying to amplify."""
    panel = load_panel_membership()
    afr, eur = panel["AFR"], panel["EUR"]
    best = defaultdict(lambda: [-1.0, -1.0])  # max_afr, max_eur
    with SIM.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[col["group.a"]] != query and f[col["group.b"]] != query:
                continue
            other = f[col["group.b"]] if f[col["group.a"]] == query else f[col["group.a"]]
            try:
                ident = float(f[col["estimated.identity"]])
            except ValueError:
                continue
            key = (int(f[col["start"]]), int(f[col["end"]]))
            if other in afr and ident > best[key][0]:
                best[key][0] = ident
            elif other in eur and ident > best[key][1]:
                best[key][1] = ident
    rows = []
    for (s, e), (m_afr, m_eur) in best.items():
        rows.append((s, e,
                     m_afr if m_afr >= 0 else float("nan"),
                     m_eur if m_eur >= 0 else float("nan")))
    rows.sort()
    return rows


def draw_truth_bar(ax, segs, y, h):
    for s, e, anc in segs:
        ax.add_patch(Rectangle((s / 1e6, y - h / 2),
                               (e - s) / 1e6, h,
                               facecolor=COLORS.get(anc, "#999999"),
                               edgecolor="white", linewidth=0.4))


def main() -> int:
    if not POST.exists() or not SEGS.exists():
        print("ERROR: run bash code/03_run_ancestry.sh first.", file=sys.stderr)
        return 1

    OUT_TRACE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------ Plot A: posterior trace ---------------------
    post = load_posteriors(QUERY)
    segs = load_segments(QUERY)
    truth = load_truth(QUERY)

    starts = np.array([(s + e) / 2 / 1e6 for s, e, *_ in post])
    p_afr = np.array([row[2] for row in post])

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.25, 1.6, 0.25], hspace=0.18)

    ax_truth = fig.add_subplot(gs[0])
    draw_truth_bar(ax_truth, truth, 0, 0.7)
    ax_truth.set_xlim(50, 120)
    ax_truth.set_ylim(-0.5, 0.5); ax_truth.set_yticks([])
    ax_truth.tick_params(bottom=False, labelbottom=False)
    ax_truth.set_title(f"{QUERY} — ground truth (top), forward-backward P(AFR) per window (middle), Viterbi segments (bottom)",
                       fontsize=11)
    for sp in ("top", "right", "left", "bottom"):
        ax_truth.spines[sp].set_visible(False)

    ax_post = fig.add_subplot(gs[1], sharex=ax_truth)
    ax_post.fill_between(starts, p_afr, 0, color=COLORS["AFR"], alpha=0.35, lw=0)
    ax_post.fill_between(starts, p_afr, 1, color=COLORS["EUR"], alpha=0.35, lw=0)
    ax_post.plot(starts, p_afr, color="#222222", lw=0.6)
    ax_post.axhline(0.5, color="#888888", lw=0.6, ls="--")
    ax_post.set_ylim(-0.02, 1.02)
    ax_post.set_ylabel("P(AFR | data)")
    ax_post.tick_params(bottom=False, labelbottom=False)
    for sp in ("top", "right"):
        ax_post.spines[sp].set_visible(False)

    ax_vit = fig.add_subplot(gs[2], sharex=ax_post)
    draw_truth_bar(ax_vit, segs, 0, 0.7)
    ax_vit.set_ylim(-0.5, 0.5); ax_vit.set_yticks([])
    ax_vit.set_xlabel("chr12 position (Mb)")
    ax_vit.tick_params(bottom=True, labelbottom=True)
    for sp in ("top", "right", "left"):
        ax_vit.spines[sp].set_visible(False)

    ax_truth.text(-0.05, 0, "truth",  ha="right", va="center", fontsize=9, transform=ax_truth.get_yaxis_transform())
    ax_vit  .text(-0.05, 0, "Viterbi", ha="right", va="center", fontsize=9, transform=ax_vit.get_yaxis_transform())

    fig.tight_layout()
    fig.savefig(OUT_TRACE, dpi=150)
    print(f"Wrote {OUT_TRACE}")

    # ------------------------ Plot B: panel similarity --------------------
    # Show the Δ contrast (AFR_sim - EUR_sim) per window. The two raw
    # similarity series are visually indistinguishable because Δ ≈ 1e-3,
    # so we plot the difference directly and smooth it with a rolling mean.
    sim_rows = panel_similarity_per_window(QUERY)
    pos = np.array([(s + e) / 2 / 1e6 for s, e, *_ in sim_rows])
    sim_afr = np.array([row[2] for row in sim_rows])
    sim_eur = np.array([row[3] for row in sim_rows])
    delta = sim_afr - sim_eur

    win = 25  # ~250 kb rolling window
    kernel = np.ones(win) / win
    delta_smooth = np.convolve(delta, kernel, mode="same")
    med_abs_delta = np.nanmedian(np.abs(delta))

    fig2 = plt.figure(figsize=(11, 5.0))
    gs2 = fig2.add_gridspec(2, 1, height_ratios=[0.25, 1.6], hspace=0.18)

    ax_t = fig2.add_subplot(gs2[0])
    draw_truth_bar(ax_t, truth, 0, 0.7)
    ax_t.set_xlim(50, 120)
    ax_t.set_ylim(-0.5, 0.5); ax_t.set_yticks([])
    ax_t.tick_params(bottom=False, labelbottom=False)
    ax_t.set_title(f"{QUERY} — raw input to the HMM emissions\n"
                   f"Δ = max-AFR − max-EUR per window, 250-kb rolling mean.  "
                   f"median |Δ| = {med_abs_delta:.4f} per single window.",
                   fontsize=10)
    for sp in ("top", "right", "left", "bottom"):
        ax_t.spines[sp].set_visible(False)

    ax_d = fig2.add_subplot(gs2[1], sharex=ax_t)
    pos_above = delta_smooth > 0
    ax_d.fill_between(pos, 0, delta_smooth, where=pos_above,
                      color=COLORS["AFR"], alpha=0.55, lw=0,
                      label="more AFR-like (Δ > 0)")
    ax_d.fill_between(pos, 0, delta_smooth, where=~pos_above,
                      color=COLORS["EUR"], alpha=0.55, lw=0,
                      label="more EUR-like (Δ < 0)")
    ax_d.plot(pos, delta_smooth, color="#222222", lw=0.6)
    ax_d.axhline(0, color="#888888", lw=0.6)
    ax_d.set_ylabel("Δ identity = max(AFR) − max(EUR)")
    ax_d.set_xlabel("chr12 position (Mb)")
    ax_d.legend(loc="lower right", frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax_d.spines[sp].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(OUT_PANEL, dpi=150)
    print(f"Wrote {OUT_PANEL}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
