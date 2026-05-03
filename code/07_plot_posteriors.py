#!/usr/bin/env python3
"""Part 3 deeper dive — posterior trace and panel-similarity contrast.

Reads:
  solutions/ancestry_posteriors.tsv   (per-window forward-backward posteriors)
  solutions/ancestry_segments.tsv     (Viterbi segments)
  data/identity_chr12_admix.tsv       (raw per-window identity)
  data/populations.tsv                (panel labels)
  data/ground_truth_tracts.tsv        (truth)

Writes:
  figures/posterior_trace.svg
  figures/panel_similarity.svg

Plot A — posterior trace for CHIM_03:
  top   : ground-truth ancestry bar
  middle: P(AFR) per window from forward-backward (the "marginal posterior")
  bottom: Viterbi-decoded segments (the most likely state path)

Plot B — panel-similarity contrast for CHIM_03:
  Δ = max-AFR similarity − max-EUR similarity per window, then a
  250-kb rolling mean of Δ. The two raw similarity series are visually
  indistinguishable at this scale, so the difference is plotted directly.

Pure stdlib — no numpy / matplotlib required.
"""

from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svgplot as svg

POST = Path("solutions/ancestry_posteriors.tsv")
SEGS = Path("solutions/ancestry_segments.tsv")
SIM = Path("data/identity_chr12_admix.tsv")
POPS = Path("data/populations.tsv")
TRUTH = Path("data/ground_truth_tracts.tsv")

OUT_TRACE = Path("figures/posterior_trace.svg")
OUT_PANEL = Path("figures/panel_similarity.svg")

QUERY = "CHIM_03#1"
COLORS = {"AFR": "#cc6600", "EUR": "#3366cc"}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

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
    """For each window, return (start, end, max_AFR_identity, max_EUR_identity)."""
    panel = load_panel_membership()
    afr, eur = panel["AFR"], panel["EUR"]
    best = defaultdict(lambda: [-1.0, -1.0])
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


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def draw_truth_bar(out, segs, x_to_px, y_top, y_bot):
    h = y_bot - y_top
    for s, e, anc in segs:
        x0 = x_to_px(s / 1e6)
        x1 = x_to_px(e / 1e6)
        out.append(svg.rect(x0, y_top, x1 - x0, h,
                            COLORS.get(anc, "#999999"),
                            stroke="white", stroke_width=0.4))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    if not POST.exists() or not SEGS.exists():
        print("ERROR: run bash code/03_run_ancestry.sh first.", file=sys.stderr)
        return 1

    OUT_TRACE.parent.mkdir(parents=True, exist_ok=True)

    truth = load_truth(QUERY)

    # ====================================================================
    # Plot A: posterior trace (3 panels stacked, common x range 50-120 Mb)
    # ====================================================================
    post = load_posteriors(QUERY)
    segs = load_segments(QUERY)

    starts_mb = [(s + e) / 2 / 1e6 for s, e, *_ in post]
    p_afr = [row[2] for row in post]

    W, H = 1500, 720
    L, R = 90, 30
    plot_w = W - L - R
    ax_lo, ax_hi = 50.0, 120.0

    def x_to_px(mb): return L + (mb - ax_lo) / (ax_hi - ax_lo) * plot_w

    # vertical layout: title (top), truth bar, posterior, Viterbi bar, x-axis
    y_title_baseline = 30
    truth_top, truth_bot = 65, 95
    post_top, post_bot = 110, 580
    vit_top, vit_bot = 595, 625
    x_axis_y = vit_bot + 6

    out = [svg.header(W, H)]

    # title
    out.append(svg.text(
        W / 2, y_title_baseline,
        f"{QUERY} — ground truth (top), forward-backward P(AFR) per window (middle), Viterbi segments (bottom)",
        size=13, anchor="middle", va="baseline", color="#222"))

    # legend (top-right, inline with title row)
    leg_x0 = W - R - 200
    leg_y = y_title_baseline - 12
    sw, sh = 18, 12
    out.append(svg.rect(leg_x0, leg_y, sw, sh, COLORS["AFR"]))
    out.append(svg.text(leg_x0 + sw + 6, leg_y + sh / 2, "AFR",
                        size=11, anchor="start", va="middle"))
    out.append(svg.rect(leg_x0 + 70, leg_y, sw, sh, COLORS["EUR"]))
    out.append(svg.text(leg_x0 + 70 + sw + 6, leg_y + sh / 2, "EUR",
                        size=11, anchor="start", va="middle"))

    # row labels (truth, Viterbi)
    out.append(svg.text(L - 10, (truth_top + truth_bot) / 2, "truth",
                        size=11, anchor="end", va="middle"))
    out.append(svg.text(L - 10, (vit_top + vit_bot) / 2, "Viterbi",
                        size=11, anchor="end", va="middle"))

    # truth bar
    draw_truth_bar(out, truth, x_to_px, truth_top, truth_bot)

    # posterior panel — y axis [0, 1], filled regions: below curve in AFR color,
    # above curve up to 1 in EUR color
    p_h = post_bot - post_top
    def y_post(p): return post_top + (1 - p) * p_h  # data y in [0,1]

    # build the polygon for the area below the curve (P(AFR)): from
    # (x[0], 0) along the curve to (x[-1], 0), close.
    if starts_mb:
        pts_below = [(x_to_px(starts_mb[0]), y_post(0))]
        for x, p in zip(starts_mb, p_afr):
            pts_below.append((x_to_px(x), y_post(p)))
        pts_below.append((x_to_px(starts_mb[-1]), y_post(0)))
        out.append(svg.polygon(pts_below, fill=COLORS["AFR"], opacity=0.35))

        # area above the curve up to 1
        pts_above = [(x_to_px(starts_mb[0]), y_post(1))]
        for x, p in zip(starts_mb, p_afr):
            pts_above.append((x_to_px(x), y_post(p)))
        pts_above.append((x_to_px(starts_mb[-1]), y_post(1)))
        out.append(svg.polygon(pts_above, fill=COLORS["EUR"], opacity=0.35))

        # the dark line itself
        line_pts = [(x_to_px(x), y_post(p)) for x, p in zip(starts_mb, p_afr)]
        out.append(svg.polyline(line_pts, stroke="#222222", stroke_width=0.6))

    # 0.5 dashed reference line
    out.append(svg.line(L, y_post(0.5), L + plot_w, y_post(0.5),
                        "#888888", stroke_width=0.6, dash="4,3"))

    # posterior y-axis
    p_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out.append(svg.y_axis(L, post_top, p_h, 0, 1, p_ticks))
    out.append(
        f'<text x="{L - 50}" y="{(post_top + post_bot) / 2:.2f}" '
        f'font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 {L - 50} {(post_top + post_bot) / 2:.2f})" '
        f'fill="#222">P(AFR | data)</text>'
    )

    # Viterbi bar
    draw_truth_bar(out, segs, x_to_px, vit_top, vit_bot)

    # x-axis
    x_ticks = [50, 60, 70, 80, 90, 100, 110, 120]
    out.append(svg.x_axis(L, x_axis_y, plot_w, ax_lo, ax_hi, x_ticks,
                          tick_len=5, label_size=11))
    out.append(svg.text(L + plot_w / 2, x_axis_y + 35,
                        "chr12 position (Mb)",
                        size=13, anchor="middle", va="baseline"))

    out.append(svg.footer())
    OUT_TRACE.write_text("".join(out))
    print(f"Wrote {OUT_TRACE}")

    # ====================================================================
    # Plot B: panel similarity Δ
    # ====================================================================
    sim_rows = panel_similarity_per_window(QUERY)
    pos = [(s + e) / 2 / 1e6 for s, e, *_ in sim_rows]
    sim_afr = [row[2] for row in sim_rows]
    sim_eur = [row[3] for row in sim_rows]

    # delta per window; treat NaN as 0 for the smoothing input (matches
    # nan-propagation behavior of the previous numpy version close enough
    # for visualization — and the input data has no nan in practice for
    # this region).
    delta = []
    for a, e in zip(sim_afr, sim_eur):
        if a != a or e != e:  # NaN
            delta.append(0.0)
        else:
            delta.append(a - e)

    win = 25
    delta_smooth = svg.rolling_mean_same(delta, win)

    # for the title, median |Δ| over the raw (non-smoothed) windows,
    # excluding NaNs
    abs_delta = [abs(a - e) for a, e in zip(sim_afr, sim_eur)
                 if a == a and e == e]
    med_abs_delta = median(abs_delta) if abs_delta else float("nan")

    # vertical layout
    W2, H2 = 1500, 660
    L2, R2 = 110, 60
    plot_w2 = W2 - L2 - R2
    ax_lo2, ax_hi2 = 50.0, 120.0

    def x_to_px2(mb): return L2 + (mb - ax_lo2) / (ax_hi2 - ax_lo2) * plot_w2

    title_y1 = 30
    title_y2 = 50
    truth_top2, truth_bot2 = 75, 105
    delta_top, delta_bot = 130, 590
    x_axis_y2 = delta_bot + 6

    # y-axis for delta — clipped to ±0.0015 (≈ 15× the median |Δ|) so the
    # typical per-window contrast is legible. A handful of outlier windows
    # exceed this range; they're drawn at the boundary and flagged with a
    # small triangular marker so the reader knows the bar is truncated.
    d_lo, d_hi = -0.0015, 0.0015
    d_h = delta_bot - delta_top

    def y_delta(d):
        d_clip = max(d_lo, min(d_hi, d))
        return delta_top + (1 - (d_clip - d_lo) / (d_hi - d_lo)) * d_h

    out = [svg.header(W2, H2)]

    # two-line title
    out.append(svg.text(
        W2 / 2, title_y1,
        f"{QUERY} — raw input to the HMM emissions",
        size=13, anchor="middle", va="baseline", color="#222"))
    out.append(svg.text(
        W2 / 2, title_y2,
        f"Δ = max-AFR − max-EUR per window, 250-kb rolling mean.  "
        f"median |Δ| = {med_abs_delta:.4f}; y-axis clipped to ±0.0015 "
        f"(▲▼ mark out-of-range windows).",
        size=11, anchor="middle", va="baseline", color="#222"))

    # truth bar
    draw_truth_bar(out, truth, x_to_px2, truth_top2, truth_bot2)

    # delta panel: per-window rectangles from y=0 to y=delta_smooth[i].
    n = len(pos)
    if n > 0:
        for i in range(n):
            d = delta_smooth[i]
            if d == 0:
                continue
            x_left = x_to_px2(pos[0]) if i == 0 else x_to_px2((pos[i - 1] + pos[i]) / 2)
            x_right = x_to_px2(pos[-1]) if i == n - 1 else x_to_px2((pos[i] + pos[i + 1]) / 2)
            color = COLORS["AFR"] if d > 0 else COLORS["EUR"]
            y0 = y_delta(0)
            y1 = y_delta(d)
            top = min(y0, y1)
            bot = max(y0, y1)
            out.append(svg.rect(x_left, top, x_right - x_left, bot - top,
                                color, opacity=0.55))
            # out-of-range marker: small triangle pointing in the direction
            # of the actual (clipped) value, drawn just inside the boundary
            if d > d_hi or d < d_lo:
                cx = (x_left + x_right) / 2
                if d > d_hi:
                    yt = delta_top + 2
                    pts = [(cx - 4, yt + 6), (cx + 4, yt + 6), (cx, yt)]
                else:
                    yb = delta_bot - 2
                    pts = [(cx - 4, yb - 6), (cx + 4, yb - 6), (cx, yb)]
                out.append(svg.polygon(pts, fill=color, opacity=0.9))

        # smooth black curve overlay (also clipped)
        line_pts = [(x_to_px2(p), y_delta(d)) for p, d in zip(pos, delta_smooth)]
        out.append(svg.polyline(line_pts, stroke="#222222", stroke_width=0.6))

    # zero baseline
    out.append(svg.line(L2, y_delta(0), L2 + plot_w2, y_delta(0),
                        "#888888", stroke_width=0.6))

    # delta y-axis
    d_ticks = svg.nice_ticks(d_lo, d_hi, target=7)
    out.append(svg.y_axis(L2, delta_top, d_h, d_lo, d_hi, d_ticks))
    out.append(
        f'<text x="{L2 - 70}" y="{(delta_top + delta_bot) / 2:.2f}" '
        f'font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 {L2 - 70} {(delta_top + delta_bot) / 2:.2f})" '
        f'fill="#222">Δ identity = max(AFR) − max(EUR)</text>'
    )

    # x-axis
    x_ticks2 = [50, 60, 70, 80, 90, 100, 110, 120]
    out.append(svg.x_axis(L2, x_axis_y2, plot_w2, ax_lo2, ax_hi2, x_ticks2,
                          tick_len=5, label_size=11))
    out.append(svg.text(L2 + plot_w2 / 2, x_axis_y2 + 35,
                        "chr12 position (Mb)",
                        size=13, anchor="middle", va="baseline"))

    # legend (bottom-right)
    leg_x = L2 + plot_w2 - 220
    leg_y = delta_bot - 40
    sw, sh = 18, 12
    out.append(svg.rect(leg_x, leg_y, sw, sh, COLORS["AFR"], opacity=0.55))
    out.append(svg.text(leg_x + sw + 6, leg_y + sh / 2,
                        "more AFR-like (Δ > 0)",
                        size=11, anchor="start", va="middle"))
    out.append(svg.rect(leg_x, leg_y + sh + 6, sw, sh, COLORS["EUR"], opacity=0.55))
    out.append(svg.text(leg_x + sw + 6, leg_y + sh + 6 + sh / 2,
                        "more EUR-like (Δ < 0)",
                        size=11, anchor="start", va="middle"))

    out.append(svg.footer())
    OUT_PANEL.write_text("".join(out))
    print(f"Wrote {OUT_PANEL}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
