"""Minimal SVG plotting primitives — pure Python stdlib, no deps.

Used by the workshop figure scripts (01, 04, 06, 07) to replace
matplotlib for the small set of figure types we actually need:

    * rectangles (paintings, histogram bars, area bins)
    * polylines and filled polygons (posterior traces, delta curves)
    * text, axis ticks, legend swatches

Every helper returns an SVG fragment as a string. A figure is built by
concatenating fragments and wrapping them with `header()` / `footer()`.

The output is plain SVG 1.1 — opens in any browser, in Eye of GNOME,
Inkscape, Firefox, Chrome, Safari, Preview.app, etc.
"""

from __future__ import annotations
from xml.sax.saxutils import escape as _esc
import math


# ---------------------------------------------------------------------------
# document scaffold
# ---------------------------------------------------------------------------

def header(width: float, height: float, bg: str = "#ffffff") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.2f} {height:.2f}" '
        'font-family="Helvetica, Arial, sans-serif">\n'
        f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" fill="{bg}"/>\n'
    )


def footer() -> str:
    return "</svg>\n"


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def rect(x, y, w, h, fill, opacity=1.0, stroke=None, stroke_width=0.0):
    a = (f'<rect x="{x:.2f}" y="{y:.2f}" '
         f'width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
         f'fill="{fill}"')
    if opacity < 1.0:
        a += f' fill-opacity="{opacity:.3f}"'
    if stroke and stroke_width > 0:
        a += f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return a + "/>"


def line(x1, y1, x2, y2, stroke, stroke_width=1.0, dash=None):
    a = (f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
         f'x2="{x2:.2f}" y2="{y2:.2f}" '
         f'stroke="{stroke}" stroke-width="{stroke_width}"')
    if dash:
        a += f' stroke-dasharray="{dash}"'
    return a + "/>"


def polyline(points, stroke, stroke_width=1.0, fill="none", opacity=1.0):
    if not points:
        return ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    a = (f'<polyline points="{pts}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="{stroke_width}"')
    if opacity < 1.0:
        a += f' fill-opacity="{opacity:.3f}"'
    return a + "/>"


def polygon(points, fill, opacity=1.0, stroke="none", stroke_width=0.0):
    if not points:
        return ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    a = f'<polygon points="{pts}" fill="{fill}"'
    if opacity < 1.0:
        a += f' fill-opacity="{opacity:.3f}"'
    if stroke != "none" and stroke_width > 0:
        a += f' stroke="{stroke}" stroke-width="{stroke_width}"'
    else:
        a += ' stroke="none"'
    return a + "/>"


_BASELINE = {
    "baseline": "alphabetic",
    "alphabetic": "alphabetic",
    "middle": "central",
    "central": "central",
    "hanging": "hanging",
    "top": "hanging",
    "bottom": "alphabetic",
}


def text(x, y, s, size=10, anchor="start", va="baseline",
         color="#222", weight="normal"):
    """Anchor: start, middle, end. va: baseline, middle, hanging/top."""
    a = (f'<text x="{x:.2f}" y="{y:.2f}" '
         f'font-size="{size}" '
         f'text-anchor="{anchor}" '
         f'dominant-baseline="{_BASELINE.get(va, va)}" '
         f'fill="{color}"')
    if weight != "normal":
        a += f' font-weight="{weight}"'
    return a + ">" + _esc(str(s)) + "</text>"


# ---------------------------------------------------------------------------
# axis helpers
# ---------------------------------------------------------------------------

def x_axis(x_px, y_px, w_px, x_lo, x_hi, ticks,
           tick_len=4, label_size=9, color="#222", label_color="#222"):
    """Horizontal axis at SVG (x_px, y_px), width w_px, mapping data
    [x_lo, x_hi] to that span. `ticks` is a list of data-coordinate
    tick positions; each gets a short downward tick mark and a centered
    label below."""
    out = [line(x_px, y_px, x_px + w_px, y_px, color, 1.0)]
    for tval in ticks:
        tx = x_px + (tval - x_lo) / (x_hi - x_lo) * w_px
        out.append(line(tx, y_px, tx, y_px + tick_len, color, 1.0))
        out.append(text(tx, y_px + tick_len + label_size + 1,
                        _fmt(tval), size=label_size,
                        anchor="middle", va="hanging", color=label_color))
    return "\n".join(out)


def y_axis(x_px, y_px, h_px, y_lo, y_hi, ticks,
           tick_len=4, label_size=9, color="#222", label_color="#222"):
    """Vertical axis at SVG (x_px, y_px), height h_px, mapping data
    [y_lo, y_hi] to that span (data y_hi is at the top)."""
    out = [line(x_px, y_px, x_px, y_px + h_px, color, 1.0)]
    for tval in ticks:
        ty = y_px + (1 - (tval - y_lo) / (y_hi - y_lo)) * h_px
        out.append(line(x_px, ty, x_px - tick_len, ty, color, 1.0))
        out.append(text(x_px - tick_len - 3, ty,
                        _fmt(tval), size=label_size,
                        anchor="end", va="middle", color=label_color))
    return "\n".join(out)


def nice_ticks(lo, hi, target=6):
    """Return ~target evenly-spaced 'nice' tick values within [lo, hi]."""
    if hi <= lo:
        return [lo]
    span = hi - lo
    raw_step = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    if norm < 1.5:
        step = 1
    elif norm < 3:
        step = 2
    elif norm < 7:
        step = 5
    else:
        step = 10
    step *= mag
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + step * 1e-9:
        if abs(v - round(v)) < 1e-9:
            out.append(int(round(v)))
        else:
            out.append(round(v, 10))
        v += step
    return out


def _fmt(v):
    if isinstance(v, int):
        return str(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    # trim trailing zeros
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ---------------------------------------------------------------------------
# small statistics helpers (replacing numpy)
# ---------------------------------------------------------------------------

def quantile(xs, q):
    """Linear-interpolation quantile, matching numpy's default ('linear').
    `xs` need NOT be sorted (we sort a copy). q in [0, 1]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def rolling_mean_same(xs, win):
    """Equivalent of np.convolve(xs, ones(win)/win, mode='same'):
    uniform kernel of length `win`, centered, edges treated as zero
    (i.e. divisor is always `win`, not the count of available samples).
    This matches the original matplotlib script's smoothing."""
    n = len(xs)
    out = [0.0] * n
    half = win // 2  # for win=25, half=12, window indices i-12 .. i+12
    # np.convolve's 'same' mode, for kernel of length win, returns
    # output[i] = sum_{k=0..win-1} kernel[k] * x[i + half - k]
    # with x out of range = 0. For symmetric uniform kernel that simplifies
    # to: average of x[i - (win - 1 - half) .. i + half], inclusive,
    # zero-padded, divided by win.
    left = win - 1 - half  # for win=25 half=12 → left=12
    for i in range(n):
        a = i - left
        b = i + half  # inclusive
        s = 0.0
        for j in range(a, b + 1):
            if 0 <= j < n:
                s += xs[j]
        out[i] = s / win
    return out
