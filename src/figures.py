"""Regenerate all four manuscript figures at 400 dpi (PNG) and as vector PDF.

Figures 1 and 2 are schematics drawn in matplotlib. Figures 3 and 4 plot the
measured results read from `results/summary_full.csv`. Nothing here is a
generative image: every element is placed by code.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .config import DATASET_ORDER, FIGURES, RESULTS

NAVY, TEAL, AMBER, CORAL = "#1b3a5c", "#2a9d8f", "#e3a82b", "#d8572a"
SLATE, PAPER, INK = "#7a8699", "#f6f3ee", "#16202b"
DS_LABEL = {"cfpb": "CFPB", "banking77": "Banking77",
            "ledgar": "LEDGAR", "clinc150": "CLINC150"}
DS_COLOUR = {"cfpb": CORAL, "banking77": NAVY, "ledgar": AMBER, "clinc150": TEAL}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.edgecolor": "#43505f", "axes.linewidth": 0.8,
    "xtick.color": "#43505f", "ytick.color": "#43505f",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "savefig.facecolor": "white",
})


def _save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", stem)


def _rbox(ax, x, y, w, h, fc, ec=None, lw=1.0, r=0.018, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=lw, zorder=z,
                                mutation_aspect=1))


def _arrow(ax, p1, p2, colour=SLATE, lw=1.1, rad=0.0, z=3, ms=7):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=colour, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


# --------------------------------------------------------------------------
def figure1_concept() -> None:
    """Confidence routed cascade under a retention floor."""
    fig = plt.figure(figsize=(3.05, 3.25), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _rbox(ax, 0.04, 0.855, 0.92, 0.105, PAPER, "#d9d2c6", 0.9)
    ax.text(0.5, 0.932, "Repetitive classification workload", ha="center",
            va="center", fontsize=7.4, color=INK, fontweight="bold")
    rng = np.random.default_rng(4)
    for i in range(34):
        ax.add_patch(Rectangle((0.075 + i * 0.0255, 0.872), 0.016, 0.034,
                               fc=SLATE, ec="none", alpha=rng.uniform(.35, .8), zorder=3))
    _arrow(ax, (0.5, 0.853), (0.5, 0.80), INK, 1.2)

    _rbox(ax, 0.10, 0.625, 0.80, 0.168, NAVY, NAVY)
    ax.text(0.5, 0.757, "Small model scores every label", ha="center", va="center",
            fontsize=7.3, color="white", fontweight="bold")
    bars = [0.30, 0.52, 0.95, 0.44, 0.26, 0.18, 0.12]
    for i, b in enumerate(bars):
        ax.add_patch(Rectangle((0.235 + i * 0.047, 0.655), 0.034, b * 0.072,
                               fc=AMBER if b == max(bars) else "#5b7899",
                               ec="none", zorder=4))
    ax.text(0.80, 0.676, "one\nforward\npass", ha="center", va="center",
            fontsize=5.9, color="#c3d2e2", linespacing=1.25)
    _arrow(ax, (0.5, 0.620), (0.5, 0.573), INK, 1.2)

    _rbox(ax, 0.14, 0.455, 0.72, 0.113, "#fdf6e3", AMBER, 1.1)
    ax.text(0.5, 0.538, "Confidence gate", ha="center", va="center",
            fontsize=7.3, color=INK, fontweight="bold")
    gx = np.linspace(0, 1, 160)
    gy = np.exp(-((gx - 0.60) ** 2) / 0.060)
    gy = 0.040 * gy / gy.max()
    ax.plot(0.20 + gx * 0.60, 0.472 + gy, color=SLATE, lw=0.9, zorder=4)
    ax.fill_between(0.20 + gx * 0.60, 0.472, 0.472 + gy, where=(gx <= 0.40),
                    color=CORAL, alpha=0.55, zorder=3)
    ax.fill_between(0.20 + gx * 0.60, 0.472, 0.472 + gy, where=(gx >= 0.40),
                    color=TEAL, alpha=0.55, zorder=3)
    ax.plot([0.20 + 0.40 * 0.60] * 2, [0.468, 0.519], color=INK, lw=1.0,
            ls=(0, (2.2, 1.6)), zorder=5)
    ax.text(0.20 + 0.40 * 0.60, 0.449, "threshold", ha="center", va="top",
            fontsize=5.8, color=INK)
    _arrow(ax, (0.33, 0.452), (0.235, 0.372), CORAL, 1.2, rad=0.12)
    _arrow(ax, (0.67, 0.452), (0.765, 0.372), TEAL, 1.2, rad=-0.12)

    _rbox(ax, 0.055, 0.237, 0.365, 0.133, "#fdeae3", CORAL, 1.0)
    ax.text(0.2375, 0.337, "Escalate", ha="center", va="center", fontsize=7.2,
            color=CORAL, fontweight="bold")
    ax.text(0.2375, 0.283, "large model\nanswers", ha="center", va="center",
            fontsize=6.4, color=INK, linespacing=1.3)
    _rbox(ax, 0.580, 0.237, 0.365, 0.133, "#e4f4f1", TEAL, 1.0)
    ax.text(0.7625, 0.337, "Keep", ha="center", va="center", fontsize=7.2,
            color="#1d7268", fontweight="bold")
    ax.text(0.7625, 0.283, "small model\nanswer stands", ha="center", va="center",
            fontsize=6.4, color=INK, linespacing=1.3)
    _arrow(ax, (0.2375, 0.234), (0.37, 0.178), CORAL, 1.1, rad=-0.10)
    _arrow(ax, (0.7625, 0.234), (0.63, 0.178), TEAL, 1.1, rad=0.10)

    _rbox(ax, 0.075, 0.040, 0.85, 0.133, INK, INK)
    ax.text(0.5, 0.139, "Retention floor fixes the operating point", ha="center",
            va="center", fontsize=7.2, color="white", fontweight="bold")
    ax.text(0.5, 0.080, "keep 95% of large-model accuracy,\n"
            "then spend as little as the data allows", ha="center", va="center",
            fontsize=6.3, color="#b9c6d6", linespacing=1.35)
    _save(fig, "figure1_concept")


# --------------------------------------------------------------------------
def figure2_pipeline() -> None:
    """Scoring and routing pipeline applied to every item."""
    fig = plt.figure(figsize=(3.05, 4.35), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _rbox(ax, 0.03, 0.838, 0.94, 0.144, PAPER, "#d9d2c6", 0.9)
    ax.text(0.065, 0.962, "A", fontsize=8.2, color=NAVY, fontweight="bold", va="center")
    ax.text(0.125, 0.962, "Prompt assembly", fontsize=7.4, color=INK,
            fontweight="bold", va="center")
    _rbox(ax, 0.065, 0.850, 0.40, 0.086, "white", SLATE, 0.8, r=0.012)
    ax.text(0.265, 0.921, "label list", ha="center", va="center", fontsize=6.2,
            color=NAVY, fontweight="bold")
    for i in range(4):
        ax.add_patch(Rectangle((0.088, 0.900 - i * 0.0125), 0.30 - i * 0.03, 0.006,
                               fc=SLATE, ec="none", alpha=0.55, zorder=5))
    _rbox(ax, 0.515, 0.850, 0.40, 0.086, "white", SLATE, 0.8, r=0.012)
    ax.text(0.715, 0.921, "input text", ha="center", va="center", fontsize=6.2,
            color=NAVY, fontweight="bold")
    for i in range(4):
        ax.add_patch(Rectangle((0.538, 0.900 - i * 0.0125), 0.355, 0.006,
                               fc=SLATE, ec="none", alpha=0.35, zorder=5))
    _arrow(ax, (0.5, 0.835), (0.5, 0.808), INK, 1.1)

    _rbox(ax, 0.03, 0.620, 0.94, 0.185, "#eef3f8", "#c3d2e2", 0.9)
    ax.text(0.065, 0.780, "B", fontsize=8.2, color=NAVY, fontweight="bold", va="center")
    ax.text(0.125, 0.780, "Shared prefix, scored once", fontsize=7.4, color=INK,
            fontweight="bold", va="center")
    _rbox(ax, 0.065, 0.690, 0.20, 0.060, NAVY, NAVY, r=0.012)
    ax.text(0.165, 0.720, "prefix\nKV cache", ha="center", va="center",
            fontsize=6.0, color="white", linespacing=1.3)
    for i, lab in enumerate(["label 1", "label 2", "label K"]):
        yy = 0.742 - i * 0.040
        _arrow(ax, (0.268, 0.720), (0.44, yy), "#5b7899", 0.85, rad=0.18 - i * 0.14, ms=5)
        _rbox(ax, 0.445, yy - 0.0155, 0.30, 0.031, "white", "#9fb2c7", 0.7, r=0.010)
        ax.text(0.595, yy, lab, ha="center", va="center", fontsize=5.8, color=INK)
        ax.text(0.775, yy, "log p", ha="left", va="center", fontsize=5.8,
                color="#5b7899", style="italic")
    ax.text(0.165, 0.655, "cache restored\nafter each label", ha="center",
            va="center", fontsize=5.6, color="#5b7899", linespacing=1.3)
    _arrow(ax, (0.5, 0.617), (0.5, 0.583), INK, 1.1)

    _rbox(ax, 0.03, 0.330, 0.94, 0.250, "#fdf6e3", "#ecd9a6", 0.9)
    ax.text(0.065, 0.555, "C", fontsize=8.2, color=NAVY, fontweight="bold", va="center")
    ax.text(0.125, 0.555, "Distribution and free signals", fontsize=7.4, color=INK,
            fontweight="bold", va="center")
    vals = [0.95, 0.60, 0.33, 0.26, 0.21, 0.16, 0.13, 0.10, 0.07, 0.05]
    for i, v in enumerate(vals):
        c = AMBER if i == 0 else ("#d9b45e" if i == 1 else "#cfc3a4")
        ax.add_patch(Rectangle((0.075 + i * 0.044, 0.432), 0.032, v * 0.085,
                               fc=c, ec="none", zorder=4))
    ax.plot([0.091, 0.091], [0.513, 0.535], color=INK, lw=0.7, zorder=5)
    ax.plot([0.135, 0.135], [0.483, 0.535], color=INK, lw=0.7, zorder=5)
    ax.annotate("", xy=(0.091, 0.528), xytext=(0.135, 0.528),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.7,
                                mutation_scale=5), zorder=6)
    ax.text(0.155, 0.528, "margin", fontsize=5.7, color=INK, va="center")
    ax.text(0.545, 0.470, "length-normalised\nscores, softmax\nover K labels",
            fontsize=5.8, color="#8a7b4e", va="center", linespacing=1.35)
    for i, (nm, sub) in enumerate([("top probability", "p(1)"),
                                   ("margin", "p(1) - p(2)"),
                                   ("negative entropy", "1 - H / log K")]):
        xx = 0.065 + i * 0.300
        _rbox(ax, xx, 0.345, 0.275, 0.058, "white", AMBER, 0.8, r=0.012)
        ax.text(xx + 0.1375, 0.385, nm, ha="center", va="center", fontsize=5.9,
                color=INK, fontweight="bold")
        ax.text(xx + 0.1375, 0.361, sub, ha="center", va="center", fontsize=5.7,
                color="#8a7b4e", style="italic")
    _arrow(ax, (0.5, 0.327), (0.5, 0.293), INK, 1.1)

    _rbox(ax, 0.03, 0.038, 0.94, 0.252, "#e4f4f1", "#a8ddd5", 0.9)
    ax.text(0.065, 0.265, "D", fontsize=8.2, color="#1d7268", fontweight="bold", va="center")
    ax.text(0.125, 0.265, "Threshold under a retention floor", fontsize=7.4,
            color=INK, fontweight="bold", va="center")
    axi = fig.add_axes([0.115, 0.075, 0.44, 0.135])
    r = np.linspace(0, 1, 120)
    acc = 0.30 + 0.46 * (1 - np.exp(-3.4 * r))
    acc = acc / acc.max() * 0.757
    cost = 0.12 + 0.88 * r
    axi.plot(cost, acc, color=TEAL, lw=1.4)
    axi.plot(cost, 0.296 + (0.757 - 0.296) * r, color=SLATE, lw=0.9, ls=(0, (2.5, 1.8)))
    axi.axhline(0.95 * 0.757, color=CORAL, lw=0.9, ls=(0, (1.6, 1.4)))
    j = int(np.argmax(acc >= 0.95 * 0.757))
    axi.plot([cost[j]], [acc[j]], "o", ms=3.4, color=INK, zorder=5)
    axi.set_xlabel("cost", fontsize=5.9, labelpad=1.2)
    axi.set_ylabel("accuracy", fontsize=5.9, labelpad=1.6)
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_linewidth(0.7)
    ax.text(0.585, 0.228, "retention floor", fontsize=5.9, color=CORAL, fontweight="bold")
    ax.text(0.585, 0.200, "routed frontier", fontsize=5.9, color="#1d7268", fontweight="bold")
    ax.text(0.585, 0.172, "random escalation", fontsize=5.9, color=SLATE)
    ax.text(0.585, 0.132, "cheapest run that\nclears the floor", fontsize=5.8,
            color=INK, linespacing=1.35, va="top")
    ax.text(0.585, 0.082, "threshold fitted\nout of fold", fontsize=5.7,
            color="#1d7268", style="italic", linespacing=1.3, va="top")
    _save(fig, "figure2_pipeline")


# --------------------------------------------------------------------------
def _summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "summary_full.csv")


def figure3_retention() -> None:
    """Escalation rate and cost saving at the 95% retention floor."""
    S = _summary()
    fig, axes = plt.subplots(2, 1, figsize=(3.05, 3.45), dpi=400, sharex=True)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.935, bottom=0.115, hspace=0.22)
    idx, w = np.arange(len(DATASET_ORDER)), 0.36
    smalls = sorted(S.small.unique())

    def col(ds, sm, field):
        return S[(S.dataset == ds) & (S.small == sm)][field].iloc[0]

    ax = axes[0]
    ax.bar(idx - w / 2, [col(d, smalls[0], "esc_at_95") for d in DATASET_ORDER], w,
           color="white", edgecolor=NAVY, lw=1.1, label=smalls[0])
    ax.bar(idx + w / 2, [col(d, smalls[1], "esc_at_95") for d in DATASET_ORDER], w,
           color=NAVY, edgecolor=NAVY, lw=1.1, label=smalls[1])
    ax.set_ylabel("escalation rate", fontsize=7.0, labelpad=2)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handletextpad=0.4, columnspacing=1.2)

    ax = axes[1]
    ax.bar(idx - w / 2, [col(d, smalls[0], "save_at_95") for d in DATASET_ORDER], w,
           color="white", edgecolor=TEAL, lw=1.1)
    ax.bar(idx + w / 2, [col(d, smalls[1], "save_at_95") for d in DATASET_ORDER], w,
           color=TEAL, edgecolor=TEAL, lw=1.1)
    ax.set_ylabel("cost saving", fontsize=7.0, labelpad=2)
    ax.set_ylim(0, 0.80); ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_xticks(idx)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASET_ORDER], fontsize=6.6)

    for a in axes:
        a.grid(axis="y", color="#e3e6ea", lw=0.6); a.set_axisbelow(True)
        a.tick_params(labelsize=6.3, length=2.2, pad=1.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    _save(fig, "figure3_retention")


def figure4_scaling() -> None:
    """Cost saving at the 95% retention floor against small-model accuracy."""
    S = _summary()
    fig, ax = plt.subplots(figsize=(3.05, 2.55), dpi=400)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.145)

    x, y = S.small_acc.values, S.save_at_95.values
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(0.24, 0.77, 50)
    ax.plot(xs, a + b * xs, color=SLATE, lw=0.9, ls=(0, (3, 2)), zorder=1)
    r = np.corrcoef(x, y)[0, 1]

    smalls = sorted(S.small.unique())
    for ds in DATASET_ORDER:
        sub = S[S.dataset == ds].sort_values("small_acc")
        ax.plot(sub.small_acc, sub.save_at_95, color=DS_COLOUR[ds], lw=0.8,
                alpha=0.5, zorder=2)
        for _, row in sub.iterrows():
            big = row.small == smalls[1]
            ax.scatter(row.small_acc, row.save_at_95, s=46 if big else 38,
                       marker="s" if big else "o",
                       facecolor=DS_COLOUR[ds] if big else "white",
                       edgecolor=DS_COLOUR[ds], linewidth=1.3, zorder=4)
        top = sub.iloc[-1]
        off = {"clinc150": (-6, -11), "banking77": (6, -8),
               "ledgar": (6, -2), "cfpb": (7, -4)}[ds]
        ax.annotate(DS_LABEL[ds], (top.small_acc, top.save_at_95), fontsize=6.0,
                    color=DS_COLOUR[ds], fontweight="bold", xytext=off,
                    textcoords="offset points",
                    ha="right" if ds == "clinc150" else "left")

    ax.text(0.755, 0.055, f"r = {r:.2f}", fontsize=7.0, color=SLATE, ha="right")
    ax.set_xlabel("small-model accuracy", fontsize=7.2, labelpad=2)
    ax.set_ylabel("cost saving at 95% retention", fontsize=7.2, labelpad=2)
    ax.set_xlim(0.23, 0.77); ax.set_ylim(0.0, 0.80)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 0.81, 0.2)])
    ax.tick_params(labelsize=6.4, length=2.2, pad=1.6)
    ax.grid(axis="y", color="#e3e6ea", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec=SLATE, mew=1.2, ms=4.6,
               label=smalls[0]),
        Line2D([], [], marker="s", ls="", mfc=SLATE, mec=SLATE, ms=4.6,
               label=smalls[1])],
        fontsize=6.0, frameon=False, loc="upper left", handletextpad=0.4,
        borderpad=0.1, labelspacing=0.3)
    _save(fig, "figure4_scaling")


def build_all() -> None:
    figure1_concept()
    figure2_pipeline()
    figure3_retention()
    figure4_scaling()


if __name__ == "__main__":
    build_all()
