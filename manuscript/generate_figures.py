"""
manuscript/generate_figures.py
===============================
Regenerate every figure used in the manuscript from live analysis runs, so all
reported numbers are reproducible. Figures are written to manuscript/figures/.

    python manuscript/generate_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Rectangle, Circle

from structure.frames import frame_2d_simple, frame_building_2d, frame_pratt_bridge, frame_vogel_six_storey
from entropy import robustness as rb
from analysis import criteria as C
from analysis import importance as imp
from analysis import parametric as par
from analysis import design_variants as dv

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TRUSS_STEP = 0.3   # load-factor step that drives the redundant truss to failure


def fig_entropy_trajectory():
    """Normalized entropy and compliance vs load step (truss bridge)."""
    hist, _ = C.run_with_metrics(frame_pratt_bridge.build(), max_steps=40,
                                 load_factor_start=1.0, load_factor_step=TRUSS_STEP)
    steps = [m.step for m in hist]
    H = [m.entropy_norm for m in hist]
    compliance = [m.total_energy / m.load_factor ** 2 for m in hist]
    first_fail = next((m.step for m in hist if m.n_failed > 0), None)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(steps, H, color="firebrick", marker="o", ms=3, label="Normalized entropy H")
    ax1.set_xlabel("Load step")
    ax1.set_ylabel("Normalized entropy H", color="firebrick")
    ax1.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(steps, compliance, color="steelblue", ls="--", label="Compliance U/lambda^2")
    ax2.set_ylabel("Compliance U / lambda^2 (J)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    if first_fail is not None:
        ax1.axvline(first_fail, color="black", ls=":", lw=1)
        ax1.annotate("first member failure", xy=(first_fail, 0.5),
                     xytext=(first_fail - 12, 0.30),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax1.set_title("Entropy and compliance during progressive collapse (truss bridge)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_entropy_trajectory.png"), dpi=160)
    plt.close(fig)


def fig_criteria_comparison():
    """First-trigger load factor per criterion (truss bridge)."""
    cmp = C.compare(frame_pratt_bridge.build(), max_steps=40,
                    load_factor_start=1.0, load_factor_step=TRUSS_STEP)
    order = ["displacement", "dcr", "energy", "entropy"]
    labels = ["Displacement", "DCR", "Energy", "Entropy"]
    lf = [cmp.history[cmp.first_trigger[k]].load_factor
          if cmp.first_trigger[k] is not None else 0.0 for k in order]
    colors = ["#888888", "#888888", "#888888", "firebrick"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, lf, color=colors)
    ax.set_ylabel("Load factor at first trigger")
    ax.set_title("Collapse-criteria first-trigger comparison (truss bridge)")
    for b, v in zip(bars, lf):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.text(0.99, 0.02, "grey = needs calibrated threshold; red = calibration-free",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_criteria_comparison.png"), dpi=160)
    plt.close(fig)


def fig_robustness_ranking():
    """Per-member entropy drop for the moment frame and the truss bridge."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, mod, title in (
        (axes[0], frame_building_2d, "Moment frame"),
        (axes[1], frame_pratt_bridge, "Truss bridge"),
    ):
        rep = rb.analyze(mod.build())
        ranking = rep.ranking()
        ids = [str(i) for i, _ in ranking]
        drops = [d for _, d in ranking]
        ax.bar(range(len(ids)), drops, color="teal")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=6, rotation=90)
        ax.set_xlabel("Member id (ranked)")
        ax.set_ylabel(r"Entropy drop on removal $\Delta H_k$")
        ax.set_title(f"{title}: R_S = {rep.robustness_index:.3f}")
    fig.suptitle("Entropy-based member criticality (alternate load path)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_robustness_ranking.png"), dpi=160)
    plt.close(fig)


def fig_redundancy_sweep():
    """R_S and mechanism fraction vs degree of static indeterminacy."""
    pts = par.redundancy_sweep()
    dsi = [p.dsi for p in pts]
    rs = [p.robustness_index for p in pts]
    mech = [p.mechanism_fraction * 100 for p in pts]
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(dsi, rs, color="teal", marker="o", label="R_S")
    ax1.set_xlabel("Degree of static indeterminacy (counter-diagonals added)")
    ax1.set_ylabel("Entropy Robustness Index R_S", color="teal")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_ylim(0, 1.0)
    ax2 = ax1.twinx()
    ax2.plot(dsi, mech, color="firebrick", marker="s", ls="--", label="mechanism %")
    ax2.set_ylabel("Single-loss mechanisms (%)", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_title("R_S tracks redundancy (truss bridge, 0-6 counter-diagonals)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_redundancy_sweep.png"), dpi=160)
    plt.close(fig)


def fig_importance_scatter():
    """dH_k vs compliance importance I_k, with Spearman rho, for both frames."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mod, title in (
        (axes[0], frame_building_2d, "Moment frame"),
        (axes[1], frame_pratt_bridge, "Truss bridge"),
    ):
        c = imp.compare(mod.build())
        ids = list(c.entropy_drop.keys())
        x = [c.compliance_importance[i] for i in ids]
        y = [c.entropy_drop[i] for i in ids]
        # Drop non-finite (mechanism) points from the scatter; note them in title.
        xs, ys = zip(*[(a, b) for a, b in zip(x, y) if a != float("inf")]) if any(
            a != float("inf") for a in x) else ([], [])
        ax.scatter(xs, ys, color="darkorange", s=18)
        ax.set_xlabel("Compliance importance I_k = dU/U_0")
        ax.set_ylabel(r"Entropy drop $\Delta H_k$")
        ax.set_title(f"{title}: Spearman rho = {c.spearman_rho:+.2f}")
    fig.suptitle("Entropy criticality vs compliance importance (they are not the same)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_importance_scatter.png"), dpi=160)
    plt.close(fig)


def fig_importance_ensemble():
    """Spearman rho (dH_k vs compliance I_k) across an ensemble of frames."""
    rows = par.importance_ensemble()
    labels = [r.label for r in rows]
    rhos = [r.spearman_rho for r in rows]
    colors = ["teal" if r >= 0 else "firebrick" for r in rhos]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(range(len(rhos)), rhos, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(r"Spearman rho ($\Delta H_k$ vs I_k)")
    ax.set_ylim(-1, 1)
    ax.set_title("Entropy criticality vs compliance importance across frames "
                 "(rho far from 1, often negative)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_importance_ensemble.png"), dpi=160)
    plt.close(fig)


def fig_alp_agreement():
    """
    Paper 2: entropy column criticality dH_k vs code-style post-removal max DCR
    for the Vogel building; strong agreement means R_S triages the columns that
    the alternate-load-path check flags.
    """
    c = imp.compare_column_alp(frame_vogel_six_storey.build())
    cols = c.column_ids
    x = [c.entropy_drop[i] for i in cols]
    y = [c.alp_severity[i] for i in cols]
    top2 = c.entropy_rank[:2]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, xi, yi in zip(cols, x, y):
        crit = i in top2
        ax.scatter(xi, yi, s=58 if crit else 32,
                   color="firebrick" if crit else "steelblue",
                   zorder=3 if crit else 2, edgecolor="black", linewidth=0.4)
    # Label the two most-critical columns jointly, below-left with a connector.
    cx = sum(c.entropy_drop[i] for i in top2) / 2
    cy = sum(c.alp_severity[i] for i in top2) / 2
    ax.annotate(f"most critical:\ncols {top2[0]} & {top2[1]}",
                xy=(cx, cy), xytext=(cx - 0.12, cy - 1.6), fontsize=8.5,
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Entropy criticality  dH_k  (one linear solve)")
    ax.set_ylabel("Code ALP severity: post-removal max DCR")
    ax.set_title(f"R_S column triage vs alternate-load-path severity\n"
                 f"(Vogel frame, Spearman rho = {c.spearman_rho:+.2f})")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_alp_agreement.png"), dpi=160)
    plt.close(fig)


def fig_design_variants():
    """
    Paper 2: R_S distinguishes the fixed-base Vogel building from a pinned-base
    variant; the pinned design admits single-column-loss mechanisms at the base
    columns, which R_S detects as a drop in robustness.
    """
    study = dv.vogel_base_fixity_study()
    variants = sorted(study.variants, key=lambda v: v.robustness_index, reverse=True)
    labels = [v.label.replace(" (as-built)", "\n(as-built)") for v in variants]
    rs = [v.robustness_index for v in variants]
    mech = [v.mechanism_fraction for v in variants]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = ["seagreen", "indianred"]
    bars = ax.bar(labels, rs, color=colors[:len(variants)], edgecolor="black", width=0.55)
    for b, v, mf in zip(bars, variants, mech):
        ax.text(b.get_x() + b.get_width() / 2, v.robustness_index + 0.012,
                f"R_S = {v.robustness_index:.3f}", ha="center", fontsize=9, fontweight="bold")
        note = (f"{mf:.0%} single-loss\nmechanisms"
                + (f"\n(cols {v.mechanism_columns})" if v.mechanism_columns else ""))
        ax.text(b.get_x() + b.get_width() / 2, v.robustness_index / 2, note,
                ha="center", va="center", fontsize=8, color="white")
    ax.set_ylabel("Entropy Robustness Index  R_S")
    ax.set_ylim(0, 0.9)
    ax.set_title("R_S detects the less-redundant design (Vogel building: base fixity)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_design_variants.png"), dpi=160)
    plt.close(fig)


def _support_kind(node, frame):
    fixed = set(node.fixed_dofs)
    if not {0, 1} & fixed:
        return None
    has_frame_member = any(
        m.kind == "frame" and node.id in (m.node_start, m.node_end) for m in frame.members
    )
    if {0, 1} <= fixed and 5 in fixed and has_frame_member:
        return "fixed"
    if {0, 1} <= fixed:
        return "pin"
    return "roller"


def _draw_support(ax, x, y, kind, size):
    if kind == "fixed":
        ax.add_patch(Rectangle((x - size, y - size * 0.6), 2 * size, size * 0.6,
                               facecolor="0.6", edgecolor="black", lw=0.8, zorder=1))
        return
    ax.add_patch(Polygon([(x, y), (x - size * 0.7, y - size), (x + size * 0.7, y - size)],
                         closed=True, facecolor="white", edgecolor="black", lw=1.0, zorder=1))
    if kind == "roller":
        for dx in (-size * 0.35, size * 0.35):
            ax.add_patch(Circle((x + dx, y - size * 1.2), size * 0.18,
                                facecolor="white", edgecolor="black", lw=0.8, zorder=1))


def _draw_structure(ax, frame, title, size, arrow, loads_below=False):
    nodes = {n.id: n for n in frame.nodes}
    xs = sorted({n.x for n in frame.nodes})
    ys = sorted({n.y for n in frame.nodes})
    for m in frame.members:
        a, b = nodes[m.node_start], nodes[m.node_end]
        ax.plot([a.x, b.x], [a.y, b.y], color="black", lw=1.6, zorder=2)
        t = 0.5 if (a.x == b.x or a.y == b.y) else 0.3
        ax.text(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), str(m.id), fontsize=7,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="teal", lw=0.6))
    for n in frame.nodes:
        ax.plot(n.x, n.y, "o", color="black", ms=2.5, zorder=3)
        kind = _support_kind(n, frame)
        if kind:
            _draw_support(ax, n.x, n.y, kind, size)
    for load in frame.loads:
        n = nodes[load.node_id]
        column_above = any(
            {m.node_start, m.node_end} == {n.id, o.id} and o.x == n.x and o.y > n.y
            for m in frame.members for o in frame.nodes
        )
        label = f"{abs(load.magnitude) / 1000:.0f} kN"
        if loads_below:
            x = n.x
            if _support_kind(n, frame):
                x += -1.6 * size if n.x == xs[0] else 1.6 * size
            tip, tail = (x, n.y - arrow), (x, n.y)
            ax.text(x, tip[1] - 0.1 * arrow, label, fontsize=6.5, color="firebrick",
                    ha="center", va="top", zorder=5)
        else:
            x = n.x + (0.3 if column_above else 0.0)
            tip, tail = (x, n.y), (x, n.y + arrow)
            ax.text(x + 0.1, tail[1], label, fontsize=6.5, color="firebrick",
                    ha="left", va="bottom", zorder=5)
        ax.annotate("", xy=tip, xytext=tail,
                    arrowprops=dict(arrowstyle="-|>", color="firebrick", lw=1.0), zorder=5)
        ax.plot([x, x], [n.y - 1.6 * arrow, n.y + 1.6 * arrow], alpha=0.0)
    low = ys[0] - (2.0 * arrow if loads_below else 0.0)
    y_dim = low - 2.4 * size
    for x0, x1 in zip(xs, xs[1:]):
        _dimension(ax, (x0, y_dim), (x1, y_dim), f"{x1 - x0:g} m", "below", size)
    x_dim = xs[-1] + 2.6 * size
    for y0, y1 in zip(ys, ys[1:]):
        _dimension(ax, (x_dim, y0), (x_dim, y1), f"{y1 - y0:g} m", "right", size)
    ax.set_title(title, fontsize=9, loc="left", pad=6)
    ax.set_aspect("equal")
    ax.axis("off")


def _dimension(ax, start, end, label, side, size):
    (x0, y0), (x1, y1) = start, end
    ax.plot([x0, x1], [y0, y1], color="0.35", lw=0.7)
    tick = 0.35 * size
    for x, y in (start, end):
        if side == "below":
            ax.plot([x, x], [y - tick, y + tick], color="0.35", lw=0.7)
        else:
            ax.plot([x - tick, x + tick], [y, y], color="0.35", lw=0.7)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if side == "below":
        ax.text(mx, my - tick, label, fontsize=7, color="0.25", ha="center", va="top")
    else:
        ax.text(mx + tick, my, label, fontsize=7, color="0.25", ha="left", va="center")


def fig_benchmark_geometry():
    fig = plt.figure(figsize=(12, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.7, 3], width_ratios=[1, 2.2])
    ax_beam = fig.add_subplot(grid[0, :])
    ax_frame = fig.add_subplot(grid[1, 0])
    ax_truss = fig.add_subplot(grid[1, 1])
    _draw_structure(ax_beam, frame_2d_simple.build(),
                    "(a) Single-span beam, pinned at both ends",
                    size=0.3, arrow=0.8)
    _draw_structure(ax_frame, frame_building_2d.build(),
                    "(b) Moment frame, fixed bases",
                    size=0.5, arrow=1.0)
    _draw_structure(ax_truss, frame_pratt_bridge.build(),
                    "(c) Truss bridge, pin (left) and roller (right)",
                    size=0.5, arrow=1.0, loads_below=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_benchmark_geometry.png"), dpi=200)
    plt.close(fig)


FIGURES = {
    "benchmark_geometry": lambda: fig_benchmark_geometry(),
    "entropy_trajectory": lambda: fig_entropy_trajectory(),
    "criteria_comparison": lambda: fig_criteria_comparison(),
    "robustness_ranking": lambda: fig_robustness_ranking(),
    "redundancy_sweep": lambda: fig_redundancy_sweep(),
    "importance_scatter": lambda: fig_importance_scatter(),
    "importance_ensemble": lambda: fig_importance_ensemble(),
    "alp_agreement": lambda: fig_alp_agreement(),
    "design_variants": lambda: fig_design_variants(),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate manuscript figures from live runs.")
    parser.add_argument("names", nargs="*", help="figures to render, from: " + ", ".join(FIGURES) + " (default: all)")
    names = parser.parse_args().names or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        parser.error("unknown figure(s): " + ", ".join(unknown))
    for name in names:
        FIGURES[name]()
        print(f"Wrote fig_{name}.png")
    print(f"Figures written to {FIG_DIR}")
