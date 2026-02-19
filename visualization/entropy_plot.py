"""
visualization/entropy_plot.py
==============================
Plots the structural entropy evolution over the full simulation.

Three subplots in one figure:
  1. S vs step        — entropy curve with collapse event marker
  2. dS/dt vs step    — rate of change, collapse spike clearly visible
  3. Gini index       — localization index as a second collapse indicator

Together these give a complete picture of the energy concentration
process leading to collapse — suitable for publication figures.

Consumed by main.py after runner.run() completes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from core.models import SimulationResult
from entropy.metrics import max_entropy


def plot_entropy(
    result: SimulationResult,
    show: bool = True,
    save_path: str | None = None
) -> plt.Figure:
    """
    Render the full entropy analysis figure for a completed simulation.

    Three vertically stacked subplots:
      - Top:    Entropy S vs step (normalized to [0,1])
      - Middle: dS/dt vs step (raw, with zero reference line)
      - Bottom: Gini localization index vs step

    Collapse step is marked with a vertical red dashed line on all subplots.
    Member failure events are marked with grey vertical lines.

    Args:
        result: Completed SimulationResult from runner.run().
        show: Whether to call plt.show() immediately.
        save_path: If provided, saves figure to this path.

    Returns:
        matplotlib Figure object.
    """
    steps = [r.step for r in result.entropy_history]
    entropy = [r.entropy for r in result.entropy_history]
    delta_entropy = [r.delta_entropy for r in result.entropy_history]
    gini = [_gini(r.energy_distribution) for r in result.entropy_history]

    # Normalize entropy to [0, 1]
    n_members_per_step = _active_member_counts(result)
    s_max_per_step = [max_entropy(n) for n in n_members_per_step]
    entropy_norm = [
        s / s_max if s_max > 0 else 0.0
        for s, s_max in zip(entropy, s_max_per_step)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Entropy Analysis — {result.frame_name}",
        fontsize=13, fontweight="bold"
    )

    _plot_entropy_curve(axes[0], steps, entropy_norm)
    _plot_delta_entropy(axes[1], steps, delta_entropy)
    _plot_gini(axes[2], steps, gini)

    # Mark collapse and failure events on all axes
    for ax in axes:
        _mark_failures(ax, result)
        if result.collapse_detected and result.collapse_step is not None:
            _mark_collapse(ax, result.collapse_step)

    _add_legend(axes[0], result.collapse_detected)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    elif show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Subplot renderers
# ---------------------------------------------------------------------------

def _plot_entropy_curve(ax, steps, entropy_norm):
    """
    Plot normalized structural entropy S/S_max vs step.

    Args:
        ax: matplotlib axis.
        steps: List of step indices.
        entropy_norm: Normalized entropy values in [0, 1].
    """
    ax.plot(steps, entropy_norm, color="steelblue", linewidth=2)
    ax.set_ylabel("S / S_max", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="steelblue", linestyle=":", linewidth=1, alpha=0.5,
               label="Uniform distribution (S_max)")
    ax.set_title("Normalized Structural Entropy", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_delta_entropy(ax, steps, delta_entropy):
    """
    Plot dS/dt (entropy rate of change) vs step.
    A large negative spike here is the collapse signal.

    Args:
        ax: matplotlib axis.
        steps: List of step indices.
        delta_entropy: dS values per step.
    """
    ax.plot(steps, delta_entropy, color="darkorange", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("dS / dt", fontsize=10)
    ax.set_title("Entropy Rate of Change", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_gini(ax, steps, gini):
    """
    Plot the Gini localization index vs step.
    Rises toward 1.0 as energy concentrates before collapse.

    Args:
        ax: matplotlib axis.
        steps: List of step indices.
        gini: Gini coefficient per step.
    """
    ax.plot(steps, gini, color="firebrick", linewidth=2)
    ax.set_ylabel("Gini Index", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Simulation Step", fontsize=10)
    ax.set_title("Energy Localization Index", fontsize=10)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _mark_collapse(ax, collapse_step: int):
    """
    Draw a vertical red dashed line at the detected collapse step.

    Args:
        ax: matplotlib axis.
        collapse_step: Step index of detected collapse.
    """
    ax.axvline(collapse_step, color="red", linewidth=1.8,
               linestyle="--", alpha=0.85, label=f"Collapse (step {collapse_step})")


def _mark_failures(ax, result: SimulationResult):
    """
    Draw thin grey vertical lines at steps where member failures occurred.
    Failure steps are approximated as evenly distributed across failed_sequence.

    Args:
        ax: matplotlib axis.
        result: Simulation result with failed_sequence.
    """
    if not result.failed_sequence:
        return
    total_steps = len(result.entropy_history)
    n_failures = len(result.failed_sequence)
    failure_steps = np.linspace(0, total_steps - 1, n_failures, dtype=int)
    for fs in failure_steps:
        ax.axvline(fs, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)


def _add_legend(ax, collapse_detected: bool):
    """
    Add a legend to the top subplot indicating collapse status.

    Args:
        ax: Top matplotlib axis.
        collapse_detected: Whether collapse was detected in the run.
    """
    status = "Collapse Detected" if collapse_detected else "No Collapse"
    color = "red" if collapse_detected else "green"
    patch = mpatches.Patch(color=color, label=status)
    ax.legend(handles=[patch], loc="lower left", fontsize=9)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _gini(energy_distribution: list[tuple[int, float]]) -> float:
    """
    Compute the Gini coefficient from an energy distribution list.

    Args:
        energy_distribution: List of (member_id, p_i) tuples.

    Returns:
        Gini coefficient in [0, 1].
    """
    if not energy_distribution:
        return 0.0
    values = np.array([p for _, p in energy_distribution], dtype=float)
    values = np.sort(values)
    n = len(values)
    if values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)


def _active_member_counts(result: SimulationResult) -> list[int]:
    """
    Count non-failed members at each step from the energy history.

    Args:
        result: Full simulation result.

    Returns:
        List of active member counts per step.
    """
    return [
        sum(1 for ms in es.member_states if not ms.failed)
        for es in result.energy_history
    ]