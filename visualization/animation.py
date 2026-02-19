"""
visualization/animation.py
===========================
Produces an animated visualization of the progressive collapse simulation.

Three stacked subplots, all sharing the x-axis (simulation step):
  1. Normalized structural entropy S/S_max
  2. Entropy rate of change dS/dt
  3. Gini localization index

A black vertical marker sweeps across all three plots as the animation
progresses. Collapse step is marked with a dashed red line. Member
failure events are marked with grey dotted lines.

The 3D frame diagram is intentionally excluded — it does not deform
visually and caused zoom/scaling issues in animation.

Output formats:
  .gif  — universal, no extra codec required
  .mp4  — requires ffmpeg on PATH, smaller file, higher quality

Consumed by main.py via the --animate flag.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

from core.models import FrameData, SimulationResult
from entropy.metrics import max_entropy


def animate_collapse(
    result: SimulationResult,
    frame: FrameData,
    output_path: str,
    fps: int = 10,
    dpi: int = 120,
) -> None:
    """
    Render and save an animated entropy analysis for a completed simulation.

    Each animation frame advances a step marker across three subplots:
    normalized entropy, dS/dt, and Gini localization index. The full curves
    are drawn as a static background; only the marker moves.

    Args:
        result: Completed SimulationResult from runner.run().
        frame: The FrameData used to produce the result (unused for drawing,
               kept in signature for API consistency).
        output_path: File path for the output animation. Extension determines
                     format: ".gif" uses Pillow writer, ".mp4" uses ffmpeg.
        fps: Frames per second for the animation (default 10).
        dpi: Output resolution in dots per inch (default 120).

    Raises:
        ValueError: If output_path does not end in ".gif" or ".mp4".
    """
    ext = output_path.rsplit(".", 1)[-1].lower()
    if ext not in ("gif", "mp4"):
        raise ValueError(f"Unsupported output format '.{ext}'. Use '.gif' or '.mp4'.")

    n_steps = len(result.energy_history)
    if n_steps == 0:
        raise ValueError("SimulationResult has no steps to animate.")

    steps         = [r.step for r in result.entropy_history]
    entropy_norm  = _compute_normalized_entropy(result)
    delta_entropy = [r.delta_entropy for r in result.entropy_history]
    gini          = [_gini(r.energy_distribution) for r in result.entropy_history]

    fig, (ax_s, ax_ds, ax_gini) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Entropy Analysis — {result.frame_name}", fontsize=13, fontweight="bold")

    x_max = max(steps) + 1

    # --- Static curves ---
    ax_s.plot(steps, entropy_norm, color="steelblue", linewidth=2)
    ax_s.set_xlim(0, x_max)
    ax_s.set_ylim(0, 1.1)
    ax_s.axhline(1.0, color="steelblue", linestyle=":", linewidth=1, alpha=0.5)
    ax_s.set_ylabel("S / S_max", fontsize=10)
    ax_s.set_title("Normalized Structural Entropy", fontsize=10)
    ax_s.grid(True, alpha=0.3)

    ax_ds.plot(steps, delta_entropy, color="darkorange", linewidth=2)
    ax_ds.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_ds.set_xlim(0, x_max)
    ax_ds.set_ylabel("dS / dt", fontsize=10)
    ax_ds.set_title("Entropy Rate of Change", fontsize=10)
    ax_ds.grid(True, alpha=0.3)

    ax_gini.plot(steps, gini, color="firebrick", linewidth=2)
    ax_gini.set_xlim(0, x_max)
    ax_gini.set_ylim(0, 1.05)
    ax_gini.set_ylabel("Gini Index", fontsize=10)
    ax_gini.set_xlabel("Simulation Step", fontsize=10)
    ax_gini.set_title("Energy Localization Index", fontsize=10)
    ax_gini.grid(True, alpha=0.3)

    # --- Static annotations: collapse and failure event lines ---
    for ax in (ax_s, ax_ds, ax_gini):
        if result.failed_sequence:
            failure_steps = np.linspace(0, n_steps - 1, len(result.failed_sequence), dtype=int)
            for fs in failure_steps:
                ax.axvline(fs, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        if result.collapse_detected and result.collapse_step is not None:
            ax.axvline(
                result.collapse_step,
                color="red", linewidth=1.8, linestyle="--", alpha=0.8,
            )

    patches = []
    if result.collapse_detected:
        patches.append(mpatches.Patch(color="red", label=f"Collapse (step {result.collapse_step})"))
    if result.failed_sequence:
        patches.append(mpatches.Patch(color="grey", label="Member failure"))
    if patches:
        ax_s.legend(handles=patches, fontsize=8, loc="lower left")

    # --- Moving step marker ---
    markers = [
        ax_s.axvline(0, color="black", linewidth=1.5, alpha=0.7),
        ax_ds.axvline(0, color="black", linewidth=1.5, alpha=0.7),
        ax_gini.axvline(0, color="black", linewidth=1.5, alpha=0.7),
    ]
    step_label = ax_s.text(
        0.01, 0.95, "", transform=ax_s.transAxes,
        fontsize=9, verticalalignment="top", color="black"
    )

    plt.tight_layout()

    def update(step_idx: int) -> list:
        """
        Advance the step marker to the given simulation step.

        Args:
            step_idx: Index into entropy_history.

        Returns:
            List of updated artists.
        """
        x = result.entropy_history[step_idx].step
        for marker in markers:
            marker.set_xdata([x, x])
        step_label.set_text(f"Step {step_idx + 1} / {n_steps}")
        return markers + [step_label]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=1000 // fps,
        repeat=False,
        blit=True,
    )

    writer = _get_writer(ext, fps)
    print(f"  Rendering {n_steps} frames to {output_path} ...")
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_normalized_entropy(result: SimulationResult) -> list[float]:
    """
    Normalize entropy history to [0, 1] using S / S_max per step.

    Args:
        result: Full simulation result.

    Returns:
        List of normalized entropy values, one per step.
    """
    normalized = []
    for es, er in zip(result.energy_history, result.entropy_history):
        n_active = sum(1 for ms in es.member_states if not ms.failed)
        s_max = max_entropy(n_active)
        normalized.append(er.entropy / s_max if s_max > 0 else 0.0)
    return normalized


def _gini(energy_distribution: list) -> float:
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


def _get_writer(ext: str, fps: int):
    """
    Return an appropriate matplotlib animation writer for the given format.

    Args:
        ext: File extension — "gif" or "mp4".
        fps: Frames per second.

    Returns:
        matplotlib writer instance.

    Raises:
        RuntimeError: If the required writer is not available.
    """
    if ext == "gif":
        try:
            return animation.PillowWriter(fps=fps)
        except Exception as e:
            raise RuntimeError(
                "Pillow is required for GIF output. Install with: pip install Pillow"
            ) from e
    else:
        try:
            return animation.FFMpegWriter(fps=fps, bitrate=1800)
        except Exception as e:
            raise RuntimeError(
                "ffmpeg is required for MP4 output. Install from https://ffmpeg.org/"
            ) from e