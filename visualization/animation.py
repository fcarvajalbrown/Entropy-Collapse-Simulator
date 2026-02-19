"""
visualization/animation.py
===========================
Produces an animated visualization of the progressive collapse simulation.

Each frame of the animation shows two panels:
  Left:  3D structural frame with members color-coded by normalized strain
         energy (hot = high energy concentration, cool = low). Failed members
         are drawn as dashed grey lines.
  Right: Entropy curve (S vs step, dS/dt vs step) with a moving vertical
         marker tracking the current step. Collapse event is annotated when
         detected.

Output formats:
  .gif  — universal, no extra codec required (slower to render)
  .mp4  — requires ffmpeg on PATH, smaller file, higher quality

Consumed by main.py via the --animate flag.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from core.models import FrameData, SimulationResult, EnergyState, EntropyRecord
from entropy.metrics import max_entropy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_collapse(
    result: SimulationResult,
    frame: FrameData,
    output_path: str,
    fps: int = 10,
    dpi: int = 120,
) -> None:
    """
    Render and save an animation of the full progressive collapse simulation.

    Each animation frame corresponds to one simulation step. The left panel
    shows the structural frame with an energy heatmap; the right panel tracks
    the entropy and dS/dt curves with a moving marker.

    Args:
        result: Completed SimulationResult from runner.run().
        frame: The FrameData used to produce the result (geometry source).
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

    # Pre-compute entropy normalization across all steps
    entropy_norm = _compute_normalized_entropy(result)

    # Pre-compute global energy range for consistent colormap across all frames
    global_max_energy = _global_max_normalized_energy(result)

    fig, axes = _build_figure(result)
    ax_frame, ax_entropy, ax_delta = axes

    # Pre-draw entropy curves (static background, updated only by marker)
    steps = [r.step for r in result.entropy_history]
    delta_entropy = [r.delta_entropy for r in result.entropy_history]

    ax_entropy.plot(steps, entropy_norm, color="steelblue", linewidth=2, zorder=2)
    ax_entropy.set_xlim(0, max(steps) + 1)
    ax_entropy.set_ylim(0, 1.1)
    ax_entropy.set_ylabel("S / S_max", fontsize=9)
    ax_entropy.set_title("Normalized Entropy", fontsize=10)
    ax_entropy.grid(True, alpha=0.3)

    ax_delta.plot(steps, delta_entropy, color="darkorange", linewidth=2, zorder=2)
    ax_delta.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_delta.set_xlim(0, max(steps) + 1)
    ax_delta.set_ylabel("dS / dt", fontsize=9)
    ax_delta.set_xlabel("Step", fontsize=9)
    ax_delta.set_title("Entropy Rate of Change", fontsize=10)
    ax_delta.grid(True, alpha=0.3)

    if result.collapse_detected and result.collapse_step is not None:
        for ax in (ax_entropy, ax_delta):
            ax.axvline(
                result.collapse_step,
                color="red", linewidth=1.5, linestyle="--", alpha=0.7,
                label=f"Collapse (step {result.collapse_step})",
                zorder=1,
            )
        ax_entropy.legend(fontsize=8, loc="lower left")

    # Mutable marker objects, replaced each frame
    entropy_marker = [None]
    delta_marker = [None]
    cbar_container = [None]

    norm = mcolors.Normalize(vmin=0.0, vmax=max(global_max_energy, 1e-9))
    cmap = cm.get_cmap("RdYlBu_r")

    def update(step_idx: int) -> list:
        """
        Render one animation frame for the given simulation step index.

        Args:
            step_idx: Index into result.energy_history / entropy_history.

        Returns:
            List of updated artists (required by FuncAnimation).
        """
        energy_state  = result.energy_history[step_idx]
        entropy_record = result.entropy_history[step_idx]

        # --- Update frame panel ---
        ax_frame.cla()
        _draw_frame_3d(ax_frame, frame, energy_state, entropy_record, norm, cmap)

        # Rebuild colorbar after clearing axis
        if cbar_container[0] is not None:
            try:
                cbar_container[0].remove()
            except Exception:
                pass
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_container[0] = fig.colorbar(
            sm, ax=ax_frame, shrink=0.5, pad=0.08, fraction=0.03
        )
        cbar_container[0].set_label("Normalized Energy (pᵢ)", fontsize=8)

        # --- Update entropy markers ---
        if entropy_marker[0] is not None:
            try:
                entropy_marker[0].remove()
            except Exception:
                pass
        if delta_marker[0] is not None:
            try:
                delta_marker[0].remove()
            except Exception:
                pass

        entropy_marker[0] = ax_entropy.axvline(
            entropy_record.step, color="red", linewidth=1.8, alpha=0.9, zorder=3
        )
        delta_marker[0] = ax_delta.axvline(
            entropy_record.step, color="red", linewidth=1.8, alpha=0.9, zorder=3
        )

        fig.suptitle(
            f"{result.frame_name}   —   Step {step_idx + 1}/{n_steps}",
            fontsize=12, fontweight="bold"
        )

        return [ax_frame]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_steps,
        interval=1000 // fps,
        repeat=False,
        blit=False,
    )

    writer = _get_writer(ext, fps)
    print(f"  Rendering {n_steps} frames to {output_path} ...")
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def _build_figure(result: SimulationResult):
    """
    Create the figure and axes layout for the animation.

    Layout: 3D frame on the left (wide), entropy + dS stacked on the right.

    Args:
        result: Used to set axis titles.

    Returns:
        (fig, (ax_frame, ax_entropy, ax_delta)) tuple.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.2, 1], hspace=0.4, wspace=0.35)

    ax_frame   = fig.add_subplot(gs[:, 0], projection="3d")
    ax_entropy = fig.add_subplot(gs[0, 1])
    ax_delta   = fig.add_subplot(gs[1, 1])

    return fig, (ax_frame, ax_entropy, ax_delta)


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def _draw_frame_3d(
    ax,
    frame: FrameData,
    energy_state: EnergyState,
    entropy_record: EntropyRecord,
    norm: mcolors.Normalize,
    cmap,
) -> None:
    """
    Draw the full 3D frame on ax for the given energy state.

    Members are colored by normalized strain energy pᵢ. Failed members
    are drawn as dashed grey lines. Support nodes are marked as triangles.

    Args:
        ax: matplotlib 3D axis (already cleared by caller).
        frame: Frame geometry.
        energy_state: Energy state for this step.
        entropy_record: Entropy metrics (used in title).
        norm: Colormap normalizer (consistent across all frames).
        cmap: Colormap instance.
    """
    energy_map = _build_energy_map(energy_state)

    for member in frame.members:
        n_start = _get_node(frame, member.node_start)
        n_end   = _get_node(frame, member.node_end)
        xs = [n_start.x, n_end.x]
        ys = [n_start.y, n_end.y]
        zs = [n_start.z, n_end.z]

        if member.failed:
            ax.plot(xs, ys, zs, color="grey", linewidth=1, linestyle="--", alpha=0.35)
        else:
            p_i   = energy_map.get(member.id, 0.0)
            color = cmap(norm(p_i))
            ax.plot(xs, ys, zs, color=color, linewidth=3)

    for node in frame.nodes:
        if node.fixed_dofs:
            ax.scatter(node.x, node.y, node.z, color="black", s=80, marker="^", zorder=5)
        else:
            ax.scatter(node.x, node.y, node.z, color="steelblue", s=35, zorder=5)

    failed_count = sum(1 for m in frame.members if m.failed)
    ax.set_title(
        f"S = {entropy_record.entropy:.4f}  |  dS = {entropy_record.delta_entropy:+.4f}"
        f"  |  Failed: {failed_count}/{len(frame.members)}",
        fontsize=9,
    )
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _build_energy_map(energy_state: EnergyState) -> dict[int, float]:
    """
    Map member_id -> normalized energy pᵢ = U_i / sum(U).

    Args:
        energy_state: Current energy state.

    Returns:
        Dict of member_id -> pᵢ in [0, 1].
    """
    total = energy_state.total_energy
    if total <= 0:
        return {ms.member_id: 0.0 for ms in energy_state.member_states}
    return {
        ms.member_id: ms.strain_energy / total
        for ms in energy_state.member_states
    }


def _global_max_normalized_energy(result: SimulationResult) -> float:
    """
    Find the maximum pᵢ across all steps and all members.

    Used to set a consistent colormap range so the heatmap does not
    rescale between frames.

    Args:
        result: Full simulation result.

    Returns:
        Maximum normalized energy fraction observed in any step.
    """
    global_max = 0.0
    for es in result.energy_history:
        if es.total_energy <= 0:
            continue
        for ms in es.member_states:
            p_i = ms.strain_energy / es.total_energy
            if p_i > global_max:
                global_max = p_i
    return global_max


def _compute_normalized_entropy(result: SimulationResult) -> list[float]:
    """
    Normalize entropy history to [0, 1] range (S / S_max) per step.

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
    else:  # mp4
        try:
            return animation.FFMpegWriter(fps=fps, bitrate=1800)
        except Exception as e:
            raise RuntimeError(
                "ffmpeg is required for MP4 output. Install from https://ffmpeg.org/"
            ) from e


def _get_node(frame: FrameData, node_id: int):
    """
    Retrieve a node by ID from the frame.

    Raises:
        ValueError: If node_id is not found.
    """
    for node in frame.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Node {node_id} not found in frame.")
