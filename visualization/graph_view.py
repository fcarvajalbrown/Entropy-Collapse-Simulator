"""
visualization/graph_view.py
============================
Renders the structural frame in 3D with members color-coded by strain energy.

Uses matplotlib's mpl_toolkits.mplot3d for an interactive rotatable view.
Members are drawn as lines; color maps to normalized strain energy (p_i),
so hot colors (red) indicate energy concentration and cool colors (blue)
indicate low energy — visually matching the entropy localization concept.

Failed members are drawn as dashed grey lines.
Nodes are drawn as scatter points, supports marked distinctly.

Consumed by main.py or called directly for step-by-step animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from core.models import FrameData, EnergyState, EntropyRecord


def plot_frame(
    frame: FrameData,
    energy_state: EnergyState,
    entropy_record: EntropyRecord,
    step: int,
    show: bool = True,
    save_path: str | None = None
) -> plt.Figure:
    """
    Render the frame at a given simulation step with energy heatmap.

    Args:
        frame: Frame geometry (nodes and members).
        energy_state: Energy state for this step (used for coloring).
        entropy_record: Entropy metrics for this step (shown in title).
        step: Current step index (shown in title).
        show: Whether to call plt.show() immediately.
        save_path: If provided, saves figure to this path instead of showing.

    Returns:
        matplotlib Figure object.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    energy_map = _build_energy_map(energy_state)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(energy_map.values(), default=1.0))
    cmap = cm.get_cmap("RdYlBu_r")

    _draw_members(ax, frame, energy_map, norm, cmap)
    _draw_nodes(ax, frame)
    _add_colorbar(fig, cmap, norm)
    _style_axes(ax, frame, step, entropy_record)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    elif show:
        plt.show()

    return fig


def plot_collapse_sequence(
    frame: FrameData,
    failed_sequence: list[int],
    show: bool = True,
    save_path: str | None = None
) -> plt.Figure:
    """
    Render the frame with failed members highlighted in red and
    surviving members in green, showing the full collapse sequence.

    Args:
        frame: Frame geometry.
        failed_sequence: Ordered list of member IDs that failed.
        show: Whether to call plt.show() immediately.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    failed_set = set(failed_sequence)

    for member in frame.members:
        n_start = _get_node(frame, member.node_start)
        n_end = _get_node(frame, member.node_end)
        xs = [n_start.x, n_end.x]
        ys = [n_start.y, n_end.y]
        zs = [n_start.z, n_end.z]

        if member.id in failed_set:
            order = failed_sequence.index(member.id) + 1
            ax.plot(xs, ys, zs, color="red", linewidth=2, linestyle="--")
            mx, my, mz = np.mean(xs), np.mean(ys), np.mean(zs)
            ax.text(mx, my, mz, str(order), color="red", fontsize=8)
        else:
            ax.plot(xs, ys, zs, color="green", linewidth=2)

    _draw_nodes(ax, frame)
    ax.set_title("Collapse Sequence (numbers = failure order)", fontsize=12)
    _set_axis_labels(ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    elif show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_members(ax, frame, energy_map, norm, cmap):
    """
    Draw all members as 3D lines colored by normalized strain energy.
    Failed members are drawn as dashed grey lines.

    Args:
        ax: matplotlib 3D axis.
        frame: Frame geometry.
        energy_map: Dict of member_id -> normalized energy p_i.
        norm: Colormap normalizer.
        cmap: Colormap instance.
    """
    for member in frame.members:
        n_start = _get_node(frame, member.node_start)
        n_end = _get_node(frame, member.node_end)
        xs = [n_start.x, n_end.x]
        ys = [n_start.y, n_end.y]
        zs = [n_start.z, n_end.z]

        if member.failed:
            ax.plot(xs, ys, zs, color="grey", linewidth=1, linestyle="--", alpha=0.4)
        else:
            p_i = energy_map.get(member.id, 0.0)
            color = cmap(norm(p_i))
            ax.plot(xs, ys, zs, color=color, linewidth=3)


def _draw_nodes(ax, frame):
    """
    Draw all nodes as scatter points. Support nodes (fixed_dofs not empty)
    are drawn as larger triangles to distinguish them visually.

    Args:
        ax: matplotlib 3D axis.
        frame: Frame geometry.
    """
    for node in frame.nodes:
        if node.fixed_dofs:
            ax.scatter(node.x, node.y, node.z, color="black", s=80, marker="^", zorder=5)
        else:
            ax.scatter(node.x, node.y, node.z, color="steelblue", s=40, zorder=5)


def _build_energy_map(energy_state: EnergyState) -> dict[int, float]:
    """
    Build a dict mapping member_id to normalized energy p_i = U_i / sum(U).

    Args:
        energy_state: Current energy state.

    Returns:
        Dict of member_id -> p_i (float in [0, 1]).
    """
    total = energy_state.total_energy
    if total == 0:
        return {ms.member_id: 0.0 for ms in energy_state.member_states}
    return {
        ms.member_id: ms.strain_energy / total
        for ms in energy_state.member_states
    }


def _add_colorbar(fig, cmap, norm):
    """
    Add a colorbar legend for the energy heatmap.

    Args:
        fig: matplotlib Figure.
        cmap: Colormap.
        norm: Normalizer.
    """
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes[0], shrink=0.5, pad=0.1)
    cbar.set_label("Normalized Strain Energy (pᵢ)", fontsize=9)


def _style_axes(ax, frame, step, entropy_record):
    """
    Set axis title, labels, and display entropy metrics in the title.

    Args:
        ax: matplotlib 3D axis.
        frame: Frame (for name).
        step: Current step index.
        entropy_record: Used to display S and dS in the title.
    """
    ax.set_title(
        f"{frame.name} — Step {step} | "
        f"S = {entropy_record.entropy:.4f} | "
        f"dS = {entropy_record.delta_entropy:+.4f}",
        fontsize=11
    )
    _set_axis_labels(ax)


def _set_axis_labels(ax):
    """Set standard X/Y/Z axis labels."""
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


def _get_node(frame, node_id):
    """Retrieve a node by ID from the frame. Raises ValueError if not found."""
    for node in frame.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Node {node_id} not found.")