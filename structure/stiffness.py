"""
structure/stiffness.py
======================
Assembles the global stiffness matrix K from a FrameData object.
Supports 2D and 3D Euler-Bernoulli beam elements.
Output is consumed by solver/equilibrium.py.
"""

import numpy as np
from core.models import FrameData, Member, Node


def assemble_global_stiffness(frame: FrameData) -> np.ndarray:
    """
    Build the global stiffness matrix K for the entire frame.

    Iterates over all non-failed members, computes their local stiffness,
    transforms to global coordinates, and assembles into K.

    Args:
        frame: Full frame definition including nodes and members.

    Returns:
        K (np.ndarray): Global stiffness matrix of shape (n_dof, n_dof).
    """
    n_dof = len(frame.nodes) * 6
    K = np.zeros((n_dof, n_dof))

    for member in frame.members:
        if member.failed:
            continue
        k_local = _local_stiffness(member, frame)
        T = _transformation_matrix(member, frame)
        k_global = T.T @ k_local @ T
        dofs = _member_dofs(member)
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += k_global[i, j]

    return K


def apply_boundary_conditions(K: np.ndarray, frame: FrameData) -> np.ndarray:
    """
    Zero out rows and columns for fixed DOFs, set diagonal to 1.
    Modifies K in-place to enforce boundary conditions via the penalty method.

    Args:
        K: Global stiffness matrix (modified in-place).
        frame: Frame definition containing node boundary conditions.

    Returns:
        K (np.ndarray): Modified stiffness matrix.
    """
    for node in frame.nodes:
        for dof in node.fixed_dofs:
            global_dof = node.id * 6 + dof
            K[global_dof, :] = 0
            K[:, global_dof] = 0
            K[global_dof, global_dof] = 1
    return K


def _local_stiffness(member: Member, frame: FrameData) -> np.ndarray:
    """
    Compute the 12x12 local stiffness matrix for an Euler-Bernoulli beam element.

    Args:
        member: Member with material and section properties.
        frame: Used to retrieve node coordinates for length calculation.

    Returns:
        k_local (np.ndarray): 12x12 local stiffness matrix.
    """
    L = _member_length(member, frame)
    E, A, I = member.E, member.A, member.I
    k_local = np.zeros((12, 12))

    # Axial terms
    k_local[0, 0] = k_local[6, 6] = E * A / L
    k_local[0, 6] = k_local[6, 0] = -E * A / L

    # Bending terms (strong axis)
    k_local[1, 1] = k_local[7, 7] = 12 * E * I / L**3
    k_local[1, 7] = k_local[7, 1] = -12 * E * I / L**3
    k_local[1, 5] = k_local[5, 1] = 6 * E * I / L**2
    k_local[1, 11] = k_local[11, 1] = 6 * E * I / L**2
    k_local[7, 5] = k_local[5, 7] = -6 * E * I / L**2
    k_local[7, 11] = k_local[11, 7] = -6 * E * I / L**2
    k_local[5, 5] = k_local[11, 11] = 4 * E * I / L
    k_local[5, 11] = k_local[11, 5] = 2 * E * I / L

    return k_local


def _transformation_matrix(member: Member, frame: FrameData) -> np.ndarray:
    """
    Build the 12x12 transformation matrix T from local to global coordinates.
    Uses the direction cosines of the member axis.

    Args:
        member: Member connecting two nodes.
        frame: Used to retrieve node coordinates.

    Returns:
        T (np.ndarray): 12x12 transformation matrix.
    """
    n_start = _get_node(frame, member.node_start)
    n_end = _get_node(frame, member.node_end)

    dx = n_end.x - n_start.x
    dy = n_end.y - n_start.y
    dz = n_end.z - n_start.z
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    cx, cy, cz = dx / L, dy / L, dz / L

    # 3x3 rotation block
    # Reference vector for local y-axis
    ref = np.array([0, 0, 1]) if abs(cz) < 0.99 else np.array([0, 1, 0])
    local_x = np.array([cx, cy, cz])
    local_z = np.cross(local_x, ref)
    local_z /= np.linalg.norm(local_z)
    local_y = np.cross(local_z, local_x)

    R = np.array([local_x, local_y, local_z])

    T = np.zeros((12, 12))
    for i in range(4):
        T[i*3:(i+1)*3, i*3:(i+1)*3] = R

    return T


def _member_length(member: Member, frame: FrameData) -> float:
    """Compute Euclidean length of a member from its two node coordinates."""
    n_start = _get_node(frame, member.node_start)
    n_end = _get_node(frame, member.node_end)
    return np.sqrt(
        (n_end.x - n_start.x)**2 +
        (n_end.y - n_start.y)**2 +
        (n_end.z - n_start.z)**2
    )


def _member_dofs(member: Member) -> list:
    """Return the 12 global DOF indices for a member's two nodes (6 DOFs each)."""
    return (
        list(range(member.node_start * 6, member.node_start * 6 + 6)) +
        list(range(member.node_end * 6, member.node_end * 6 + 6))
    )


def _get_node(frame: FrameData, node_id: int) -> Node:
    """Retrieve a node by ID from the frame. Raises ValueError if not found."""
    for node in frame.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"Node {node_id} not found in frame.")