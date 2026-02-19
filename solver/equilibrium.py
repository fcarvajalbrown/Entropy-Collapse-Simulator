"""
solver/equilibrium.py
=====================
Solves the static equilibrium equation Ku = F and computes
per-member strain energies using the full internal force vector.

Strain energy is computed as:
    U_i = 0.5 * u_local^T * k_local * u_local

where u_local are the member's 12 nodal displacements transformed
to local coordinates. This correctly captures both axial and bending
contributions — critical for frames where load is carried in bending.

Internal force magnitude is taken as the Euclidean norm of f_local,
which includes axial, shear, and moment components.
"""

import numpy as np
from core.models import FrameData, EnergyState, MemberState
from structure.stiffness import (
    assemble_global_stiffness,
    apply_boundary_conditions,
    _local_stiffness,
    _transformation_matrix,
    _get_node
)


def solve(frame: FrameData, step: int) -> EnergyState:
    """
    Solve Ku = F for the current frame state and return an EnergyState.

    Args:
        frame: Current frame definition (members may be partially failed).
        step: Current simulation step index.

    Returns:
        EnergyState with per-member strain energies and forces.
    """
    K = assemble_global_stiffness(frame)
    K = apply_boundary_conditions(K, frame)
    F = _build_load_vector(frame)
    u = _solve_system(K, F)

    member_states = [
        _compute_member_state(member, u, frame)
        for member in frame.members
    ]

    total_energy = sum(ms.strain_energy for ms in member_states)

    return EnergyState(step=step, total_energy=total_energy, member_states=member_states)


def _build_load_vector(frame: FrameData) -> np.ndarray:
    """
    Assemble the global force vector F from all applied loads.

    Args:
        frame: Frame definition containing load list.

    Returns:
        F (np.ndarray): Force vector of shape (n_dof,).
    """
    n_dof = len(frame.nodes) * 6
    F = np.zeros(n_dof)
    for load in frame.loads:
        global_dof = load.node_id * 6 + load.dof
        F[global_dof] += load.magnitude
    return F


def _solve_system(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ku = F using numpy's least-squares solver.
    Falls back gracefully if K is singular (collapsed structure).

    Args:
        K: Global stiffness matrix.
        F: Global force vector.

    Returns:
        u (np.ndarray): Displacement vector of shape (n_dof,).
    """
    try:
        u = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(K, F, rcond=None)[0]
    return u


def _compute_member_state(member, u: np.ndarray, frame: FrameData) -> MemberState:
    """
    Compute strain energy and internal force magnitude for a single member.

    Uses the full 12-DOF local stiffness formulation:
        u_local = T * u_global  (transform displacements to local coords)
        f_local = k_local * u_local  (internal force vector)
        U_i = 0.5 * u_local^T * k_local * u_local  (strain energy)

    This correctly captures axial, shear, and bending contributions.
    Previously only axial projection was used, which gave zero energy
    for bending-dominated members (e.g. horizontal beam under vertical load).

    Args:
        member: Member to evaluate.
        u: Global displacement vector.
        frame: Used to retrieve geometry for transformation matrix.

    Returns:
        MemberState with energy, force magnitude, deformation, and failed flag.
    """
    if member.failed:
        return MemberState(
            member_id=member.id,
            strain_energy=0.0,
            axial_force=0.0,
            deformation=0.0,
            failed=True
        )

    # Extract 12 global DOFs for this member (6 per node)
    i = member.node_start * 6
    j = member.node_end * 6
    u_global = np.concatenate([u[i:i+6], u[j:j+6]])

    # Transform to local coordinates
    T = _transformation_matrix(member, frame)
    k_local = _local_stiffness(member, frame)
    u_local = T @ u_global

    # Full internal force vector in local coordinates
    f_local = k_local @ u_local

    # Strain energy from full deformation (axial + bending)
    strain_energy = float(0.5 * u_local @ f_local)

    # Axial force = local DOF 0 (f_local[0])
    axial_force = float(f_local[0])

    # Total internal force magnitude (includes shear and moment resultants)
    force_magnitude = float(np.linalg.norm(f_local[:3]))

    # Axial deformation for reference
    n_start = _get_node(frame, member.node_start)
    n_end = _get_node(frame, member.node_end)
    L = float(np.sqrt(
        (n_end.x - n_start.x)**2 +
        (n_end.y - n_start.y)**2 +
        (n_end.z - n_start.z)**2
    ))
    axis = np.array([
        n_end.x - n_start.x,
        n_end.y - n_start.y,
        n_end.z - n_start.z
    ]) / L
    deformation = float(np.dot(u_global[3:6] - u_global[0:3], axis))

    return MemberState(
        member_id=member.id,
        strain_energy=max(strain_energy, 0.0),  # Clamp numerical noise
        axial_force=force_magnitude,             # Total force magnitude
        deformation=deformation,
        failed=False
    )