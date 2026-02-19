"""
solver/equilibrium.py
=====================
Solves the static equilibrium equation Ku = F and computes
per-member strain energies. Produces an EnergyState consumed
by the entropy module each simulation step.
"""

import numpy as np
from core.models import FrameData, EnergyState, MemberState, Load
from structure.stiffness import assemble_global_stiffness, apply_boundary_conditions, _member_length, _get_node


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
    Compute strain energy and axial force for a single member given displacements.

    Projects nodal displacements onto the member axis to get axial deformation,
    then computes strain energy as U = 0.5 * EA/L * delta^2.

    NOTE: Currently uses axial deformation only. Bending contributions
    (0.5 * EI * curvature^2) can be added later without breaking the interface.

    Args:
        member: Member to evaluate.
        u: Global displacement vector.
        frame: Used to retrieve node coordinates and length.

    Returns:
        MemberState with energy, force, deformation, and failed flag.
    """
    import numpy as np

    if member.failed:
        return MemberState(
            member_id=member.id,
            strain_energy=0.0,
            axial_force=0.0,
            deformation=0.0,
            failed=True
        )

    n_start = _get_node(frame, member.node_start)
    n_end = _get_node(frame, member.node_end)
    L = _member_length(member, frame)

    # Unit vector along member axis
    axis = np.array([
        n_end.x - n_start.x,
        n_end.y - n_start.y,
        n_end.z - n_start.z
    ]) / L

    # Displacement vectors at each node (ux, uy, uz only)
    i = member.node_start * 6
    j = member.node_end * 6
    u_start = u[i:i+3]
    u_end   = u[j:j+3]

    # Axial deformation = projection of relative displacement onto member axis
    delta = float(np.dot(u_end - u_start, axis))
    k_axial = member.E * member.A / L
    axial_force = k_axial * delta
    strain_energy = 0.5 * k_axial * delta**2

    return MemberState(
        member_id=member.id,
        strain_energy=strain_energy,
        axial_force=axial_force,
        deformation=delta,
        failed=False
    )